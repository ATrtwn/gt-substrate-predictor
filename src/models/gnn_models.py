import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, summary, GATv2Conv, GCNConv, GATConv, GINConv, GINEConv, SAGEConv, TransformerConv, GatedGraphConv 
from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.data import Data, Batch
from collections import OrderedDict

from egnn_pytorch import EGNN_Sparse, EGNN_Network
# Optional utility for aggregating edge attributes into node features
try:
    from torch_scatter import scatter_mean
except Exception:
    scatter_mean = None



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1,use_residual=True , dropout=0.5, concat=True, layer_name="GATv2"):
        super(ConvBlock, self).__init__()

        self.layer_name = layer_name
        self.concat = concat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=dropout)
        self.use_residual = use_residual

        # GNN Layer
        if layer_name == "GAT":
            self.conv = GATConv(in_channels, out_channels, heads=heads, concat=concat)
            conv_out_channels = out_channels * heads if concat else out_channels
        elif layer_name == "GATv2":
            self.conv = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat)
            conv_out_channels = out_channels * heads if concat else out_channels
        elif layer_name == "GIN":
            nn_gin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            self.conv = GINConv(nn_gin)
            conv_out_channels = out_channels
        elif layer_name == "GINE":
            # GINE uses an MLP for the edge network similar to GIN
            nn_gine = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            self.conv = GINEConv(nn_gine)
            conv_out_channels = out_channels
        elif layer_name == "Transformer":
            # TransformerConv supports multi-head attention and optional concatenation
            self.conv = TransformerConv(in_channels, out_channels, heads=heads, concat=concat)
            conv_out_channels = out_channels * heads if concat else out_channels
        elif layer_name == "SAGE":
            self.conv = SAGEConv(in_channels, out_channels)
            conv_out_channels = out_channels
        else:
            raise ValueError(f"Unknown layer_name: {layer_name}")

        # BatchNorm
        self.bn = nn.BatchNorm1d(conv_out_channels)

        # Residual projection if in/out channels differ
        if in_channels != conv_out_channels:
            self.res_proj = nn.Linear(in_channels, conv_out_channels)
        else:
            self.res_proj = None

    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass for a convolution block. If the underlying conv supports
        an `edge_attr` argument, it will be forwarded; otherwise, a normal
        (x, edge_index) call is used."""
        x_in = x
        # Try passing edge_attr if the conv supports it
        try:
            conv_out = self.conv(x, edge_index, edge_attr)
        except TypeError:
            conv_out = self.conv(x, edge_index)
        x = self.bn(conv_out)
        if self.use_residual and self.res_proj is not None:
            x_in = self.res_proj(x_in)
        x = self.dropout(F.relu(x + x_in))
        return x
    
class GNNClassifier(nn.Module):
    def __init__(self, 
                 in_channels=19, 
                 encoding=32, 
                 hidden_channels=[32, 64, 128], 
                 embedding_size=32, 
                 num_classes=2, 
                 heads=8, 
                 dropout=0.5,
                 scalar_dim=0,
                 use_residual=True, 
                 layer_name="GATv2", 
                 concat=True):
        
        super(GNNClassifier, self).__init__()

        self.dropout = dropout

        # Preprocessing layers
        self.preprocessing_1 = nn.Linear(in_channels, encoding)
        self.preprocessing_2 = nn.Linear(encoding, encoding)

        # Convolution blocks
        self.encoder_convs = torch.nn.ModuleList()
        self.previous_hidden_channel = encoding

        for hidden_channel in hidden_channels:
            self.encoder_convs.append(
                ConvBlock(
                    self.previous_hidden_channel, 
                    hidden_channel, 
                    layer_name=layer_name, 
                    heads=heads, 
                    use_residual=use_residual,
                    dropout=dropout, 
                    concat=concat
                )
            )
            if concat and layer_name in ["GATv2", "GAT"]:
                self.previous_hidden_channel = hidden_channel * heads
            else:
                self.previous_hidden_channel = hidden_channel

        # Final embedding layer
        self.encoder_convs.append(
            ConvBlock(
                self.previous_hidden_channel, 
                embedding_size, 
                layer_name=layer_name, 
                heads=heads, 
                dropout=dropout, 
                concat=False
            )
        )

        # Global pooling
        self.pool = global_mean_pool

        self.scalar_dim = scalar_dim

        # Classification head (MLP)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + scalar_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, data):
        # Preprocessing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.preprocessing_1(x))
        x = F.relu(self.preprocessing_2(x))

        for conv in self.encoder_convs:
            x = conv(x, edge_index, getattr(data, 'edge_attr', None))

        x = self.pool(x, batch)  # [num_graphs, embedding_size]
        if self.scalar_dim > 0:
            scalars = data.scalar_feats
            x = torch.cat([x, scalars], dim=1)

        logits = self.classifier(x)

        return logits
    


class MolecularEGNN_Sparse(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, depth=4, dropout=0.5, scalar_dim=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=in_dim) 
        self.linear_in = nn.Linear(in_dim+1, hidden_dim)
        self.scalar_dim = scalar_dim
        # EGNN backbone
        self.egnn_sparse = EGNN_Sparse(
            feats_dim=hidden_dim,
            pos_dim=3,
            edge_attr_dim=0,  # or >0 if you have edge features
            m_dim=64,         # hidden message dimension
            update_feats=True,
            update_coors=True,
            aggr="add"
        )

        # Graph-level pooling
        self.pool = global_mean_pool

        # Flexible MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )



    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        atom_ids = x[:, 0].long()       # [N]
        ligand_flag = x[:, 1:2]        # [N, 1]

        atom_vec = self.embedding(atom_ids)  # [N, E]

        x = torch.cat([atom_vec, ligand_flag], dim=-1)  # [N, E+1]

        x = F.relu(self.linear_in(x))
        x_input = torch.cat([pos, x], dim=-1)  # [N, 3+feats_dim]
        x_out = self.egnn_sparse(x=x_input, edge_index=edge_index, batch=batch)

        pos = x_out[:, :3]      # updated coordinates
        x = x_out[:, 3:]        #updated features

        # Forward through EGNN

        x = self.pool(x, batch)  # shape: [num_graphs, hidden_dim]
        if self.scalar_dim > 0:
            x = torch.cat([x, data.scalar_feats], dim=1)
        logits = self.head(x)

        return logits



class MolecularEGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, depth=4, dropout=0.5, scalar_dim=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=in_dim) 
        self.linear_in = nn.Linear(in_dim+1, hidden_dim) 
        # EGNN backbone
        self.egnn = EGNN_Network(
            dim=in_dim,
            depth=depth,
            dropout=dropout,
            num_nearest_neighbors=0,     # disables internal KNN
            only_sparse_neighbors = True
    )

        # Graph-level pooling
        self.pool = global_mean_pool

        # Flexible MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )



    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        atom_ids = x[:, 0].long()       # [N]
        ligand_flag = x[:, 1:2]        # [N, 1]

        atom_vec = self.embedding(atom_ids)  # [N, E]

        x = torch.cat([atom_vec, ligand_flag], dim=-1)  # [N, E+1]

        x = F.relu(self.linear_in(x))

        N = x.size(0)
        adj_mat = torch.zeros(N, N, device=x.device, dtype=torch.bool)

        src, dst = edge_index
        adj_mat[src, dst] = True
        x = x.unsqueeze(0)      # [1, N, dim]

        # Forward through EGNN
        x, pos = self.egnn(feats=x, coors=pos, adj_mat=adj_mat)

        x = self.pool(x, batch)  # shape: [num_graphs, hidden_dim]
        if self.scalar_dim > 0:
            x = torch.cat([x, data.scalar_feats], dim=1)
        logits = self.head(x)

        return logits



# ------------------------ New 3-graph model (protein + 2 substrates) ------------------------
class GNN_3G_Classifier(nn.Module):
    """Graph classifier that combines a protein graph and two substrate graphs using cross-attention.
    Input :3D structure of an enzyme and two substrates.
    Output: classification logits (e.g., interaction prediction).
    Expected inputs to forward(): either a tuple (protein_batch, sub1_batch, sub2_batch)
    where each is a PyG Batch object, or a HeteroData-like object where
    'protein', 'substrate1', 'substrate2' are available.

    The indidual graphs are the pput inte thier seperate GNNs, which output embeddings

    Protein encoder mirrors GNNClassifier (without the classification head).
    Each substrate uses a single GNN layer and global pooling. Substrate embeddings
    are concatenated and projected to the protein embedding dim, then cross-attended
    with the protein embedding is applies. The output of cross attention is passed to a small MLP classifier.
    """
    def __init__(self,
                 protein_in_channels:int,
                 ligand_in_channels:int,
                 ligand1_in_channels: int = None,
                 ligand2_in_channels: int = None,
                 encoding=32,
                 hidden_channels=[32,64],
                 embedding_size=32,
                 num_classes=2,
                 heads=4,
                 dropout=0.5,
                 protein_scalar_dim=0,
                 ligand_scalar_dim=0,
                 attn_heads=4,
                 layer_name='GATv2',
                 use_residual=True,
                 concat=True):
        """Add explicit scalar dims for protein and ligand graphs."""
        super().__init__()
        self.dropout = dropout
        self.protein_scalar_dim = protein_scalar_dim
        self.ligand_scalar_dim = ligand_scalar_dim
        self.embedding_size = embedding_size

        # helper: whether scatter_mean is available
        self._has_scatter = scatter_mean is not None

        # Protein preprocessing (two linear layers)
        self.preprocessing_1 = nn.Linear(protein_in_channels, encoding)
        self.preprocessing_2 = nn.Linear(encoding, encoding)

        # Protein Conv blocks
        self.encoder_convs = torch.nn.ModuleList()
        prev = encoding
        for h in hidden_channels:
            self.encoder_convs.append(
                ConvBlock(prev, h, layer_name=layer_name, heads=heads, use_residual=use_residual, dropout=dropout, concat=concat)
            )
            if concat and layer_name in ['GATv2','GAT']:
                prev = h * heads
            else:
                prev = h
        # final projection to embedding_size
        self.proj = nn.Linear(prev, embedding_size)

        # Substrate (ligand) encoders - lightweight (support different input dims per ligand)
        lig1_in = ligand1_in_channels if ligand1_in_channels is not None else ligand_in_channels
        lig2_in = ligand2_in_channels if ligand2_in_channels is not None else ligand_in_channels
        self.ligand1_lin = nn.Linear(lig1_in, embedding_size)
        self.ligand1_conv = SAGEConv(embedding_size, embedding_size)
        self.ligand2_lin = nn.Linear(lig2_in, embedding_size)
        self.ligand2_conv = SAGEConv(embedding_size, embedding_size)

        # Pooling
        self.pool = global_mean_pool

        # Project concatenated substrates into embedding_size
        # substrate embeddings may include ligand scalar features, so input dim accounts for ligand_scalar_dim
        self.substrate_proj = nn.Linear((embedding_size + self.ligand_scalar_dim)*2, embedding_size)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=attn_heads, dropout=dropout, batch_first=True)

        # If protein scalar features exist, project concatenated (prot_emb + scalars) back to embedding_size
        if self.protein_scalar_dim > 0:
            self.prot_scalar_proj = nn.Linear(embedding_size + self.protein_scalar_dim, embedding_size)
        else:
            self.prot_scalar_proj = None

        # Classifier head
        # Input will be [prot_emb_proj (embedding_size) ; attn_out (embedding_size)] -> total = 2*embedding_size
        classifier_in = embedding_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_classes)
        )

    def _aggregate_edge_attrs_to_nodes(self, x, edge_index, edge_attr):
        """Aggregate per-edge attributes into per-node features by averaging incident edge attrs.
        Returns tensor of shape [N, E_attr] (or zeros if aggregation unavailable).
        """
        if edge_attr is None or edge_attr.numel() == 0:
            return None
        if not self._has_scatter:
            # fallback: return zeros of appropriate shape
            return torch.zeros((x.size(0), edge_attr.size(1)), device=x.device, dtype=edge_attr.dtype)
        src, dst = edge_index
        # mean over incoming and outgoing edges
        src_agg = scatter_mean(edge_attr, src, dim=0, dim_size=x.size(0))
        dst_agg = scatter_mean(edge_attr, dst, dim=0, dim_size=x.size(0))
        node_edge_feat = (src_agg + dst_agg) * 0.5
        return node_edge_feat

    def _encode_protein(self, protein_batch):
        x, edge_index, batch = protein_batch.x, protein_batch.edge_index, protein_batch.batch
        # Do NOT aggregate edge attributes into node features. If a conv supports edge attributes
        # it will receive `edge_attr` via its call below (preserving a clean node/edge separation).

        x = F.relu(self.preprocessing_1(x))
        x = F.relu(self.preprocessing_2(x))
        for conv in self.encoder_convs:
            x = conv(x, edge_index, getattr(protein_batch, 'edge_attr', None))
        # global pool to get graph embedding (pure embedding without scalars)
        g = self.pool(x, batch)  # [B, prev]
        g = self.proj(g)        # [B, embedding_size]
        # return embedding (scalars handled in forward)
        return g

    def _encode_ligand1(self, lig_batch):
        x, edge_index, batch = lig_batch.x, lig_batch.edge_index, lig_batch.batch
        x = F.relu(self.ligand1_lin(x))
        try:
            x = self.ligand1_conv(x, edge_index, getattr(lig_batch, 'edge_attr', None))
        except TypeError:
            x = self.ligand1_conv(x, edge_index)
        g = self.pool(x, batch)  # [B, embedding_size]
        if self.ligand_scalar_dim > 0:
            lig_scalars = getattr(lig_batch, 'scalars', None) or getattr(lig_batch, 'scalar_feats', None)
            if lig_scalars is None:
                raise ValueError('ligand_scalar_dim>0 but no scalars found on ligand batch')
            g = torch.cat([g, lig_scalars], dim=1)
        return g

    def _encode_ligand2(self, lig_batch):
        x, edge_index, batch = lig_batch.x, lig_batch.edge_index, lig_batch.batch
        x = F.relu(self.ligand2_lin(x))
        try:
            x = self.ligand2_conv(x, edge_index, getattr(lig_batch, 'edge_attr', None))
        except TypeError:
            x = self.ligand2_conv(x, edge_index)
        g = self.pool(x, batch)  # [B, embedding_size]
        if self.ligand_scalar_dim > 0:
            lig_scalars = getattr(lig_batch, 'scalars', None) or getattr(lig_batch, 'scalar_feats', None)
            if lig_scalars is None:
                raise ValueError('ligand_scalar_dim>0 but no scalars found on ligand batch')
            g = torch.cat([g, lig_scalars], dim=1)
        return g

    def forward(self, inputs):
        # inputs: either tuple (prot, sub1, sub2) or HeteroData-like dict
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            prot_batch, s1_batch, s2_batch = inputs
        else:
            # try to extract from HeteroData-like
            try:
                prot_batch = inputs['protein']
                # support two possible substrate keys
                if 'substrate1' in inputs:
                    s1_batch = inputs['substrate1']
                    s2_batch = inputs['substrate2']
                elif 'ligand1' in inputs:
                    s1_batch = inputs['ligand1']
                    s2_batch = inputs['ligand2']
                elif 'ligand' in inputs:
                    # single ligand -> duplicate
                    s1_batch = inputs['ligand']
                    s2_batch = inputs['ligand']
                else:
                    raise KeyError('No substrate keys found in inputs')
            except Exception as e:
                raise ValueError(f'Unsupported input format for GNN_3G_Classifier: {e}')

        # Encode
        prot_emb = self._encode_protein(prot_batch)  # [B, E]
        s1_emb = self._encode_ligand1(s1_batch)
        s2_emb = self._encode_ligand2(s2_batch)

        # concat substrates and project
        substrate_cat = torch.cat([s1_emb, s2_emb], dim=1)  # [B, 2*(E + ligand_scalar_dim)]
        substrate_proj = F.relu(self.substrate_proj(substrate_cat))  # [B, E]

        # Cross-attention: protein queries substrate
        # reshape for batch_first MHA: MHA expects (B, S, E)
        # we use a single-token sequence for both
        # If protein scalars exist, concatenate them into the prot embedding and project back
        if self.protein_scalar_dim > 0:
            prot_scalars = getattr(prot_batch, 'scalars', None) or getattr(prot_batch, 'scalar_feats', None)
            if prot_scalars is None:
                raise ValueError('protein_scalar_dim>0 but no scalar features found on protein batch')
           # prot_cat = torch.cat([prot_emb, prot_scalars], dim=1)  # [B, E + prot_scalar_dim]
            prot_for_attn = F.relu(self.prot_scalar_proj(prot_emb))  # [B, E]
        else:
            prot_for_attn = prot_emb

        # Cross-attention: protein queries substrate
        prot_q = prot_for_attn.unsqueeze(1)        # [B, 1, E]
        sub_kv = substrate_proj.unsqueeze(1)  # [B, 1, E]

        attn_out, attn_weights = self.cross_attn(prot_q, sub_kv, sub_kv)  # [B,1,E]
        attn_out = attn_out.squeeze(1)  # [B, E]

        # prepare final combined vector (protein embedding already includes scalars if present)
        combined = torch.cat([prot_for_attn, attn_out], dim=1)  # [B, 2E]

        logits = self.classifier(combined)
        return logits


class MolecularEGNN_3G_Sparse(nn.Module):
    """Three-graph model using EGNN_Sparse for the protein encoder and lightweight
    ligand encoders. Mirrors `GNN_3G_Classifier` behaviour but replaces the
    protein GNN stack with an `EGNN_Sparse` backbone.

    Inputs to forward(): tuple (protein_batch, sub1_batch, sub2_batch) or a HeteroData
    with keys 'protein', 'substrate1'/'substrate2' (or 'ligand1'/'ligand2').
    """
    def __init__(self,
                 ligand_in_channels:int,
                 ligand1_in_channels: int = None,
                 ligand2_in_channels: int = None,
                 in_dim:int=32,
                 hidden_dim:int=64,
                 embedding_size:int=32,
                 depth:int=4,
                 num_classes:int=2,
                 dropout:float=0.5,
                 protein_scalar_dim:int=0,
                 ligand_scalar_dim:int=0,
                 attn_heads:int=4,
                 edge_attr_dim:int=11,
                 m_dim:int=64,
                 use_residual:bool=True):
        super().__init__()
        self.dropout = dropout
        self.protein_scalar_dim = protein_scalar_dim
        self.ligand_scalar_dim = ligand_scalar_dim
        self.embedding_size = embedding_size
        self._has_scatter = scatter_mean is not None
        self.edge_attr_dim = edge_attr_dim

        # Protein embedding (atom id -> vector)
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=in_dim)
        # Linear to map node features (atom embedding only) -> hidden_dim
        self.linear_in = nn.Linear(in_dim, hidden_dim)

        # EGNN Sparse backbone (accept per-edge attributes)
        self.egnn_sparse = EGNN_Sparse(
            feats_dim=hidden_dim,
            pos_dim=3,
            edge_attr_dim=self.edge_attr_dim,
            m_dim=m_dim,
            update_feats=True,
            update_coors=True,
            aggr="add"
        )

        # projection to embedding space
        self.proj = nn.Linear(hidden_dim, embedding_size)

        # Ligand encoders (support different input dims per ligand)
        lig1_in = ligand1_in_channels if ligand1_in_channels is not None else ligand_in_channels
        lig2_in = ligand2_in_channels if ligand2_in_channels is not None else ligand_in_channels
        self.ligand1_lin = nn.Linear(lig1_in, embedding_size)
        self.ligand1_conv = SAGEConv(embedding_size, embedding_size)
        self.ligand2_lin = nn.Linear(lig2_in, embedding_size)
        self.ligand2_conv = SAGEConv(embedding_size, embedding_size)

        # Pooling and substrate projection
        self.pool = global_mean_pool
        self.substrate_proj = nn.Linear((embedding_size + self.ligand_scalar_dim) * 2, embedding_size)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=attn_heads, dropout=dropout, batch_first=True)

        # Project protein embedding+scalars back to embedding_size if needed
        if self.protein_scalar_dim > 0:
            self.prot_scalar_proj = nn.Linear(embedding_size + self.protein_scalar_dim, embedding_size)
        else:
            self.prot_scalar_proj = None

        # Classifier expects 2*embedding_size (protein + attn output)
        classifier_in = embedding_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_classes)
        )

    def _aggregate_edge_attrs_to_nodes(self, x, edge_index, edge_attr):
        if edge_attr is None or edge_attr.numel() == 0:
            return None
        if not self._has_scatter:
            return torch.zeros((x.size(0), edge_attr.size(1)), device=x.device, dtype=edge_attr.dtype)
        src, dst = edge_index
        src_agg = scatter_mean(edge_attr, src, dim=0, dim_size=x.size(0))
        dst_agg = scatter_mean(edge_attr, dst, dim=0, dim_size=x.size(0))
        node_edge_feat = (src_agg + dst_agg) * 0.5
        return node_edge_feat

    def _encode_protein(self, prot_batch):
        x, edge_index, pos, batch = prot_batch.x, prot_batch.edge_index, prot_batch.pos, prot_batch.batch
        # atom ids -> embedding
        atom_ids = x[:, 0].long()
        atom_vec = self.embedding(atom_ids)

        # Use atom embedding as node features only (do NOT inject edge attributes into node features)
        node_feats = F.relu(self.linear_in(atom_vec))  # [N, hidden_dim]

        x_input = torch.cat([pos, node_feats], dim=-1)             # [N, 3 + hidden_dim]
        x_out = self.egnn_sparse(x=x_input, edge_index=edge_index, batch=batch,
                                 edge_attr=getattr(prot_batch, 'edge_attr', None))

        pos_updated = x_out[:, :3]
        feats = x_out[:, 3:]
        g = self.pool(feats, batch)   # [B, hidden_dim]
        g = self.proj(g)              # [B, embedding_size]
        return g

    def _encode_ligand1(self, lig_batch):
        x, edge_index, batch = lig_batch.x, lig_batch.edge_index, lig_batch.batch
        x = F.relu(self.ligand1_lin(x))
        x = self.ligand1_conv(x, edge_index)
        
        g = self.pool(x, batch)
        if self.ligand_scalar_dim > 0:
            lig_scalars = getattr(lig_batch, 'scalars', None)
            if lig_scalars is None:
                raise ValueError('ligand_scalar_dim>0 but no scalars found on ligand batch')
            g = torch.cat([g, lig_scalars], dim=1)
        return g

    def _encode_ligand2(self, lig_batch):
        x, edge_index, batch = lig_batch.x, lig_batch.edge_index, lig_batch.batch
        x = F.relu(self.ligand2_lin(x))
        x = self.ligand2_conv(x, edge_index)
        
        g = self.pool(x, batch)
        if self.ligand_scalar_dim > 0:
            lig_scalars = getattr(lig_batch, 'scalars', None)
            if lig_scalars is None:
                raise ValueError('ligand_scalar_dim>0 but no scalars found on ligand batch')
            g = torch.cat([g, lig_scalars], dim=1)
        return g

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            prot_batch, s1_batch, s2_batch = inputs
            print(type(prot_batch))
        else:
            try:
                prot_batch = inputs['protein']
                if 'substrate1' in inputs:
                    s1_batch = inputs['substrate1']
                    s2_batch = inputs['substrate2']
                elif 'ligand1' in inputs:
                    s1_batch = inputs['ligand1']
                    s2_batch = inputs['ligand2']
                elif 'ligand' in inputs:
                    s1_batch = inputs['ligand']
                    s2_batch = inputs['ligand']
                else:
                    raise KeyError('No substrate keys found in inputs')
            except Exception as e:
                raise ValueError(f'Unsupported input format for MolecularEGNN_3G_Sparse: {e}')

        prot_emb = self._encode_protein(prot_batch)
        s1_emb = self._encode_ligand1(s1_batch)
        s2_emb = self._encode_ligand2(s2_batch)

        substrate_cat = torch.cat([s1_emb, s2_emb], dim=1)
        substrate_proj = F.relu(self.substrate_proj(substrate_cat))

        # If protein scalars exist, concatenate them and project back
        if self.protein_scalar_dim > 0:
            raw_scalars = getattr(prot_batch, 'scalars', None)
            
            # If the loader treated it as node-level (Size 80)
            if raw_scalars.shape[0] == prot_batch.num_nodes:
                prot_scalars = global_mean_pool(raw_scalars, prot_batch.batch)
            else:
                # If it's already graph-level (Size 16)
                prot_scalars = raw_scalars

            prot_cat = torch.cat([prot_emb, prot_scalars], dim=1)
            if prot_scalars.dim() == 1:
                prot_scalars = prot_scalars.unsqueeze(1)
            if prot_scalars is None:
                raise ValueError('protein_scalar_dim>0 but no scalar features found on protein batch')
            if prot_scalars.dim() == 1:
                prot_scalars = prot_scalars.unsqueeze(1)
            prot_cat = torch.cat([prot_emb, prot_scalars], dim=1)
            prot_for_attn = F.relu(self.prot_scalar_proj(prot_cat))
        else:
            prot_for_attn = prot_emb

        prot_q = prot_for_attn.unsqueeze(1)
        sub_kv = substrate_proj.unsqueeze(1)
        attn_out, attn_weights = self.cross_attn(prot_q, sub_kv, sub_kv)
        attn_out = attn_out.squeeze(1)

        combined = torch.cat([prot_for_attn, attn_out], dim=1)

        logits = self.classifier(combined)
        return logits
