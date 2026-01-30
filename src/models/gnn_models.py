import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, summary, GATv2Conv, GCNConv, GATConv, GINConv, SAGEConv, TransformerConv, GatedGraphConv 
from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.data import Data, Batch
from collections import OrderedDict

from egnn_pytorch import EGNN_Network


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

    def forward(self, x, edge_index):
        x_in = x  
        x = self.bn(self.conv(x, edge_index))
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
            x = conv(x, edge_index)

        x = self.pool(x, batch)  # [num_graphs, embedding_size]
        if self.scalar_dim > 0:
            scalars = data.scalar_feats
            x = torch.cat([x, scalars], dim=1)

        logits = self.classifier(x)

        return logits
    

class MolecularEGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, depth=4, dropout=0.5, scalar_dim=0):
        super().__init__()

        # EGNN backbone
        self.egnn = EGNN_Network(
            dim=in_dim,
            depth=depth,
            hidden_dim=hidden_dim,
            num_neighbors=0,     # disables internal KNN
            use_edges=True
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
        x, pos = self.egnn(x=x, pos=pos, edge_index=edge_index)
        x = self.pool(x, batch)  # shape: [num_graphs, hidden_dim]
        if self.scalar_dim > 0:
            x = torch.cat([x, data.scalar_feats], dim=1)
        logits = self.head(x)

        return logits
