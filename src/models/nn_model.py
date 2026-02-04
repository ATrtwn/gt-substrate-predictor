import torch
import torch.nn as nn
from pathlib import Path


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (Drop Path) layer.
    Randomly drops the entire residual branch during training.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Create random tensor with same shape as first dimension
        random_tensor = keep_prob + torch.rand((x.size(0), 1), dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        return x * random_tensor / keep_prob


def save_model(model, optimizer, epoch, loss, path="experiments/checkpoint.pth"):
    """Save model checkpoint to experiments folder"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(model, optimizer, path, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

class GT_NN(nn.Module):
    """
    Flexible MLP for binary classification of GT-substrate activity.
    
    Architecture:
    - Input: Concatenated protein + substrate embeddings
    - Variable number of hidden layers with batch norm, ReLU, and dropout
    - Output: Binary classification (active/inactive)
    
    Examples:
        Simple (2 layers): hidden_dims=[512, 256]
        Deep (5 layers): hidden_dims=[1024, 512, 256, 128, 64]
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3, activation='relu', stochastic_depth=0.0):
        super().__init__()
        
        # Map activation string to PyTorch class (not instance)
        activation_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU
        }
        activation_class = activation_map.get(activation, nn.ReLU)
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation_class(),  # Create new instance for each layer
                nn.Dropout(dropout),
                StochasticDepth(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer (logits for BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass returns logits (not probabilities)"""
        return self.network(x).squeeze(-1)  # Shape: (batch_size,)


class BilinearInteractionNet(nn.Module):
    """
    Lightweight Bilinear Interaction Network for protein-substrate activity prediction.
    
    Architecture:
    1. Split concatenated embedding into protein (E) and substrate (S)
    2. Project both to lower dimensions (e.g., 128D)
    3. Compute element-wise product: E_proj * S_proj (Hadamard)
    4. Compute dot product: sum(E_proj * S_proj) → scalar interaction score
    5. Concatenate: [E_proj, S_proj, E_proj*S_proj, dot_score]
    6. Feed through MLP for final prediction
    
    This uses projections instead of full bilinear to drastically reduce parameters.
    """
    def __init__(self, protein_dim, substrate_dim, hidden_dims=[512, 256], dropout=0.3, projection_dim=128, activation='relu', stochastic_depth=0.0):
        super().__init__()
        
        # Map activation string to PyTorch class (not instance)
        activation_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU
        }
        activation_class = activation_map.get(activation, nn.ReLU)
        
        self.protein_dim = protein_dim
        self.substrate_dim = substrate_dim
        self.projection_dim = projection_dim
        
        # Project to lower dimensions (this is the key to reducing parameters)
        self.protein_proj = nn.Linear(protein_dim, projection_dim)
        self.substrate_proj = nn.Linear(substrate_dim, projection_dim)
        
        # Calculate total feature dimension after concatenation
        # [protein_proj, substrate_proj, hadamard, dot_score]
        total_dim = projection_dim + projection_dim + projection_dim + 1
        
        # MLP for final prediction
        layers = []
        prev_dim = total_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation_class(),  # Create new instance for each layer
                nn.Dropout(dropout),
                StochasticDepth(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass with efficient interaction modeling.
        
        Args:
            x: Concatenated [protein_embedding, substrate_embedding]
        
        Returns:
            logits for binary classification
        """
        # Split into protein and substrate embeddings
        protein_emb = x[:, :self.protein_dim]
        substrate_emb = x[:, self.protein_dim:]
        
        # Project to lower dimensions
        protein_proj = self.protein_proj(protein_emb)  # (batch, projection_dim)
        substrate_proj = self.substrate_proj(substrate_emb)  # (batch, projection_dim)
        
        # Element-wise product (Hadamard) - captures feature-wise interactions
        hadamard = protein_proj * substrate_proj  # (batch, projection_dim)
        
        # Dot product - single scalar capturing overall compatibility
        dot_score = (hadamard).sum(dim=1, keepdim=True)  # (batch, 1)
        
        # Concatenate all features
        combined = torch.cat([
            protein_proj,      # Compressed protein features
            substrate_proj,    # Compressed substrate features
            hadamard,          # Element-wise interactions
            dot_score          # Scalar interaction score
        ], dim=1)
        
        # Pass through MLP
        return self.mlp(combined).squeeze(-1)


class CrossAttention(nn.Module):
    """
    Cross-attention layer for protein-substrate interaction.
    Query attends to Key-Value pairs to compute weighted features.
    """
    def __init__(self, query_dim, key_value_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_value_dim, query_dim)
        self.value_proj = nn.Linear(key_value_dim, query_dim)
        
        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value):
        """
        Args:
            query: (batch, query_dim) - e.g., protein features
            key_value: (batch, key_value_dim) - e.g., substrate features
        Returns:
            attended: (batch, query_dim) - attended query features
            attention_weights: (batch, num_heads) - for interpretability
        """
        batch_size = query.size(0)
        
        # Project and reshape for multi-head attention
        # We treat the feature dimension as "sequence length" with seq_len=1
        Q = self.query_proj(query).unsqueeze(1)  # (batch, 1, query_dim)
        K = self.key_proj(key_value).unsqueeze(1)  # (batch, 1, query_dim)
        V = self.value_proj(key_value).unsqueeze(1)  # (batch, 1, query_dim)
        
        # Reshape for multi-head: (batch, num_heads, 1, head_dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (batch, num_heads, 1, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values: (batch, num_heads, 1, head_dim)
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back: (batch, 1, query_dim) -> (batch, query_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, -1).squeeze(1)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights.squeeze(-1).squeeze(-1)  # (batch, num_heads)


class AttentionMLP(nn.Module):
    """
    MLP with cross-attention mechanism for protein-substrate interaction prediction.
    
    Architecture:
    1. Protein attends to substrate features (what substrate info is relevant?)
    2. Substrate attends to protein features (what protein info is relevant?)
    3. Concatenate attended features + optional interaction terms
    4. MLP for classification
    """
    def __init__(self, protein_dim, substrate_dim, num_heads=4, 
                 hidden_dims=[512, 256], dropout=0.4, use_residual=True, activation='relu', stochastic_depth=0.0):
        super().__init__()
        
        # Map activation string to PyTorch class (not instance)
        activation_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh, 
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU
        }
        activation_class = activation_map.get(activation, nn.ReLU)
        
        self.protein_dim = protein_dim
        self.substrate_dim = substrate_dim
        self.use_residual = use_residual
        
        # Project to common dimension for attention
        common_dim = 256  # Fixed dimension for Q, K, V
        self.protein_proj = nn.Linear(protein_dim, common_dim)
        self.substrate_proj = nn.Linear(substrate_dim, common_dim)
        
        # Cross-attention layers
        self.protein_to_substrate_attn = CrossAttention(
            query_dim=common_dim,
            key_value_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.substrate_to_protein_attn = CrossAttention(
            query_dim=common_dim,
            key_value_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization for stability
        self.protein_norm = nn.LayerNorm(common_dim)
        self.substrate_norm = nn.LayerNorm(common_dim)
        
        # MLP for final classification
        # Input: attended protein + attended substrate
        mlp_input_dim = common_dim * 2
        
        layers = []
        prev_dim = mlp_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation_class(),  # Create new instance for each layer
                nn.Dropout(dropout),
                StochasticDepth(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass with cross-attention.
        
        Args:
            x: Concatenated [protein_embedding, substrate_embedding]
        
        Returns:
            logits: (batch,) - predictions for binary classification
        """
        # Split into protein and substrate embeddings
        protein_emb = x[:, :self.protein_dim]
        substrate_emb = x[:, self.protein_dim:]
        
        # Project to common dimension
        protein_proj = self.protein_proj(protein_emb)  # (batch, 256)
        substrate_proj = self.substrate_proj(substrate_emb)  # (batch, 256)
        
        # Cross-attention: protein queries substrate
        protein_attended, protein_attn_weights = self.protein_to_substrate_attn(
            query=protein_proj,
            key_value=substrate_proj
        )
        
        # Cross-attention: substrate queries protein
        substrate_attended, substrate_attn_weights = self.substrate_to_protein_attn(
            query=substrate_proj,
            key_value=protein_proj
        )
        
        # Residual connection (optional)
        if self.use_residual:
            protein_attended = protein_attended + protein_proj
            substrate_attended = substrate_attended + substrate_proj
        
        # Layer normalization
        protein_attended = self.protein_norm(protein_attended)
        substrate_attended = self.substrate_norm(substrate_attended)
        
        # Concatenate attended features
        combined = torch.cat([protein_attended, substrate_attended], dim=1)
        
        # Final prediction
        output = self.mlp(combined).squeeze(-1)
        
        return output