import torch
import torch.nn as nn
from pathlib import Path

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
    Simple MLP for binary classification of GT-substrate activity.
    
    Architecture:
    - Input: Concatenated protein + substrate embeddings
    - Hidden layers with batch norm, ReLU, and dropout
    - Output: Binary classification (active/inactive)
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (logits for BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass returns logits (not probabilities)"""
        return self.network(x).squeeze(-1)  # Shape: (batch_size,)


class DeepMLP(nn.Module):
    """
    Deep MLP for binary classification of GT-substrate activity.
    
    Architecture:
    - Input: Concatenated protein + substrate embeddings
    - Multiple hidden layers with batch norm, ReLU, and dropout
    - Residual connections every 2 layers for better gradient flow
    - Output: Binary classification (active/inactive)
    """
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout=0.3):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Build hidden layers with residual connections
        self.hidden_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.hidden_blocks.append(block)
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        """Forward pass with residual connections"""
        x = self.input_projection(x)
        
        for block in self.hidden_blocks:
            x = block(x)
        
        return self.output(x).squeeze(-1)  # Shape: (batch_size,)


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
    def __init__(self, protein_dim, substrate_dim, hidden_dims=[512, 256], dropout=0.3, projection_dim=128):
        super().__init__()
        
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
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
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