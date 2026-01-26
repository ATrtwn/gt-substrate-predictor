import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, summary, GATv2Conv, GCNConv, GATConv, GINConv, SAGEConv, TransformerConv, GatedGraphConv 
from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.data import Data, Batch
from collections import OrderedDict

from egnn_pytorch import EGNN_Network


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, concat, layer_name="GATv2"):
        super(ConvBlock, self).__init__()

        if layer_name == "GAT":
            self.conv = GATConv(in_channels, out_channels, heads=heads, concat=concat)
            if concat == True:
                self.bn = nn.BatchNorm1d(out_channels * heads)
            else:
                self.bn = nn.BatchNorm1d(out_channels)

        elif layer_name == "GATv2":
            self.conv = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat)
            if concat == True:
                self.bn = nn.BatchNorm1d(out_channels * heads)
            else:
                self.bn = nn.BatchNorm1d(out_channels)

        elif layer_name == "GIN":
            nn_gin = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
            self.conv = GINConv(nn_gin)
            self.bn = nn.BatchNorm1d(out_channels)

        elif layer_name == "SAGE":
            self.conv = SAGEConv(in_channels, out_channels)
            self.bn = nn.BatchNorm1d(out_channels)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.bn(self.conv(x, edge_index)))
        x = self.dropout(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channels=19, encoding = 32, 
                 hidden_channels=[32, 64, 128], embedding_size = 32, heads=8, 
                 dropout=0.5, layer_name="GATv2", concat=True):
        super(Encoder, self).__init__()

        # Preprocessing layers
        self.preprocessing_1 = nn.Linear(in_channels, encoding)
        self.preprocessing_2 = nn.Linear(encoding, encoding)

        # Convolution blocks
        self.encoder_convs = torch.nn.ModuleList()
        self.previous_hidden_channel = encoding

        for hidden_channel in hidden_channels:
            self.encoder_convs.append(ConvBlock(self.previous_hidden_channel, hidden_channel, layer_name=layer_name, heads=heads, dropout=dropout, concat=concat))
            if concat and layer_name in ["GATv2", "GAT"]:
                self.previous_hidden_channel = hidden_channel * heads
            else:
                self.previous_hidden_channel = hidden_channel

        # Add another layer that have the same size of the last layer 
        self.encoder_convs.append(ConvBlock(self.previous_hidden_channel, embedding_size, layer_name=layer_name, heads=heads, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        # Save the outputs for residual connections
        self.conv_block_outputs = []

        x = F.relu(self.preprocessing_1(x))
        x = F.relu(self.preprocessing_2(x))
        self.conv_block_outputs.append(x)

        for _, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            # Embeddings of encoder layers will not be used!  
            #self.conv_block_outputs.append(x)

        return x, self.conv_block_outputs
    

class MolecularEGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.egnn = EGNN_Network(
            dim=in_dim,
            depth=4,
            hidden_dim=hidden_dim,
            num_neighbors=0,     # IMPORTANT: disables internal KNN
            use_edges=True
        )

        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        x, pos = self.egnn(
            x=x,
            pos=pos,
            edge_index=edge_index
        )

        return self.head(x)