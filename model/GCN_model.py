import torch
from torch_geometric.nn import ChebConv, GATConv
from torch_geometric.utils import softmax
import torch.nn as nn
import torch.nn.functional as F

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim=256, output_dim=128, dropout=0.3):
        super().__init__()
        self.GCN1 = ChebConv(num_features, hidden_dim, K=2)
        self.GCN2 = ChebConv(hidden_dim, hidden_dim, K=2)
        self.GCN3 = ChebConv(hidden_dim, output_dim, K=2)
        self.dropout = nn.Dropout(dropout)
        self.LP = nn.Linear(num_features, hidden_dim)
        self.ln1 = nn.LayerNorm(num_features, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_weight = softmax(edge_attr, edge_index[0])
        res_x = self.LP(x)

        x = self.ln1(x)
        x = self.dropout(x)
        x = self.GCN1(x, edge_index)
        x = F.silu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = self.GCN2(x, edge_index)
        x = res_x + x
        x = F.silu(x)

        x = self.GCN3(x, edge_index)
        return F.silu(x)


class GATModel(nn.Module):
    def __init__(self, num_features, hidden_dim=256, output_dim=128, dropout=0.3):
        super().__init__()
        
        num_heads = 8
        
        # GAT layers (multi-head attention)
        # hidden_dim must be divisible by num_heads
        self.GAT1 = GATConv(num_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.GAT2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.GAT3 = GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.LP = nn.Linear(num_features, hidden_dim)
        self.ln1 = nn.LayerNorm(num_features, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        res_x = self.LP(x)

        x = self.ln1(x)
        x = self.dropout(x)
        x = self.GAT1(x, edge_index)  # → [N, hidden_dim]
        x = F.silu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = self.GAT2(x, edge_index)  # → [N, hidden_dim]
        x = res_x + x
        x = F.silu(x)

        x = self.GAT3(x, edge_index)  # → [N, output_dim]
        return F.silu(x)

class SimpleGCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim=128, output_dim=128, dropout=0.3):
        super().__init__()
        self.GCN1 = ChebConv(num_features, hidden_dim, K=2)
        self.GCN2 = ChebConv(hidden_dim, output_dim, K=3)
        self.dropout = nn.Dropout(dropout)
        self.LP = nn.Linear(num_features, hidden_dim)
        self.ln1 = nn.LayerNorm(num_features, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        res_x = self.LP(x)

        x = self.ln1(x)
        x = self.dropout(x)
        x = self.GCN1(x, edge_index)
        x = F.silu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = res_x + x
        x = self.GCN2(x, edge_index)
        return F.silu(x)