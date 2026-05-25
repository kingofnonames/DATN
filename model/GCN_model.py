import torch
from torch_geometric.nn import ChebConv
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
        res_x = self.LP(x)

        x = self.ln1(x)
        x = self.dropout(x)
        x = self.GCN1(x, edge_index)
        x = F.relu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = self.GCN2(x, edge_index)
        x = res_x + x
        x = F.relu(x)

        x = self.GCN3(x, edge_index)
        return F.relu(x)
    

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
        x = F.relu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = res_x + x
        x = self.GCN2(x, edge_index)
        return F.relu(x)