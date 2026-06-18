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
        x = F.relu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = self.GCN2(x, edge_index)
        x = res_x + x
        x = F.relu(x)

        x = self.GCN3(x, edge_index)
        return F.relu(x)

# class ChebResBlock(nn.Module):
#     def __init__(self, dim, K, dropout=0.2):
#         super().__init__()

#         self.norm1 = nn.LayerNorm(dim)
#         self.conv = ChebConv(dim, dim, K=K)

#         self.norm2 = nn.LayerNorm(dim)

#         self.ffn = nn.Sequential(
#             nn.Linear(dim, 4 * dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(4 * dim, dim)
#         )

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index):

#         h = self.norm1(x)
#         h = self.conv(h, edge_index)
#         h = F.gelu(h)

#         x = x + self.dropout(h)

#         h = self.norm2(x)
#         h = self.ffn(h)

#         x = x + self.dropout(h)

#         return x
    
# class SEBlock(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super().__init__()

#         self.fc = nn.Sequential(
#             nn.Linear(dim, dim // reduction),
#             nn.GELU(),
#             nn.Linear(dim // reduction, dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         scale = self.fc(x.mean(0, keepdim=True))
#         return x * scale

# class GCNModel(nn.Module):

#     def __init__(
#         self,
#         num_features,
#         hidden_dim=256,
#         output_dim=128,
#         dropout=0.2
#     ):
#         super().__init__()

#         self.input_proj = nn.Linear(
#             num_features,
#             hidden_dim
#         )

#         self.block1 = ChebResBlock(
#             hidden_dim,
#             K=2,
#             dropout=dropout
#         )

#         self.block2 = ChebResBlock(
#             hidden_dim,
#             K=4,
#             dropout=dropout
#         )

#         self.block3 = ChebResBlock(
#             hidden_dim,
#             K=6,
#             dropout=dropout
#         )

#         self.se = SEBlock(hidden_dim)

#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(hidden_dim * 3),
#             nn.Linear(hidden_dim * 3, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, data):

#         x, edge_index = data.x, data.edge_index

#         x = self.input_proj(x)

#         h1 = self.block1(x, edge_index)
#         h2 = self.block2(h1, edge_index)
#         h3 = self.block3(h2, edge_index)

#         h3 = self.se(h3)

#         # Jumping Knowledge
#         h = torch.cat(
#             [h1, h2, h3],
#             dim=-1
#         )

#         return self.out_proj(h)
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
        x = F.relu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = self.GAT2(x, edge_index)  # → [N, hidden_dim]
        x = res_x + x
        x = F.relu(x)

        x = self.GAT3(x, edge_index)  # → [N, output_dim]
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