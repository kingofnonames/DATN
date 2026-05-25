import torch
import torch.nn as nn
import torch.nn.functional as F

from .GCN_model import GCNModel, SimpleGCNModel
from .attention import OmicsAttention

class HeCoAttention(nn.Module):
    def __init__(
        self,
        num_feature1,
        num_feature2,
        num_feature3,
        hidden_dim=256,
        output_dim=128,
        dropout=0.3
    ):
        super().__init__()
        self.gen_attention = OmicsAttention(output_dim)
        self.methy_attention = OmicsAttention(output_dim)
        self.mirna_attention = OmicsAttention(output_dim)

        self.ge = GCNModel(num_feature1, hidden_dim, output_dim, dropout)
        self.mp = GCNModel(num_feature2, hidden_dim, output_dim, dropout)
        self.sc = GCNModel(num_feature3, hidden_dim, output_dim, dropout)

        self.projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, encoder, attention, data):
        z = encoder(data)
        z, attn = attention(z)
        # z = self.projector(z)
        return z

    def forward(self, data1, data2, data3):
        z_ge = self.encode(self.ge, self.gen_attention, data1)
        z_mp = self.encode(self.mp, self.methy_attention, data2)
        z_sc = self.encode(self.sc, self.mirna_attention, data3)

        return z_ge, z_mp, z_sc

    @torch.no_grad()
    def get_embeds(self, data1, data2, data3):
        z_ge = self.encode(self.ge, self.gen_attention, data1)
        z_mp = self.encode(self.mp, self.methy_attention, data2)
        z_sc = self.encode(self.sc, self.mirna_attention, data3)

        z = z_ge + z_mp + z_sc

        return z.cpu().numpy()
    

class HeCo(nn.Module):
    def __init__(
        self,
        num_feature1,
        num_feature2,
        num_feature3,
        hidden_dim=256,
        output_dim=128,
        dropout=0.3
    ):
        super().__init__()

        self.ge = GCNModel(num_feature1, hidden_dim, output_dim, dropout)
        self.mp = GCNModel(num_feature2, hidden_dim, output_dim, dropout)
        self.sc = GCNModel(num_feature3, hidden_dim, output_dim, dropout)

        self.projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, encoder, data):
        z = encoder(data)
        z = self.projector(z)
        return z

    def forward(self, data1, data2, data3):
        z_ge = self.encode(self.ge, data1)
        z_mp = self.encode(self.mp, data2)
        z_sc = self.encode(self.sc, data3)

        return z_ge, z_mp, z_sc

    @torch.no_grad()
    def get_embeds(self, data1, data2, data3):
        z_ge = self.encode(self.ge, data1)
        z_mp = self.encode(self.mp, data2)
        z_sc = self.encode(self.sc, data3)

        z = z_ge + z_mp + z_sc

        return z.cpu().numpy()