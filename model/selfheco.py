import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN_model import GCNModel, SimpleGCNModel, GATModel
from ..utils import set_seed

class MultiContrastLoss(nn.Module):
    def __init__(self, hidden_dim, tau=0.5, lam=0.5, weight=1.0, weight_decay=0, eps=1e-8):
        super().__init__()
        self.tau = tau
        self.lam = lam
        self.eps = eps
        self.weight = weight
        self.weight_decay = weight_decay
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / (z1_norm * z2_norm.t() + self.eps) / self.tau)
        return sim_matrix

    def forward(self, z_ge, z_mp, z_sc, z_pos):
        z_proj_ge = self.proj(z_ge)
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)

        # Self Contrastive Loss
        matrix_ge2ge = self.sim(z_ge, z_ge)
        matrix_ge2ge = matrix_ge2ge / (torch.sum(matrix_ge2ge, dim=1).view(-1, 1) + self.eps)
        self_loss_ge = -torch.log(matrix_ge2ge.mul(z_pos).sum(dim=-1)).mean()
        matrix_mp2mp = self.sim(z_mp, z_mp)
        matrix_mp2mp = matrix_mp2mp / (torch.sum(matrix_mp2mp, dim=1).view(-1, 1) + self.eps)
        self_loss_mp = -torch.log(matrix_mp2mp.mul(z_pos).sum(dim=-1)).mean()
        matrix_sc2sc = self.sim(z_sc, z_sc)
        matrix_sc2sc = matrix_sc2sc / (torch.sum(matrix_sc2sc, dim=1).view(-1, 1) + self.eps)
        self_loss_sc = -torch.log(matrix_sc2sc.mul(z_pos).sum(dim=-1)).mean()
        # self_loss = self_loss_mp + self_loss_sc
        # self_loss = self_loss_ge + self_loss_sc
        # self_loss = self_loss_ge + self_loss_mp
        self_loss = self_loss_mp + self_loss_sc + self_loss_ge

        # Contrastive Loss

        matrix_ge2mp = self.sim(z_proj_ge, z_proj_mp)
        matrix_mp2ge = matrix_ge2mp.t()
        matrix_ge2mp = matrix_ge2mp / (torch.sum(matrix_ge2mp, dim=1).view(-1, 1) + self.eps)
        lori_ge_mp = -torch.log(matrix_ge2mp.mul(z_pos).sum(dim=-1)).mean()
        matrix_mp2ge = matrix_mp2ge / (torch.sum(matrix_mp2ge, dim=1).view(-1, 1) + self.eps)
        lori_mp_ge = -torch.log(matrix_mp2ge.mul(z_pos).sum(dim=-1)).mean()
        matrix_ge2sc = self.sim(z_proj_ge, z_proj_sc)
        matrix_sc2ge = matrix_ge2sc.t()
        matrix_ge2sc = matrix_ge2sc / (torch.sum(matrix_ge2sc, dim=1).view(-1, 1) + self.eps)
        lori_ge_sc = -torch.log(matrix_ge2sc.mul(z_pos).sum(dim=-1)).mean()
        matrix_sc2ge = matrix_sc2ge / (torch.sum(matrix_sc2ge, dim=1).view(-1, 1) + self.eps)
        lori_sc_ge = -torch.log(matrix_sc2ge.mul(z_pos).sum(dim=-1)).mean()
        cross_loss = self.lam * lori_ge_mp + (1 - self.lam) * lori_mp_ge + self.lam * lori_ge_sc + (1 - self.lam) * lori_sc_ge
        # total_loss = self.weight * self_loss + (1 - self.weight) * cross_loss
        total_loss = self_loss + self.weight_decay * cross_loss
        # total_loss = self_loss
        # total_loss = self_loss + cross_loss
                                                 
        return total_loss
class MultiHeCo(nn.Module):
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
        # self.ge = GATModel(num_feature1, hidden_dim, output_dim, dropout)
        # self.mp = GATModel(num_feature2, hidden_dim, output_dim, dropout)
        # self.sc = GATModel(num_feature3, hidden_dim, output_dim, dropout)

        self.ge_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mp_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.sc_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, encoder, projector, data):
        z = encoder(data)
        z = projector(z)
        return z
    
    def forward(self, data1, data2, data3):
        z_ge = self.encode(self.ge, self.ge_projector, data1)
        z_mp = self.encode(self.mp, self.mp_projector, data2)
        z_sc = self.encode(self.sc, self.sc_projector, data3)

        return z_ge, z_mp, z_sc
    
    @torch.no_grad()
    def get_embeds(self, data1, data2, data3, alpha1=1.0, alpha2=1.0, alpha3=1.0):
        z_ge = self.encode(self.ge, self.ge_projector, data1)
        z_mp = self.encode(self.mp, self.mp_projector, data2)
        z_sc = self.encode(self.sc, self.sc_projector, data3)

        # z = (alpha2 / (alpha2 + alpha3) * z_mp + alpha3 / (alpha2 + alpha3) * z_sc) 
        # z = (alpha1 / (alpha1 + alpha3) * z_ge + alpha3 / (alpha1 + alpha3) * z_sc)
        # z = (alpha1 / (alpha1 + alpha2) * z_ge + alpha2 / (alpha1 + alpha2) * z_mp)
        z = ((alpha1) * z_ge + alpha2 * z_mp + alpha3 * z_sc) / (alpha1 + alpha2 + alpha3)
        return z.detach().cpu().numpy()


class MultiHeCoWithoutContrastive(nn.Module):
    def __init__(
        self,
        num_feature1, num_feature2, num_feature3,
        hidden_dim=256, output_dim=128, dropout=0.3,
        num_classes=4, hidden_dim1=60, hidden_dim2=30
    ):
        super().__init__()
        self.ge = GCNModel(num_feature1, hidden_dim, output_dim, dropout)
        self.mp = GCNModel(num_feature2, hidden_dim, output_dim, dropout)
        self.sc = GCNModel(num_feature3, hidden_dim, output_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )
        self.ge_projector = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))
        self.mp_projector = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))
        self.sc_projector = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))

    def encode(self, encoder, projector, data):
        z = encoder(data)
        z = projector(z)
        return z

    def forward(self, data1, data2, data3):
        z_ge = self.encode(self.ge, self.ge_projector, data1)
        z_mp = self.encode(self.mp, self.mp_projector, data2)
        z_sc = self.encode(self.sc, self.sc_projector, data3)
        z = (z_ge + z_mp + z_sc) / 3
        return self.classifier(z)

    @torch.no_grad()
    def get_embeds(self, data1, data2, data3, alpha1=1.0, alpha2=1.0, alpha3=1.0):
        z_ge = self.encode(self.ge, self.ge_projector, data1)
        z_mp = self.encode(self.mp, self.mp_projector, data2)
        z_sc = self.encode(self.sc, self.sc_projector, data3)
        z = (alpha1 * z_ge + alpha2 * z_mp + alpha3 * z_sc) / (alpha1 + alpha2 + alpha3)
        return z