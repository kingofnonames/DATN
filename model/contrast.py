import torch
import torch.nn as nn
from ..utils import set_seed
set_seed(1234)
class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau=0.5, lam=0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / (z1_norm * z2_norm.t() + 1e-8) / self.tau)
        return sim_matrix
    
    def forward(self, z_ge, z_mp, z_sc, pos):
        z_proj_ge = self.proj(z_ge)
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)

        matrix_mp2sc1 = self.sim(z_proj_ge, z_proj_mp)
        matrix_sc2mp1= matrix_mp2sc1.t()
        matrix_mp2sc1 = matrix_mp2sc1/(torch.sum(matrix_mp2sc1, dim=1).view(-1, 1) + 1e-8)
        lori_mp1 = -torch.log(matrix_mp2sc1.mul(pos).sum(dim=-1)).mean()
        matrix_sc2mp1 = matrix_sc2mp1 / (torch.sum(matrix_sc2mp1, dim=1).view(-1, 1) + 1e-8)
        lori_sc1 = -torch.log(matrix_sc2mp1.mul(pos).sum(dim=-1)).mean()
        loss1=self.lam * lori_mp1 + (1 - self.lam) * lori_sc1

        matrix_mp2sc2 = self.sim(z_proj_ge, z_proj_sc)
        matrix_sc2mp2 = matrix_mp2sc2.t()
        matrix_mp2sc2 = matrix_mp2sc2 / (torch.sum(matrix_mp2sc2, dim=1).view(-1, 1) + 1e-8)
        lori_mp2 = -torch.log(matrix_mp2sc2.mul(pos).sum(dim=-1)).mean()
        matrix_sc2mp2 = matrix_sc2mp2 / (torch.sum(matrix_sc2mp2, dim=1).view(-1, 1) + 1e-8)
        lori_sc2 = -torch.log(matrix_sc2mp2.mul(pos).sum(dim=-1)).mean()
        loss2 = self.lam * lori_mp2 + (1 - self.lam) * lori_sc2

        loss=loss1+loss2
        return loss