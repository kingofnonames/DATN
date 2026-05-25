import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN_model import GCNModel, SimpleGCNModel
from ..utils import set_seed
set_seed(1234)

class MultiContrastLoss(nn.Module):
    def __init__(self, hidden_dim, tau=0.2, lam=0.5, weight=1, eps=1e-8):
        super().__init__()
        self.tau = tau
        self.lam = lam
        self.eps = eps
        self.weight = weight
        self.proj_ge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_sc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.self_contrast_weight = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]))
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / (z1_norm * z2_norm.t() + self.eps) / self.tau)
        return sim_matrix

    def forward(self, z_ge, z_mp, z_sc, z_pos):
        z_proj_ge = self.proj_ge(z_ge)
        z_proj_mp = self.proj_mp(z_mp)
        z_proj_sc = self.proj_sc(z_sc)

        # Self Contrastive Loss
        matrix_ge2ge = self.sim(z_proj_ge, z_proj_ge)
        matrix_ge2ge = matrix_ge2ge / (torch.sum(matrix_ge2ge, dim=1).view(-1, 1) + self.eps)
        self_loss_ge = -torch.log(matrix_ge2ge.mul(z_pos).sum(dim=-1)).mean()
        matrix_mp2mp = self.sim(z_proj_mp, z_proj_mp)
        matrix_mp2mp = matrix_mp2mp / (torch.sum(matrix_mp2mp, dim=1).view(-1, 1) + self.eps)
        self_loss_mp = -torch.log(matrix_mp2mp.mul(z_pos).sum(dim=-1)).mean()
        matrix_sc2sc = self.sim(z_proj_sc, z_proj_sc)
        matrix_sc2sc = matrix_sc2sc / (torch.sum(matrix_sc2sc, dim=1).view(-1, 1) + self.eps)
        self_loss_sc = -torch.log(matrix_sc2sc.mul(z_pos).sum(dim=-1)).mean()
        # self_contrastive_weight = F.softmax(self.self_contrast_weight, dim=0)
        # self_loss = (self_contrastive_weight[0] * self_loss_ge +
        #              self_contrastive_weight[1] * self_loss_mp +
        #              self_contrastive_weight[2] * self_loss_sc)
        self_loss = self_loss_mp + self_loss_sc + self_loss_ge
        # Contrastive Loss

        z_proj_ge_global = self.proj(z_proj_ge)
        z_proj_mp_global = self.proj(z_proj_mp)
        z_proj_sc_global = self.proj(z_proj_sc)
        matrix_ge2mp = self.sim(z_proj_ge_global, z_proj_mp_global)
        matrix_mp2ge = matrix_ge2mp.t()
        matrix_ge2mp = matrix_ge2mp / (torch.sum(matrix_ge2mp, dim=1).view(-1, 1) + self.eps)
        lori_ge_mp = -torch.log(matrix_ge2mp.mul(z_pos).sum(dim=-1)).mean()
        matrix_mp2ge = matrix_mp2ge / (torch.sum(matrix_mp2ge, dim=1).view(-1, 1) + self.eps)
        lori_mp_ge = -torch.log(matrix_mp2ge.mul(z_pos).sum(dim=-1)).mean()
        matrix_ge2sc = self.sim(z_proj_ge_global, z_proj_sc_global)
        matrix_sc2ge = matrix_ge2sc.t()
        matrix_ge2sc = matrix_ge2sc / (torch.sum(matrix_ge2sc, dim=1).view(-1, 1) + self.eps)
        lori_ge_sc = -torch.log(matrix_ge2sc.mul(z_pos).sum(dim=-1)).mean()
        matrix_sc2ge = matrix_sc2ge / (torch.sum(matrix_sc2ge, dim=1).view(-1, 1) + self.eps)
        lori_sc_ge = -torch.log(matrix_sc2ge.mul(z_pos).sum(dim=-1)).mean()
        cross_loss = self.lam * lori_ge_mp + (1 - self.lam) * lori_mp_ge + self.lam * lori_ge_sc + (1 - self.lam) * lori_sc_ge
        total_loss = self.weight * self_loss + (1 - self.weight) * cross_loss
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

        self.ge_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mp_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.sc_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
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
    def get_embeds(self, data1, data2, data3):
        z_ge = self.encode(self.ge, self.ge_projector, data1)
        z_mp = self.encode(self.mp, self.mp_projector, data2)
        z_sc = self.encode(self.sc, self.sc_projector, data3)

        z = (z_ge + z_mp + z_sc) / 3
        return z.detach().cpu().numpy()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .GCN_model import GCNModel


# class SupConLoss(nn.Module):
#     def __init__(self, tau=0.1):
#         super().__init__()
#         self.tau = tau

#     def forward(self, features1, features2, pos_mask):
#         device = features1.device
#         features1 = F.normalize(features1, dim=1)
#         features2 = F.normalize(features2, dim=1)
#         logits = torch.matmul(features1, features2.T) / self.tau
#         logits = logits - logits.max(dim=1, keepdim=True)[0]
#         exp_logits = torch.exp(logits)
#         logits_mask = torch.ones_like(pos_mask).to(device)
#         if features1.shape == features2.shape:
#             logits_mask = logits_mask.fill_diagonal_(0)
#         exp_logits = exp_logits * logits_mask
#         log_prob = logits - torch.log(
#             exp_logits.sum(dim=1, keepdim=True) + 1e-8
#         )

#         pos_mask = pos_mask * logits_mask

#         mean_log_prob_pos = (
#             (pos_mask * log_prob).sum(dim=1)
#             / (pos_mask.sum(dim=1) + 1e-8)
#         )

#         loss = -mean_log_prob_pos.mean()

#         return loss


# class MLPProjector(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()

#         self.projector = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.projector(x)


# class MultiContrast(nn.Module):
#     def __init__(
#         self,
#         hidden_dim,
#         tau=0.1
#     ):
#         super().__init__()

#         self.supcon = SupConLoss(tau=tau)
#         self.loss_weight = nn.Parameter(
#             torch.ones(6)
#         )

#     def forward(
#         self,
#         z_ge,
#         z_mp,
#         z_sc,
#         pos_mask
#     ):
#         """
#         z_ge, z_mp, z_sc : [B, D]
#         pos_mask         : [B, B]
#         """

#         # Self-view contrastive
#         loss_ge_ge = self.supcon(z_ge, z_ge, pos_mask)
#         loss_mp_mp = self.supcon(z_mp, z_mp, pos_mask)
#         loss_sc_sc = self.supcon(z_sc, z_sc, pos_mask)

#         # Cross-view contrastive
#         loss_ge_mp = self.supcon(z_ge, z_mp, pos_mask)
#         loss_ge_sc = self.supcon(z_ge, z_sc, pos_mask)
#         loss_mp_sc = self.supcon(z_mp, z_sc, pos_mask)

#         losses = torch.stack([
#             loss_ge_ge,
#             loss_mp_mp,
#             loss_sc_sc,
#             loss_ge_mp,
#             loss_ge_sc,
#             loss_mp_sc
#         ])

#         weights = F.softmax(self.loss_weight, dim=0)

#         total_loss = (weights * losses).sum()

#         return total_loss


# class MultiHeCo(nn.Module):
#     def __init__(
#         self,
#         num_feature1,
#         num_feature2,
#         num_feature3,
#         hidden_dim=256,
#         output_dim=128,
#         dropout=0.3,
#         tau=0.1
#     ):
#         super().__init__()

#         # Encoders
#         self.ge_encoder = GCNModel(
#             num_feature1,
#             hidden_dim,
#             output_dim,
#             dropout
#         )

#         self.mp_encoder = GCNModel(
#             num_feature2,
#             hidden_dim,
#             output_dim,
#             dropout
#         )

#         self.sc_encoder = GCNModel(
#             num_feature3,
#             hidden_dim,
#             output_dim,
#             dropout
#         )

#         # Projectors
#         self.ge_projector = MLPProjector(
#             output_dim,
#             hidden_dim,
#             output_dim
#         )

#         self.mp_projector = MLPProjector(
#             output_dim,
#             hidden_dim,
#             output_dim
#         )

#         self.sc_projector = MLPProjector(
#             output_dim,
#             hidden_dim,
#             output_dim
#         )

#         # Contrastive module
#         self.contrast = MultiContrast(
#             hidden_dim=output_dim,
#             tau=tau
#         )

#     def encode(
#         self,
#         encoder,
#         projector,
#         data
#     ):
#         z = encoder(data)
#         z = projector(z)
#         z = F.normalize(z, dim=1)
#         return z

#     def forward(
#         self,
#         data1,
#         data2,
#         data3,
#     ):
#         z_ge = self.encode(
#             self.ge_encoder,
#             self.ge_projector,
#             data1
#         )

#         z_mp = self.encode(
#             self.mp_encoder,
#             self.mp_projector,
#             data2
#         )

#         z_sc = self.encode(
#             self.sc_encoder,
#             self.sc_projector,
#             data3
#         )
#         return z_ge, z_mp, z_sc
    
#     @torch.no_grad()
#     def get_embeds(
#         self,
#         data1,
#         data2,
#         data3
#     ):
#         z_ge = self.encode(
#             self.ge_encoder,
#             self.ge_projector,
#             data1
#         )

#         z_mp = self.encode(
#             self.mp_encoder,
#             self.mp_projector,
#             data2
#         )

#         z_sc = self.encode(
#             self.sc_encoder,
#             self.sc_projector,
#             data3
#         )

#         z = z_ge + z_mp + z_sc

#         return z.cpu().numpy()