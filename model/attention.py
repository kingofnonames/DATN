import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import set_seed

set_seed(1234)

class OmicsAttention(nn.Module):

    def __init__(
        self,
        n_features,
        n_embedding=128,
        n_proj=128,
        dropout=0.3
    ):
        super().__init__()

        self.n_features = n_features
        self.n_embedding = n_embedding
        self.n_proj = n_proj

        self.emb = nn.Parameter(
            torch.empty(n_features, n_embedding)
        )

        nn.init.xavier_uniform_(self.emb)
        self.fx = nn.Sequential(
            nn.Linear(n_embedding, n_proj),
            nn.LayerNorm(n_proj),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        hidden_attn = n_proj // 2

        self.attn_net = nn.Sequential(
            nn.Linear(n_proj, hidden_attn),
            nn.LayerNorm(hidden_attn),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_attn, 1)
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: (B, N)

        return:
            rep  : (B, n_proj)
            attn : (B, N)
        """
        B, N = x.shape
        assert N == self.n_features
        emb = self.emb.unsqueeze(0)          # (1, N, E)

        fe = x.unsqueeze(-1) * emb           # (B, N, E)
        fx = self.fx(fe)                     # (B, N, P)
        attn_logits = self.attn_net(fx).squeeze(-1)   # (B, N)
        attn = torch.softmax(attn_logits, dim=1)
        rep = torch.sum(
            attn.unsqueeze(-1) * fx,
            dim=1
        )                                     # (B, P)

        global_feat = fx.mean(dim=1)
        rep = rep + self.gate * global_feat
        return rep, attn