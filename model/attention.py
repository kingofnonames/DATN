import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class OmicsAttention(nn.Module):
    """
    Feature-level attention for omics data.

    Input:
        x: (B, N)

    Output:
        rep:  (B, P)
        attn: (B, N)
    """

    def __init__(
        self,
        n_features,
        n_embedding=128,
        n_proj=64,
        attn_hidden=64,
        dropout=0.2
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
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attn_net = nn.Sequential(
            nn.Linear(n_embedding, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, x):
        """
        x: (B, N)
        """

        B, N = x.shape

        assert N == self.n_features, (
            f"Expected {self.n_features} features, got {N}"
        )

        # emb: (1, N, E)
        emb = self.emb.unsqueeze(0)

        fe = x.unsqueeze(-1) * emb
        fx = self.fx(fe)
        logits = self.attn_net(fe).squeeze(-1)
        attn = torch.softmax(logits, dim=1)
        rep = (attn.unsqueeze(-1) * fx).sum(dim=1)

        return rep, attn