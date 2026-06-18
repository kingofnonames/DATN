import torch
import torch.nn as nn
import torch.nn.functional as F

# class MultiClassifier(nn.Module):
#     def __init__(self, n_gene, 
#                  n_methyl, 
#                  n_mirna, 
#                  n_classes, 
#                  hidden__dim=128, 
#                  attn_dim=64, 
#                  dropout=0.3):
#         super().__init__()
#         self.n_gene = n_gene
#         self.n_methyl = n_methyl
#         self.n_mirna = n_mirna
#         self.n_classes = n_classes
#         self.hidden_dim = hidden__dim
#         self.attn_dim = attn_dim
#         self.dropout = dropout

#         self.classifier_gene = nn.Sequential(
#             nn.Linear(n_gene, hidden__dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden__dim, n_classes)
#         )
#         self.classifier_methyl = nn.Sequential(
#             nn.Linear(n_methyl, hidden__dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden__dim, n_classes)
#         )
#         self.classifier_mirna = nn.Sequential(
#             nn.Linear(n_mirna, hidden__dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden__dim, n_classes)
#         )
#         self.attn_layer = nn.Sequential(
#             nn.Linear(3, attn_dim),
#             nn.ReLU(),
#             nn.Linear(attn_dim, 1)
#         )
#     def forward(self, x_gene, x_methyl, x_mirna):
#         # x_gene: [batch_size, n_gene]
#         logits_gene = self.classifier_gene(x_gene)
#         # x_methyl: [batch_size, n_methyl]  
#         logits_methyl = self.classifier_methyl(x_methyl)
#         # x_mirna: [batch_size, n_mirna]
#         logits_mirna = self.classifier_mirna(x_mirna)
        
#         # logits: [batch_size, n_classes, 3]
#         logits = torch.stack([logits_gene, logits_methyl, logits_mirna], dim=2)
#         attn_weights = F.softmax(self.attn_layer(logits), dim=2)
#         attn_logits = (logits * attn_weights).sum(dim=2)
#         return attn_logits    


# class MultiClassifier(nn.Module):
#     def __init__(self, hidden_gene=128, hidden_methyl=128, hidden_mirna=128, n_classes=2,
#                  hidden_dim=128, dropout=0.3):
#         super().__init__()

#         # modality encoders (IMPORTANT: shared representation space)
#         self.gene_enc = nn.Sequential(
#             nn.Linear(hidden_gene, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         self.methyl_enc = nn.Sequential(
#             nn.Linear(hidden_methyl, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         self.mirna_enc = nn.Sequential(
#             nn.Linear(hidden_mirna, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         # modality gate (learn importance per sample)
#         self.gate = nn.Sequential(
#             nn.Linear(hidden_dim * 3, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3)
#         )

#         # final classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, n_classes)
#         )

#     def forward(self, x_gene, x_methyl, x_mirna):

#         g = self.gene_enc(x_gene)
#         m = self.methyl_enc(x_methyl)
#         r = self.mirna_enc(x_mirna)

#         # stack features
#         feats = torch.stack([g, m, r], dim=1)  # [B, 3, H]

#         # gating weights
#         gate_input = torch.cat([g, m, r], dim=1)
#         weights = torch.softmax(self.gate(gate_input), dim=1)  # [B, 3]

#         weights = weights.unsqueeze(-1)  # [B, 3, 1]

#         fused = (feats * weights).sum(dim=1)  # [B, H]

#         out = self.classifier(fused)

#         return out



import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Compresses a high-dim omics modality into a small embedding."""
    def __init__(self, n_features, embed_dim=32, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiClassifier(nn.Module):
    def __init__(self, n_gene=128, n_methyl=128, n_mirna=128, n_classes=4,
                 embed_dim=32, attn_dim=16, dropout=0.4):
        super().__init__()
        self.encoder_gene   = ModalityEncoder(n_gene,   embed_dim, dropout)
        self.encoder_methyl = ModalityEncoder(n_methyl, embed_dim, dropout)
        self.encoder_mirna  = ModalityEncoder(n_mirna,  embed_dim, dropout)

        self.head_gene   = nn.Linear(embed_dim, n_classes)
        self.head_methyl = nn.Linear(embed_dim, n_classes)
        self.head_mirna  = nn.Linear(embed_dim, n_classes)

        # gating computed from the *embeddings*, not the logits —
        # gives it real information to decide which modality to trust
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, 3),
        )

    def forward(self, x_gene, x_methyl, x_mirna):
        e_gene   = self.encoder_gene(x_gene)
        e_methyl = self.encoder_methyl(x_methyl)
        e_mirna  = self.encoder_mirna(x_mirna)

        logit_gene   = self.head_gene(e_gene)
        logit_methyl = self.head_methyl(e_methyl)
        logit_mirna  = self.head_mirna(e_mirna)

        # [B, n_classes, 3]
        logits = torch.stack([logit_gene, logit_methyl, logit_mirna], dim=2)

        gate_in = torch.cat([e_gene, e_methyl, e_mirna], dim=1)
        gate_weights = F.softmax(self.gate(gate_in), dim=1)   # [B, 3], correctly over modalities
        gate_weights = gate_weights.unsqueeze(1)              # [B, 1, 3]

        fused_logits = (logits * gate_weights).sum(dim=2)     # [B, n_classes]

        return fused_logits