
# import torch
# import torch.nn as nn
# import torch.nn.functional as F



# class MultiClassifier(nn.Module):
#     def __init__(self, input_dim, n_classes=4,
#                  hidden_dim1=60, hidden_dim2=30, dropout=0.3):
#         super().__init__()
#         self.weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  # learnable weights for each modality
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.Tanh(),
#             nn.Linear(hidden_dim1, hidden_dim2),
#             nn.Tanh(),
#             nn.Linear(hidden_dim2, n_classes)
#         )
#     def forward(self, x_gene, x_methyl, x_mirna):
#         weights = self.weights / self.weights.sum()  # normalize weights
#         embeddings = weights[0] * x_gene + weights[1] * x_methyl + weights[2] * x_mirna
    
#         # embeddings = self.weights[0] * x_gene + sweights[1] * x_methyl + weights[2] * x_mirna
#         # embeddings = weights[0] * x_gene + weights[1] * x_methyl + weights[2] * x_mirna
#         # embeddings = ( x_gene +  x_methyl +  x_mirna) / 3
#         logits = self.classifier(embeddings)

#         return logits, weights 

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes=4,
        hidden_dim1=60,
        hidden_dim2=30
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(input_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, n_classes)
        )

    def forward(self, x_gene, x_methyl, x_mirna):

        x_cat = torch.cat(
            [x_gene, x_methyl, x_mirna],
            dim=1
        )

        weights = F.softmax(
            self.gate(x_cat),
            dim=1
        )

        w_gene   = weights[:, 0:1]
        w_methyl = weights[:, 1:2]
        w_mirna  = weights[:, 2:3]

        embeddings = (
            w_gene * x_gene +
            w_methyl * x_methyl +
            w_mirna * x_mirna
        )

        logits = self.classifier(embeddings)

        return logits, weights
    

