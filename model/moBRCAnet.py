from .attention import OmicsAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class moBRCANetBaseline(nn.Module):
    def __init__(self,
                 n_gene,
                 n_methyl,
                 n_mirna,
                 n_classes,
                 n_embedding=128,
                 n_proj=64,
                 n_sm_h2=200,
                 dropout=0.2):
        super().__init__()
        self.gene_attn = OmicsAttention(n_gene, n_embedding, n_proj, dropout)
        self.methyl_attn = OmicsAttention(n_methyl, n_embedding, n_proj, dropout)
        self.mirna_attn = OmicsAttention(n_mirna, n_embedding, n_proj, dropout)

        in_dim = n_proj * 3
        self.fc2 = nn.Linear(in_dim, n_sm_h2)
        self.bn2 = nn.BatchNorm1d(n_sm_h2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(n_sm_h2, n_classes)

    def forward(self, gene_x, methyl_x, mirna_x):
        rep_gene, attn_gene = self.gene_attn(gene_x)
        rep_methyl, attn_methyl = self.methyl_attn(methyl_x)
        rep_mirna, attn_mirna = self.mirna_attn(mirna_x)

        rep_concat = torch.cat([rep_gene, rep_methyl, rep_mirna], dim=1)
        h = self.fc2(rep_concat)
        h = self.bn2(h)
        h = self.elu(h)
        h = self.dropout(h)
        logits = self.fc_out(h)

        return logits, (attn_gene, attn_methyl, attn_mirna)
    
def train_and_eval(
    train_ds,
    test_ds,
    n_gene,
    n_methyl,
    n_mirna,
    n_classes,
    res_dir,
    batch_size=8,
    epochs=120,
    lr=5e-3,
    dropout=0.2,
    weight_decay=5e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = moBRCANetBaseline(
        n_gene=n_gene,
        n_methyl=n_methyl,
        n_mirna=n_mirna,
        n_classes=n_classes,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for gene_x, methyl_x, mirna_x, labels in train_loader:
            gene_x, methyl_x, mirna_x, labels = gene_x.to(device), methyl_x.to(device), mirna_x.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(gene_x, methyl_x, mirna_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * gene_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")