import logging
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    adjusted_rand_score,
    matthews_corrcoef
)

from torch_geometric.data import Data

from .model.selfheco import MultiContrastLoss, MultiHeCo
from .model.classifier_model import MultiClassifier
from .utils import load_edges, set_seed


BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SEEDS = [223, 777, 2026]


if __name__ == "__main__":

    for seed in SEEDS:

        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", device)

        # ---------------- LOAD DATA ---------------- #
        data = sio.loadmat(BASE_DIR / "BRCA.mat")

        features1 = preprocessing.scale(data['BRCA_Gene_Expression'].T)
        features2 = preprocessing.scale(data['BRCA_Methy_Expression'].T)
        features3 = preprocessing.scale(data['BRCA_Mirna_Expression'].T)

        labels = torch.LongTensor(data['BRCA_clinicalMatrix'].reshape(-1))

        indexes = data['BRCA_indexes'].flatten()

        # mapping
        index_gene_dict = {}
        index_methy_dict = {}
        index_mirna_dict = {}

        for idx in indexes:
            idx = str(idx)
            index_gene_dict[idx] = len(index_gene_dict)
            index_methy_dict[idx] = len(index_methy_dict)
            index_mirna_dict[idx] = len(index_mirna_dict)

        # edges
        path = BASE_DIR / "data2" / "BRCA"
        edge_gene_index = load_edges(path / "edges_gene_brca.csv", index_gene_dict)
        edge_methy_index = load_edges(path / "edges_methy_brca.csv", index_methy_dict)
        edge_mirna_index = load_edges(path / "edges_mirna_brca.csv", index_mirna_dict)

        # tensors
        features1 = torch.FloatTensor(features1).to(device)
        features2 = torch.FloatTensor(features2).to(device)
        features3 = torch.FloatTensor(features3).to(device)
        labels = labels.to(device)

        # graph data
        cora1 = Data(x=features1, edge_index=edge_gene_index, y=labels).to(device)
        cora2 = Data(x=features2, edge_index=edge_methy_index, y=labels).to(device)
        cora3 = Data(x=features3, edge_index=edge_mirna_index, y=labels).to(device)

        # ---------------- MASK (UNCHANGED as requested) ---------------- #
        mask = torch.randperm(len(index_gene_dict))

        # ---------------- KFOLD ---------------- #
        kfold = KFold(
            n_splits=10,
            shuffle=True,
            random_state=seed
        )

        # metrics
        p_scores, r_scores, f1_scores = [], [], []
        acc_scores, ari_scores, mcc_scores = [], [], []

        result_path = BASE_DIR / f"BRCA_results_{seed}_classifier.txt"

        # ---------------- FOLD LOOP ---------------- #
        for fold, (train_idx, test_idx) in enumerate(kfold.split(mask)):

            logging.info("========== Fold %d ==========", fold + 1)

            train_idx = torch.tensor(train_idx).to(device)
            test_idx = torch.tensor(test_idx).to(device)

            # ---------------- PHASE 1: CONTRASTIVE ---------------- #
            model = MultiHeCo(
                features1.shape[1],
                features2.shape[1],
                features3.shape[1]
            ).to(device)

            optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            criterion1 = MultiContrastLoss(128, tau=0.5).to(device)

            y_train = cora1.y[train_idx]

            pos = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()

            for epoch in range(80):
                model.train()
                optimizer1.zero_grad()

                z_ge, z_mp, z_sc = model(cora1, cora2, cora3)

                loss = criterion1(
                    z_ge[train_idx],
                    z_mp[train_idx],
                    z_sc[train_idx],
                    pos
                )

                loss.backward()
                optimizer1.step()
                logging.info("Epoch %d | Loss: %.4f", epoch + 1, loss.item())

            # ---------------- EXTRACT EMBEDDINGS ---------------- #
            model.eval()
            with torch.no_grad():
                z_ge, z_mp, z_sc = model(cora1, cora2, cora3)

            z_ge = z_ge.detach()
            z_mp = z_mp.detach()
            z_sc = z_sc.detach()

            # ---------------- PHASE 2: CLASSIFIER ---------------- #
            model_classifier = MultiClassifier(
                n_classes=len(torch.unique(labels))
            ).to(device)

            optimizer2 = torch.optim.Adam(
                model_classifier.parameters(),
                lr=1e-3,
                weight_decay=5e-4
            )

            criterion2 = nn.CrossEntropyLoss()

            y_train = cora1.y[train_idx]

            for epoch in range(50):
                model_classifier.train()
                optimizer2.zero_grad()

                out = model_classifier(
                    z_ge[train_idx],
                    z_mp[train_idx],
                    z_sc[train_idx]
                )

                loss = criterion2(out, y_train)
                loss.backward()
                optimizer2.step()
                logging.info("Epoch %d | Loss: %.4f", epoch + 1, loss.item())

            # ---------------- EVAL ---------------- #
            model_classifier.eval()

            with torch.no_grad():
                logits = model_classifier(
                    z_ge[test_idx],
                    z_mp[test_idx],
                    z_sc[test_idx]
                )

                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_true = cora1.y[test_idx].cpu().numpy()

            # ---------------- METRICS ---------------- #
            p = precision_score(y_true, y_pred, average='macro')
            r = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred)
            ari = adjusted_rand_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            p_scores.append(p)
            r_scores.append(r)
            f1_scores.append(f1)
            acc_scores.append(acc)
            ari_scores.append(ari)
            mcc_scores.append(mcc)

            logging.info("P: %.4f | R: %.4f | F1: %.4f | ACC: %.4f", p, r, f1, acc)

            with open(result_path, "a") as f:
                f.write(
                    f"Fold {fold} | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} "
                    f"ACC:{acc:.4f} ARI:{ari:.4f} MCC:{mcc:.4f}\n"
                )

        # ---------------- FINAL ---------------- #
        logging.info("========== FINAL ==========")
        logging.info("Precision: %.4f", np.mean(p_scores))
        logging.info("Recall   : %.4f", np.mean(r_scores))
        logging.info("F1       : %.4f", np.mean(f1_scores))
        logging.info("Accuracy : %.4f", np.mean(acc_scores))
        logging.info("ARI      : %.4f", np.mean(ari_scores))
        logging.info("MCC      : %.4f", np.mean(mcc_scores))
        with open(result_path, "a") as f:
                f.write(
                    f"Final | P:{np.mean(p_scores):.4f} R:{np.mean(r_scores):.4f} F1:{np.mean(f1_scores):.4f} "
                    f"ACC:{np.mean(acc_scores):.4f} ARI:{np.mean(ari_scores):.4f} MCC:{np.mean(mcc_scores):.4f}\n"
                )