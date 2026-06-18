from abc import ABC
import os
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn.conv import MessagePassing
import torch
from torch_geometric.data import Data
import torch.nn.functional as fun
import torch.nn as nn
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, auc, precision_recall_curve, \
    matthews_corrcoef

import scipy.io as sio
import torch.backends.cudnn as cudnn
from utils import load_edges, set_seed, load_data
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
class ResGCN(torch.nn.Module, ABC):
    def __init__(self, num_feature, num_label, num_samples):
        super(ResGCN, self).__init__()
        self.GCN1 = GCNConv(num_feature, 64, cached=True)
        self.GCN2 = GCNConv(64, num_label, cached=True)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.LP = torch.nn.Linear(num_feature, 64)
        self.ln = torch.nn.LayerNorm([num_samples, 64], elementwise_affine=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        res_x = self.LP(x)
        x = self.GCN1(x, edge_index)
        x=res_x+x
        x=fun.silu(x)
        x=self.ln(x)
        x = self.GCN2(x, edge_index)
        return x
    
SEEDS = [223, 777, 2026]
if __name__ == '__main__':
    for seed in SEEDS:
        set_seed(seed)
        data = sio.loadmat('BRCA.mat')
        features = data['BRCA_Gene_Expression'].T
        labels = data['BRCA_clinicalMatrix']
        indexes = data['BRCA_indexes'].flatten()
        labels = labels.reshape(-1)
        path = "data2/BRCA/"
        cites = path + "edges_gene_brca.csv" 
        index_dict = dict()
        num_samples = features.shape[0]
        edge_index = []
        draw_edge_index = []
        features = preprocessing.scale(features)
        logging.info("Indexes shape: %s", indexes.shape)
        for i in indexes:
            idx = str(i)
            index_dict[idx] = len(index_dict)

        edge_index = load_edges(cites, index_dict)
        draw_edge_index = load_edges(cites, index_dict)
        # with open(cites, "r") as f:
        #     edges = f.readlines()
        #     for edge in edges:
        #         start, end = edge.split(',')
        #         edge_index.append([index_dict[start], index_dict[end]])
        #         draw_edge_index.append([index_dict[start], index_dict[end]])


        print(edge_index)
        labels = torch.LongTensor(labels)
        features = torch.FloatTensor(features)
        edge_index = torch.LongTensor(edge_index)

        mask = torch.randperm(len(index_dict))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        cora = Data(x=features, edge_index=edge_index.contiguous(), y=labels).to(device)
        print(cora)
        print(features.shape[1])


        kfold = KFold(
            n_splits=10,
            shuffle=True,
            random_state=seed
        )

        p_scores = []
        r_scores = []
        f1_scores = []
        acc_scores = []
        ari_scores = []
        mcc_scores = []
        auc_scores = []
        prauc_scores = []
        dbi_scores = []
        ss_scores = []

        os.makedirs('./final_results/ergcn', exist_ok=True)
        result_path = f'./final_results/ergcn/BRCA_results_{seed}.txt'
        num_labels = len(torch.unique(labels))
        for fold, (train_mask, test_mask) in enumerate(kfold.split(mask)):

            logging.info("========== Fold %d ==========", fold)

            model = ResGCN(
                features.shape[1],
                num_labels,
                num_samples
            ).to(device)

            criterion = nn.CrossEntropyLoss().to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                weight_decay=5e-4
            )

            for epoch in range(50):

                model.train()

                optimizer.zero_grad()

                out = model(cora)

                loss = criterion(
                    out[train_mask],
                    cora.y[train_mask]
                )

                loss.backward()

                optimizer.step()

                logging.info(
                    "Fold %d | Epoch %d | Loss %.4f",
                    fold,
                    epoch,
                    loss.item()
                )

                if (epoch + 1) % 10 == 0:

                    model.eval()

                    with torch.no_grad():

                        _, pred = model(cora).max(dim=1)

                        correct = int(
                            pred[train_mask]
                            .eq(cora.y[train_mask])
                            .sum()
                            .item()
                        )

                        acc = correct / len(train_mask)

                        logging.info(
                            "Fold %d | Epoch %d | Train ACC %.4f",
                            fold,
                            epoch,
                            acc
                        )

            model.eval()

            with torch.no_grad():

                out = model(cora)

                out = fun.softmax(out, dim=1)

                X_test = out[test_mask]

                _, pred = X_test.max(dim=1)

                X_test_np = (
                    X_test.cpu()
                    .numpy()
                )

                y_pred = (
                    pred.cpu()
                    .numpy()
                )

                targets_test = (
                    cora.y[test_mask]
                    .cpu()
                    .numpy()
                )

            precision = precision_score(
                targets_test,
                y_pred,
                average='macro'
            )

            recall = recall_score(
                targets_test,
                y_pred,
                average='macro'
            )

            f1 = f1_score(
                targets_test,
                y_pred,
                average='macro'
            )

            acc = accuracy_score(
                targets_test,
                y_pred
            )

            ari = metrics.adjusted_rand_score(
                targets_test,
                y_pred
            )

            mcc = matthews_corrcoef(
                targets_test,
                y_pred
            )

            n_unique_pred = len(np.unique(y_pred))

            if 2 <= n_unique_pred <= len(y_pred) - 1:
                dbi = metrics.davies_bouldin_score(
                    X_test_np,
                    y_pred
                )
                ss = metrics.silhouette_score(
                    X_test_np,
                    y_pred
                )
            else:
                logging.warning(
                    "Fold %d | Skipped DBI/SS: only %d unique predicted label(s)",
                    fold,
                    n_unique_pred
                )
                dbi = np.nan
                ss = np.nan

            y_one_hot = label_binarize(
                targets_test,
                classes=np.arange(num_labels)
            )

            y_score = (
                out[test_mask]
                .cpu()
                .numpy()
            )

            fpr, tpr, _ = metrics.roc_curve(
                y_one_hot.ravel(),
                y_score.ravel()
            )

            auc_score = metrics.auc(
                fpr,
                tpr
            )

            pr, re, _ = precision_recall_curve(
                y_one_hot.ravel(),
                y_score.ravel()
            )

            pr_auc = auc(
                re,
                pr
            )

            p_scores.append(precision)
            r_scores.append(recall)
            f1_scores.append(f1)
            acc_scores.append(acc)
            ari_scores.append(ari)
            mcc_scores.append(mcc)
            auc_scores.append(auc_score)
            prauc_scores.append(pr_auc)
            dbi_scores.append(dbi)
            ss_scores.append(ss)

            logging.info("Precision : %.4f", precision)
            logging.info("Recall    : %.4f", recall)
            logging.info("F1 Score  : %.4f", f1)
            logging.info("ACC       : %.4f", acc)
            logging.info("ARI       : %.4f", ari)
            logging.info("MCC       : %.4f", mcc)
            logging.info("AUC       : %.4f", auc_score)
            logging.info("PR AUC    : %.4f", pr_auc)
            logging.info("DBI       : %.4f", dbi)
            logging.info("SS        : %.4f", ss)

            with open(result_path, "a") as f:
                f.write(
                    "Fold %d | "
                    "Precision: %.4f | "
                    "Recall: %.4f | "
                    "F1 Score: %.4f | "
                    "ACC: %.4f | "
                    "ARI: %.4f | "
                    "MCC: %.4f | "
                    "AUC: %.4f | "
                    "PR AUC: %.4f | "
                    "DBI: %.4f | "
                    "SS: %.4f\n"
                    %
                    (
                        fold,
                        precision,
                        recall,
                        f1,
                        acc,
                        ari,
                        mcc,
                        auc_score,
                        pr_auc,
                        dbi,
                        ss
                    )
                )

        logging.info("========== FINAL ==========")

        logging.info(
            "Precision : %.4f ± %.4f",
            np.mean(p_scores),
            np.std(p_scores)
        )

        logging.info(
            "Recall    : %.4f ± %.4f",
            np.mean(r_scores),
            np.std(r_scores)
        )

        logging.info(
            "F1 Score  : %.4f ± %.4f",
            np.mean(f1_scores),
            np.std(f1_scores)
        )

        logging.info(
            "ACC       : %.4f ± %.4f",
            np.mean(acc_scores),
            np.std(acc_scores)
        )

        logging.info(
            "ARI       : %.4f ± %.4f",
            np.mean(ari_scores),
            np.std(ari_scores)
        )

        logging.info(
            "MCC       : %.4f ± %.4f",
            np.mean(mcc_scores),
            np.std(mcc_scores)
        )

        logging.info(
            "AUC       : %.4f ± %.4f",
            np.mean(auc_scores),
            np.std(auc_scores)
        )

        logging.info(
            "PR AUC    : %.4f ± %.4f",
            np.mean(prauc_scores),
            np.std(prauc_scores)
        )

        logging.info(
            "DBI       : %.4f ± %.4f",
            np.mean(dbi_scores),
            np.std(dbi_scores)
        )

        logging.info(
            "SS        : %.4f ± %.4f",
            np.mean(ss_scores),
            np.std(ss_scores)
        )

        with open(result_path, "a") as f:
            f.write(
                "Final Result | "
                "Precision: %.4f ± %.4f | "
                "Recall: %.4f ± %.4f | "
                "F1 Score: %.4f ± %.4f | "
                "ACC: %.4f ± %.4f | "
                "ARI: %.4f ± %.4f | "
                "MCC: %.4f ± %.4f | "
                "AUC: %.4f ± %.4f | "
                "PR AUC: %.4f ± %.4f | "
                "DBI: %.4f ± %.4f | "
                "SS: %.4f ± %.4f\n"
                %
                (
                    np.mean(p_scores), np.std(p_scores),
                    np.mean(r_scores), np.std(r_scores),
                    np.mean(f1_scores), np.std(f1_scores),
                    np.mean(acc_scores), np.std(acc_scores),
                    np.mean(ari_scores), np.std(ari_scores),
                    np.mean(mcc_scores), np.std(mcc_scores),
                    np.mean(auc_scores), np.std(auc_scores),
                    np.mean(prauc_scores), np.std(prauc_scores),
                    np.mean(dbi_scores), np.std(dbi_scores),
                    np.mean(ss_scores), np.std(ss_scores)
                )
            )
