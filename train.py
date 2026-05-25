import logging
import numpy as np
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
from sklearn import metrics, preprocessing
from sklearn.metrics import(
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    auc,
    precision_recall_curve,
    matthews_corrcoef
)

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

from torch_geometric.data import Data

from .model.heco import HeCo, HeCoAttention
from .model.contrast import Contrast
from .model.attention import OmicsAttention
from .utils import set_seed, load_edges
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
logging.info("Base directory: %s", BASE_DIR)

SEED = 1234
if __name__ == '__main__':
    set_seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    data = sio.loadmat(BASE_DIR / "BRCA.mat")

    features1 = data['BRCA_Gene_Expression'].T
    features2 = data['BRCA_Methy_Expression'].T
    features3 = data['BRCA_Mirna_Expression'].T

    labels = data['BRCA_clinicalMatrix'].reshape(-1)
    indexes = data['BRCA_indexes'].flatten()

    features1 = preprocessing.scale(features1)
    features2 = preprocessing.scale(features2)
    features3 = preprocessing.scale(features3)

    index_gene_dict = {}
    index_methy_dict = {}
    index_mirna_dict = {}

    for idx in indexes:
        idx = int(idx)

        index_gene_dict[idx] = len(index_gene_dict)
        index_methy_dict[idx] = len(index_methy_dict)
        index_mirna_dict[idx] = len(index_mirna_dict)

    path = BASE_DIR / "data2"

    cites1 = path / "edges_gene_brca.csv"
    cites2 = path / "edges_methy_brca.csv"
    cites3 = path / "edges_mirna_brca.csv"

    edge_gene_index = load_edges(cites1, index_gene_dict)
    edge_methy_index = load_edges(cites2, index_methy_dict)
    edge_mirna_index = load_edges(cites3, index_mirna_dict)

    logging.info("Gene edges shape: %s", edge_gene_index.shape)
    logging.info("Methy edges shape: %s", edge_methy_index.shape)
    logging.info("Mirna edges shape: %s", edge_mirna_index.shape)
    features1 = torch.FloatTensor(features1)
    features2 = torch.FloatTensor(features2)
    features3 = torch.FloatTensor(features3)
    logging.info("Features1 shape: %s", features1.shape)
    logging.info("Features2 shape: %s", features2.shape)
    logging.info("Features3 shape: %s", features3.shape)
    labels = torch.LongTensor(labels)
    print(labels)

    cora1 = Data(
        x=features1,
        edge_index=edge_gene_index,
        y=labels
    ).to(device)

    cora2 = Data(
        x=features2,
        edge_index=edge_methy_index,
        y=labels
    ).to(device)

    cora3 = Data(
        x=features3,
        edge_index=edge_mirna_index,
        y=labels
    ).to(device)

    mask = torch.randperm(len(index_gene_dict))

    kfold = KFold(
        n_splits=10,
        shuffle=True,
        random_state=SEED
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

    result_path = BASE_DIR / f"results_{SEED}_baseline.txt"
    for fold, (train_mask, test_mask) in enumerate(kfold.split(mask)):

        logging.info("========== Fold %d ==========", fold)
        y_train = cora1.y[train_mask]
        pos = (
            y_train.unsqueeze(0)
            == y_train.unsqueeze(1)
        ).float().to(device)

        model = HeCo(
            features1.shape[1],
            features2.shape[1],
            features3.shape[1]
        ).to(device)
        logging.info("Model initialized with %d parameters", sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=5e-4
        )
        criterion = Contrast(
            128,
            0.5,
            0.5
        ).to(device)
        for epoch in range(120):

            model.train()

            optimizer.zero_grad()

            z_ge, z_mp, z_sc = model(
                cora1,
                cora2,
                cora3
            )

            loss = criterion(
                z_ge[train_mask],
                z_mp[train_mask],
                z_sc[train_mask],
                pos
            )

            loss.backward()

            optimizer.step()

            logging.info(
                "Fold %d | Epoch %d | Loss %.4f",
                fold,
                epoch,
                loss.item()
            )
        model.eval()

        with torch.no_grad():
            embeds = model.get_embeds(
                cora1,
                cora2,
                cora3
            )

        embeds_train = embeds[train_mask]
        embeds_test = embeds[test_mask]

        targets_train = cora1.y[train_mask].cpu().numpy()
        targets_test = cora1.y[test_mask].cpu().numpy()
        classifier = MLPClassifier(
            activation='tanh',
            max_iter=2000,
            solver='adam',
            alpha=0.001,
            hidden_layer_sizes=(60, 30)
        )

        classifier.fit(
            embeds_train,
            targets_train
        )

        y_pred = classifier.predict(embeds_test)
        p_scores.append(
            precision_score(
                targets_test,
                y_pred,
                average='macro'
            )
        )

        r_scores.append(
            recall_score(
                targets_test,
                y_pred,
                average='macro'
            )
        )

        f1_scores.append(
            f1_score(
                targets_test,
                y_pred,
                average='macro'
            )
        )

        acc_scores.append(
            accuracy_score(
                targets_test,
                y_pred
            )
        )

        ari_scores.append(
            metrics.adjusted_rand_score(
                targets_test,
                y_pred
            )
        )

        mcc_scores.append(
            matthews_corrcoef(
                targets_test,
                y_pred
            )
        )

        dbi_scores.append(
            metrics.davies_bouldin_score(
                embeds_test,
                y_pred
            )
        )

        ss_scores.append(
            metrics.silhouette_score(
                embeds_test,
                y_pred
            )
        )
        n_class = len(np.unique(targets_train))

        y_one_hot = label_binarize(
            targets_test,
            classes=np.arange(n_class)
        )

        y_score = classifier.predict_proba(embeds_test)

        fpr, tpr, _ = metrics.roc_curve(
            y_one_hot.ravel(),
            y_score.ravel()
        )

        auc_scores.append(
            metrics.auc(fpr, tpr)
        )

        pr, re, _ = precision_recall_curve(
            y_one_hot.ravel(),
            y_score.ravel()
        )

        prauc_scores.append(
            auc(re, pr)
        )

        logging.info("Precision : %.4f", p_scores[-1])
        logging.info("Recall    : %.4f", r_scores[-1])
        logging.info("F1 Score  : %.4f", f1_scores[-1])
        logging.info("ACC       : %.4f", acc_scores[-1])
        logging.info("ARI       : %.4f", ari_scores[-1])
        logging.info("MCC       : %.4f", mcc_scores[-1])
        logging.info("AUC       : %.4f", auc_scores[-1])
        logging.info("PR AUC    : %.4f", prauc_scores[-1])
        logging.info("DBI       : %.4f", dbi_scores[-1])
        logging.info("SS        : %.4f", ss_scores[-1])
        with open(result_path, "a") as f:
            f.write(
                "Fold %d | Precision: %.4f | Recall: %.4f | F1 Score: %.4f | ACC: %.4f | ARI: %.4f | MCC: %.4f | AUC: %.4f | PR AUC: %.4f | DBI: %.4f | SS: %.4f\n" %
                (
                    fold,
                    p_scores[-1],
                    r_scores[-1],
                    f1_scores[-1],
                    acc_scores[-1],
                    ari_scores[-1],
                    mcc_scores[-1],
                    auc_scores[-1],
                    prauc_scores[-1],
                    dbi_scores[-1],
                    ss_scores[-1]
                )
            )
    logging.info("========== FINAL ==========")

    logging.info("Precision : %.4f", np.mean(p_scores))
    logging.info("Recall    : %.4f", np.mean(r_scores))
    logging.info("F1 Score  : %.4f", np.mean(f1_scores))
    logging.info("ACC       : %.4f", np.mean(acc_scores))
    logging.info("ARI       : %.4f", np.mean(ari_scores))
    logging.info("MCC       : %.4f", np.mean(mcc_scores))
    logging.info("AUC       : %.4f", np.mean(auc_scores))
    logging.info("PR AUC    : %.4f", np.mean(prauc_scores))
    logging.info("DBI       : %.4f", np.mean(dbi_scores))
    logging.info("SS        : %.4f", np.mean(ss_scores))

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
