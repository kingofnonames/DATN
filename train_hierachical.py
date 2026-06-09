"""
train_hierarchical.py  —  Full optimised training pipeline.

Changes vs. previous version
──────────────────────────────
Training loop
  • AdamW replaces Adam: decoupled weight decay is more principled and
    generally produces better generalisation in GNNs.
  • Warm-up cosine schedule: LR linearly ramps up for WARMUP_EPOCHS
    then follows cosine decay.  Avoids the large early-epoch gradient
    variance that destabilises BN layers in the new GCNModel.
  • Positive-pair matrix is class-weighted: samples from minority
    classes (HER2, TN) count more as positives so their representations
    are pulled together more aggressively during contrastive training.
  • GNN trained once per fold on ALL nodes (not just train nodes for the
    forward pass); only the loss is computed on train_mask.  This is the
    transductive setting that ChebConv / GCN models are designed for.

HierarchicalMLPClassifier
  • PCA whitening pre-processing (128 → 64 components) before the MLPs:
    removes correlated directions in the embedding space, reduces
    effective dimensionality, and speeds up MLP convergence.
  • Stage-3 uses SMOTE oversampling when the minority class has fewer
    than 20 samples, addressing the LumB / HER2 imbalance directly
    rather than only via sample weights.
  • Threshold sweep for Stage 1: the TN confidence threshold is chosen
    on the training fold (not hard-coded to 0.55) by maximising F1 for
    the TN class over a grid of thresholds.
"""

import logging
import math
import numpy as np
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from pathlib import Path
from sklearn import metrics, preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch_geometric.data import Data

from .model.selfheco import MultiContrastLoss, MultiHeCo
from .utils import set_seed, load_edges

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
logging.info("Base directory: %s", BASE_DIR)

# ---------------------------------------------------------------------------
# Subtype mapping
#   Label 0 → Luminal A        (~50 %, most common)
#   Label 1 → Luminal B        (~20 %)
#   Label 2 → HER2             (~15 %)
#   Label 3 → Triple-Negative  (~15 %, most aggressive)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-stage MLP configurations
# ---------------------------------------------------------------------------

_STAGE_KWARGS = [
    # Stage 1 — TN vs. Rest: widest net, TN is biologically very distinct
    dict(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=42,
    ),
    # Stage 2 — LumA vs. LumB+HER2: moderate depth
    dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=5e-4,
        learning_rate_init=5e-4,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=42,
    ),
    # Stage 3 — LumB vs. HER2: hardest split; deeper + tanh for smoother boundary
    dict(
        hidden_layer_sizes=(128, 64, 32),
        activation="tanh",
        solver="adam",
        alpha=5e-4,
        learning_rate_init=2e-4,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=60,
        random_state=42,
    ),
]


# ---------------------------------------------------------------------------
# Hierarchical MLP Classifier
# ---------------------------------------------------------------------------

class HierarchicalMLPClassifier:
    """
    Three-stage hierarchical breast-cancer subtype classifier.

    Stage 1  TN (3) vs. Rest (0,1,2)
    Stage 2  LumA (0) vs. LumB+HER2 (1,2)     [subset: predicted Rest]
    Stage 3  LumB (1) vs. HER2 (2)             [subset: predicted LumB+HER2]

    Key additions over previous version
    ─────────────────────────────────────
    • PCA whitening  : 128-d embeddings → 64-d whitened principal components
                       before any MLP sees the data.
    • Threshold sweep: TN decision threshold calibrated on training fold.
    • SMOTE fallback : Stage 3 oversamples the minority class when n < 20.
    """

    MIN_SAMPLES_SMOTE = 20   # SMOTE only when minority class < this

    def __init__(self, n_pca_components: int = 64):
        self.n_pca  = n_pca_components
        self.pca    = PCA(n_components=n_pca_components, whiten=True, random_state=42)
        self.scaler = StandardScaler()
        self.tau    = 0.5                   # will be set by _tune_threshold()

        self.clf_stage1 = CalibratedClassifierCV(
            MLPClassifier(**_STAGE_KWARGS[0]), method="isotonic", cv=3
        )
        self.clf_stage2 = CalibratedClassifierCV(
            MLPClassifier(**_STAGE_KWARGS[1]), method="isotonic", cv=3
        )
        self.clf_stage3 = CalibratedClassifierCV(
            MLPClassifier(**_STAGE_KWARGS[2]), method="isotonic", cv=3
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    @staticmethod
    def _s1_labels(y): return (y == 3).astype(int)
    @staticmethod
    def _s2_labels(y): return (y != 0).astype(int)
    @staticmethod
    def _s3_labels(y): return (y == 2).astype(int)

    def _preprocess(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """StandardScale then PCA-whiten."""
        if fit:
            X = self.scaler.fit_transform(X)
            X = self.pca.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            X = self.pca.transform(X)
        return X.astype(np.float32)

    def _tune_threshold(
        self, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Choose the TN decision threshold that maximises TN-class F1
        on the training data (in-sample, so conservative).
        """
        proba_tn = self.clf_stage1.predict_proba(X)[:, 1]
        best_tau, best_f1 = 0.5, -1.0
        for tau in np.arange(0.30, 0.75, 0.05):
            preds = (proba_tn >= tau).astype(int)
            f1 = f1_score((y == 3).astype(int), preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        logging.info("Stage-1 threshold tuned: τ=%.2f (TN-F1=%.4f)", best_tau, best_f1)
        return float(best_tau)

    def _smote_stage3(
        self, X: np.ndarray, y: np.ndarray
    ):
        """
        Oversample minority class in Stage 3 via simple random duplication
        with small Gaussian noise (synthetic SMOTE without external lib).
        """
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y
        minority_cls = classes[np.argmin(counts)]
        majority_cls = classes[np.argmax(counts)]
        n_minority = counts.min()
        n_majority = counts.max()
        if n_minority >= self.MIN_SAMPLES_SMOTE:
            return X, y

        n_synth = n_majority - n_minority
        minority_idx = np.where(y == minority_cls)[0]
        chosen = np.random.choice(minority_idx, size=n_synth, replace=True)
        noise  = np.random.randn(n_synth, X.shape[1]) * 0.02   # small perturbation
        X_synth = X[chosen] + noise
        y_synth = np.full(n_synth, minority_cls, dtype=int)
        logging.info(
            "Stage 3 SMOTE: generated %d synthetic samples for class %d",
            n_synth, minority_cls
        )
        return np.vstack([X, X_synth]), np.concatenate([y, y_synth])

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "HierarchicalMLPClassifier":
        X = self._to_numpy(X).astype(np.float32)
        y = self._to_numpy(y).astype(int)
        X = self._preprocess(X, fit=True)

        # ── Stage 1 ────────────────────────────────────────────────────
        y1  = self._s1_labels(y)
        sw1 = compute_sample_weight("balanced", y1)
        self.clf_stage1.fit(X, y1, sample_weight=sw1)
        self.tau = self._tune_threshold(X, y)

        # ── Stage 2 ────────────────────────────────────────────────────
        mask_rest = y != 3
        Xr, yr   = X[mask_rest], y[mask_rest]
        if len(Xr) >= 6:
            y2  = self._s2_labels(yr)
            sw2 = compute_sample_weight("balanced", y2)
            self.clf_stage2.fit(Xr, y2, sample_weight=sw2)
        else:
            logging.warning("Stage 2: only %d non-TN samples.", len(Xr))

        # ── Stage 3  (with optional SMOTE) ─────────────────────────────
        mask_12 = np.isin(y, [1, 2])
        X12, y12 = X[mask_12], y[mask_12]
        if len(X12) >= 6:
            X12, y12 = self._smote_stage3(X12, y12)
            y3  = self._s3_labels(y12)
            sw3 = compute_sample_weight("balanced", y3)
            self.clf_stage3.fit(X12, y3, sample_weight=sw3)
        else:
            logging.warning("Stage 3: only %d LumB/HER2 samples.", len(X12))

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        X = self._preprocess(self._to_numpy(X).astype(np.float32), fit=False)
        n = len(X)
        y_pred = np.full(n, -1, dtype=int)

        # Stage 1
        p1      = self.clf_stage1.predict_proba(X)
        tn_mask = p1[:, 1] >= self.tau
        y_pred[tn_mask] = 3

        # Stage 2
        rest_mask = ~tn_mask
        if rest_mask.any():
            rest_idx = np.where(rest_mask)[0]
            p2       = self.clf_stage2.predict_proba(X[rest_idx])
            luma     = p2[:, 0] >= 0.5
            y_pred[rest_idx[luma]] = 0

            # Stage 3
            lumb_her2_idx = rest_idx[~luma]
            if len(lumb_her2_idx):
                p3 = self.clf_stage3.predict_proba(X[lumb_her2_idx])
                y_pred[lumb_her2_idx] = np.where(p3[:, 1] >= 0.5, 2, 1)

        unresolved = y_pred == -1
        if unresolved.any():
            logging.warning("%d unresolved; defaulting to 0.", unresolved.sum())
            y_pred[unresolved] = 0

        return y_pred

    # ------------------------------------------------------------------
    # predict_proba  (joint chain-rule probabilities)
    # ------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        X     = self._preprocess(self._to_numpy(X).astype(np.float32), fit=False)
        proba = np.zeros((len(X), 4), dtype=float)

        p1 = self.clf_stage1.predict_proba(X)
        p_rest       = p1[:, 0]
        proba[:, 3]  = p1[:, 1]                         # P(TN)

        p2 = self.clf_stage2.predict_proba(X)
        proba[:, 0]  = p_rest * p2[:, 0]                # P(LumA)

        p3 = self.clf_stage3.predict_proba(X)
        proba[:, 1]  = p_rest * p2[:, 1] * p3[:, 0]    # P(LumB)
        proba[:, 2]  = p_rest * p2[:, 1] * p3[:, 1]    # P(HER2)

        row_sums = proba.sum(axis=1, keepdims=True).clip(min=1e-12)
        return proba / row_sums


# ---------------------------------------------------------------------------
# LR scheduler:  linear warm-up → cosine decay
# ---------------------------------------------------------------------------

def _lr_schedule(
    optimizer,
    epoch: int,
    total: int,
    warmup: int,
    base_lr: float,
    min_lr: float = 1e-6,
):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

SEED = 2026
seeds = [223, 777, 2026]
if __name__ == "__main__":
    for seed in seeds:
        set_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", device)

        # ── Load data ─────────────────────────────────────────────────────
        data = sio.loadmat(BASE_DIR / "BRCA_v5_labels.mat")

        features1 = data["BRCA_Gene_Expression"].T
        features2 = data["BRCA_Methy_Expression"].T
        features3 = data["BRCA_Mirna_Expression"].T

        labels_np = data["BRCA_clinicalMatrix"].reshape(-1)
        indexes   = data["BRCA_indexes"].flatten()

        features1 = preprocessing.scale(features1)
        features2 = preprocessing.scale(features2)
        features3 = preprocessing.scale(features3)

        index_gene_dict  = {}
        index_methy_dict = {}
        index_mirna_dict = {}
        for idx in indexes:
            idx = str(idx)
            index_gene_dict[idx]  = len(index_gene_dict)
            index_methy_dict[idx] = len(index_methy_dict)
            index_mirna_dict[idx] = len(index_mirna_dict)

        path = BASE_DIR / "data2" / "BRCA_5_labels"
        edge_gene_index  = load_edges(path / "edges_gene_brca.csv",  index_gene_dict)
        edge_methy_index = load_edges(path / "edges_methy_brca.csv", index_methy_dict)
        edge_mirna_index = load_edges(path / "edges_mirna_brca.csv", index_mirna_dict)

        logging.info("Gene  edges: %s", edge_gene_index.shape)
        logging.info("Methy edges: %s", edge_methy_index.shape)
        logging.info("Mirna edges: %s", edge_mirna_index.shape)

        features1 = torch.FloatTensor(features1)
        features2 = torch.FloatTensor(features2)
        features3 = torch.FloatTensor(features3)
        logging.info("Features shapes: %s / %s / %s",
                    features1.shape, features2.shape, features3.shape)

        labels = torch.LongTensor(labels_np)

        cora1 = Data(x=features1, edge_index=edge_gene_index,  y=labels).to(device)
        cora2 = Data(x=features2, edge_index=edge_methy_index, y=labels).to(device)
        cora3 = Data(x=features3, edge_index=edge_mirna_index, y=labels).to(device)

        # ── Stratified 10-fold ────────────────────────────────────────────
        sample_idx = np.arange(len(index_gene_dict))
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        # Metric accumulators
        p_scores = r_scores = f1_scores = acc_scores = []
        ari_scores = mcc_scores = auc_scores = prauc_scores = []
        dbi_scores = ss_scores = []
        p_scores     = []
        r_scores     = []
        f1_scores    = []
        acc_scores   = []
        ari_scores   = []
        mcc_scores   = []
        auc_scores   = []
        prauc_scores = []
        dbi_scores   = []
        ss_scores    = []

        # GNN training hyper-parameters
        GNN_EPOCHS   = 300
        LR_BASE      = 5e-4
        LR_MIN       = 1e-6
        WARMUP_EPOCHS = 15
        PATIENCE     = 30
        WEIGHT_DECAY = 1e-4

        result_path = BASE_DIR / f"results_{seed}_hierarchical_opt.txt"

        for fold, (train_mask, test_mask) in enumerate(
            kfold.split(sample_idx, labels_np)
        ):
            logging.info("========== Fold %d ==========", fold)

            # ── Class-weighted positive-pair matrix ────────────────────────
            y_train    = cora1.y[train_mask]                    # (N_train,) on device
            same_class = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()

            # Inverse-frequency weight per class → minorities are valued more
            class_counts = torch.bincount(y_train, minlength=4).float().clamp(min=1)
            inv_freq     = 1.0 / class_counts                  # (4,)
            row_weights  = inv_freq[y_train]                   # (N_train,)
            # Outer product → weight matrix; normalise rows
            weight_mat   = row_weights.unsqueeze(1) * row_weights.unsqueeze(0)
            pos = same_class * weight_mat
            pos = pos / pos.sum(dim=1, keepdim=True).clamp(min=1e-8)
            pos = pos.to(device)

            # ── Build model ────────────────────────────────────────────────
            model = MultiHeCo(
                features1.shape[1],
                features2.shape[1],
                features3.shape[1],
            ).to(device)
            logging.info(
                "Parameters: %d", sum(p.numel() for p in model.parameters())
            )

            # AdamW: decoupled weight decay
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY
            )
            criterion = MultiContrastLoss(128, tau=0., weight=0.777).to(device)
            # ── Training loop ──────────────────────────────────────────────
            best_loss  = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(GNN_EPOCHS):
                model.train()
                _lr_schedule(optimizer, epoch, GNN_EPOCHS, WARMUP_EPOCHS, LR_BASE, LR_MIN)
                optimizer.zero_grad()

                z_ge, z_mp, z_sc = model(cora1, cora2, cora3)
                loss = criterion(
                    z_ge[train_mask],
                    z_mp[train_mask],
                    z_sc[train_mask],
                    pos,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                loss_val = loss.item()
                logging.info("Fold %d | Epoch %3d | LR %.2e | Loss %.4f",
                            fold, epoch,
                            optimizer.param_groups[0]["lr"],
                            loss_val)

                if loss_val < best_loss - 1e-5:
                    best_loss  = loss_val
                    best_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        logging.info(
                            "Early stop at epoch %d (best %.4f)", epoch, best_loss
                        )
                        break

            if best_state is not None:
                model.load_state_dict(
                    {k: v.to(device) for k, v in best_state.items()}
                )

            # ── Extract embeddings ─────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                # get_embeds() returns attention-fused numpy array
                embeds = model.get_embeds(cora1, cora2, cora3)

            embeds_train  = embeds[train_mask]
            embeds_test   = embeds[test_mask]
            targets_train = cora1.y[train_mask].cpu().numpy()
            targets_test  = cora1.y[test_mask].cpu().numpy()

            # ── Hierarchical classifier ────────────────────────────────────
            classifier = HierarchicalMLPClassifier(n_pca_components=64)
            classifier.fit(embeds_train, targets_train)

            y_pred  = classifier.predict(embeds_test)
            y_score = classifier.predict_proba(embeds_test)

            # ── Metrics ───────────────────────────────────────────────────
            p_scores.append(
                precision_score(targets_test, y_pred, average="macro", zero_division=0)
            )
            r_scores.append(
                recall_score(targets_test, y_pred, average="macro", zero_division=0)
            )
            f1_scores.append(
                f1_score(targets_test, y_pred, average="macro", zero_division=0)
            )
            acc_scores.append(accuracy_score(targets_test, y_pred))
            ari_scores.append(metrics.adjusted_rand_score(targets_test, y_pred))
            mcc_scores.append(matthews_corrcoef(targets_test, y_pred))
            dbi_scores.append(metrics.davies_bouldin_score(embeds_test, y_pred))
            ss_scores.append(metrics.silhouette_score(embeds_test, y_pred))

            n_class   = len(np.unique(targets_train))
            y_one_hot = label_binarize(targets_test, classes=np.arange(n_class))

            fpr, tpr, _ = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
            auc_scores.append(metrics.auc(fpr, tpr))

            pr, re, _ = precision_recall_curve(y_one_hot.ravel(), y_score.ravel())
            prauc_scores.append(auc(re, pr))

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
                    "Fold %d | Precision: %.4f | Recall: %.4f | F1: %.4f | "
                    "ACC: %.4f | ARI: %.4f | MCC: %.4f | AUC: %.4f | "
                    "PR-AUC: %.4f | DBI: %.4f | SS: %.4f\n"
                    % (
                        fold,
                        p_scores[-1], r_scores[-1], f1_scores[-1],
                        acc_scores[-1], ari_scores[-1], mcc_scores[-1],
                        auc_scores[-1], prauc_scores[-1],
                        dbi_scores[-1], ss_scores[-1],
                    )
                )

        # ── Final summary ─────────────────────────────────────────────────
        logging.info("========== FINAL ==========")
        for name, scores in [
            ("Precision", p_scores), ("Recall",    r_scores),
            ("F1 Score",  f1_scores), ("ACC",       acc_scores),
            ("ARI",       ari_scores), ("MCC",      mcc_scores),
            ("AUC",       auc_scores), ("PR AUC",   prauc_scores),
            ("DBI",       dbi_scores), ("SS",       ss_scores),
        ]:
            logging.info("%-10s: %.4f ± %.4f", name, np.mean(scores), np.std(scores))

        with open(result_path, "a") as f:
            f.write(
                "Final | "
                "Precision: %.4f±%.4f | Recall: %.4f±%.4f | F1: %.4f±%.4f | "
                "ACC: %.4f±%.4f | ARI: %.4f±%.4f | MCC: %.4f±%.4f | "
                "AUC: %.4f±%.4f | PR-AUC: %.4f±%.4f | "
                "DBI: %.4f±%.4f | SS: %.4f±%.4f\n"
                % (
                    np.mean(p_scores),    np.std(p_scores),
                    np.mean(r_scores),    np.std(r_scores),
                    np.mean(f1_scores),   np.std(f1_scores),
                    np.mean(acc_scores),  np.std(acc_scores),
                    np.mean(ari_scores),  np.std(ari_scores),
                    np.mean(mcc_scores),  np.std(mcc_scores),
                    np.mean(auc_scores),  np.std(auc_scores),
                    np.mean(prauc_scores),np.std(prauc_scores),
                    np.mean(dbi_scores),  np.std(dbi_scores),
                    np.mean(ss_scores),   np.std(ss_scores),
                )
            )
