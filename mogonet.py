import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import logging
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics

cuda = True if torch.cuda.is_available() else False

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)

        return x


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)), (-1, pow(self.num_cls, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)

        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i + 1)].parameters()) + list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_brca_data(data_path):
    data = sio.loadmat(data_path)

    features1 = data['BRCA_Gene_Expression'].T
    features2 = data['BRCA_Methy_Expression'].T
    features3 = data['BRCA_Mirna_Expression'].T

    labels = data['BRCA_clinicalMatrix'].reshape(-1)
    indexes = data['BRCA_indexes'].flatten()

    logging.info("Shape of Gene Expression: %s", features1.shape)
    logging.info("Shape of Methylation Expression: %s", features2.shape)
    logging.info("Shape of miRNA Expression: %s", features3.shape)
    logging.info("Shape of labels: %s", labels.shape)
    logging.info("Shape of indexes: %s", indexes.shape)

    return features1, features2, features3, labels, indexes


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node * data.shape[0]]
    return parameter.data.cpu().numpy().item()


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError
    adj = adj * g
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])

    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def prepare_trte_data_from_arrays(feature_list, labels, tr_idx, te_idx):
    """
    Thay thế cho prepare_trte_data (đọc từ csv). Ở đây nhận trực tiếp các mảng
    features (đã load từ BRCA.mat) và chỉ số train/test của 1 fold.
    """
    num_view = len(feature_list)
    num_tr = len(tr_idx)
    num_te = len(te_idx)

    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((feature_list[i][tr_idx], feature_list[i][te_idx]), axis=0))

    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, num_tr + num_te))

    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                         data_tensor_list[i][idx_dict["te"]].clone()), 0))

    labels_trte = np.concatenate((labels[tr_idx], labels[te_idx]))

    return data_train_list, data_all_list, idx_dict, labels_trte


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)].zero_grad()
        ci = model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob


def compute_metrics(y_true, y_prob, num_class):
    """
    Tính toàn bộ các chỉ số thống kê yêu cầu: Precision, Recall, F1 (macro),
    F1 (weighted), ACC, ARI, MCC, AUC, PR AUC, DBI, SS.
    y_true: nhãn thật (1D array)
    y_prob: ma trận xác suất dự đoán (N x num_class)
    """
    y_pred = y_prob.argmax(1)

    p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)

    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_true, y_pred)

    n_class = num_class
    y_one_hot = label_binarize(y_true, classes=np.arange(n_class))
    if n_class == 2:
        # label_binarize với 2 lớp trả về vector 1 cột -> mở rộng thành 2 cột để khớp y_prob
        y_one_hot = np.hstack([1 - y_one_hot, y_one_hot])

    try:
        fpr, tpr, _ = metrics.roc_curve(y_one_hot.ravel(), y_prob.ravel())
        auc_val = metrics.auc(fpr, tpr)
    except Exception:
        auc_val = float('nan')

    from sklearn.metrics import precision_recall_curve, auc as auc_fn
    try:
        pr, re, _ = precision_recall_curve(y_one_hot.ravel(), y_prob.ravel())
        prauc_val = auc_fn(re, pr)
    except Exception:
        prauc_val = float('nan')

    try:
        if len(np.unique(y_pred)) >= 2:
            dbi = metrics.davies_bouldin_score(y_prob, y_pred)
            ss = metrics.silhouette_score(y_prob, y_pred)
        else:
            dbi = float('nan')
            ss = float('nan')
    except Exception:
        dbi = float('nan')
        ss = float('nan')

    return {
        "precision": p,
        "recall": r,
        "f1_macro": f1,
        "f1_weighted": f1_w,
        "acc": acc,
        "ari": ari,
        "mcc": mcc,
        "auc": auc_val,
        "prauc": prauc_val,
        "dbi": dbi,
        "ss": ss,
    }


def train_test_one_fold(feature_list, labels, tr_idx, te_idx, num_class,
                         lr_e_pretrain, lr_e, lr_c,
                         num_epoch_pretrain, num_epoch,
                         adj_parameter, dim_he_list, dim_hvcdn,
                         log_file=None, fold_idx=None):
    num_view = len(feature_list)

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data_from_arrays(
        feature_list, labels, tr_idx, te_idx)

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)

    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch + 1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

    te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    y_true_te = labels_trte[trte_idx["te"]]

    result = compute_metrics(y_true_te, te_prob, num_class)

    msg = (
        "Fold %d | Precision: %.4f | Recall: %.4f | F1 Score: %.4f | F1 Score (Weighted): %.4f | "
        "ACC: %.4f | ARI: %.4f | MCC: %.4f | AUC: %.4f | PR AUC: %.4f | DBI: %.4f | SS: %.4f"
        % (
            fold_idx,
            result["precision"], result["recall"], result["f1_macro"], result["f1_weighted"],
            result["acc"], result["ari"], result["mcc"], result["auc"], result["prauc"],
            result["dbi"], result["ss"],
        )
    )
    logging.info(msg)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    return result


def run_kfold_for_seed(seed, feature_list, labels, num_class,
                        lr_e_pretrain, lr_e, lr_c,
                        num_epoch_pretrain, num_epoch,
                        adj_parameter, dim_he_list, dim_hvcdn,
                        k=10, log_dir="logs"):
    set_seed(seed)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"seed_{seed}_kfold{k}.log")

    with open(log_file, "w") as f:
        f.write(f"==== Seed: {seed} | KFold: {k} ====\n")

    skf = KFold(n_splits=k, shuffle=True, random_state=seed)

    metric_keys = ["precision", "recall", "f1_macro", "f1_weighted", "acc",
                   "ari", "mcc", "auc", "prauc", "dbi", "ss"]
    all_results = {key: [] for key in metric_keys}

    fold_idx = 0
    for tr_idx, te_idx in skf.split(feature_list[0], labels):
        fold_idx += 1
        logging.info("Seed %d - Fold %d/%d", seed, fold_idx, k)
        result = train_test_one_fold(
            feature_list, labels, tr_idx, te_idx, num_class,
            lr_e_pretrain, lr_e, lr_c,
            num_epoch_pretrain, num_epoch,
            adj_parameter, dim_he_list, dim_hvcdn,
            log_file=log_file, fold_idx=fold_idx
        )
        for key in metric_keys:
            all_results[key].append(result[key])

    summary_msg = "\n========== FINAL (Seed %d) ==========\n" % seed
    for key in metric_keys:
        summary_msg += "%s: %.4f ± %.4f\n" % (key.upper(), np.mean(all_results[key]), np.std(all_results[key]))

    logging.info(summary_msg)
    with open(log_file, "a") as f:
        f.write(summary_msg)

    return all_results


SEEDS = [223, 777, 2026]

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    data_folder = 'BRCA'
    num_class = 5

    adj_parameter = 10
    dim_he_list = [400, 400, 200]

    num_view = 3
    dim_hvcdn = pow(num_class, num_view)

    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    K = 10
    LOG_DIR = "logs"

    features1, features2, features3, labels, indexes = load_brca_data(BASE_DIR / "BRCA.mat")
    feature_list = [features1, features2, features3]

    for seed in SEEDS:
        run_kfold_for_seed(
            seed, feature_list, labels, num_class,
            lr_e_pretrain, lr_e, lr_c,
            num_epoch_pretrain, num_epoch,
            adj_parameter, dim_he_list, dim_hvcdn,
            k=K, log_dir=LOG_DIR
        )