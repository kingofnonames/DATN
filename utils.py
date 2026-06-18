from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.io as sio
from scipy.io import savemat
from sklearn.decomposition import PCA
import logging
import random
import torch
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
HYPER_GENE = 0.4
HYPER_METHY = 0.5
HYPER_MIRNA = 0.4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def corr_x_y(x: np.ndarray, y: np.ndarray, eps=1e-8):
    assert x.shape[1] == y.shape[1], "Different shape"
    x = x - np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y, axis=1, keepdims=True)
    lxy = np.dot(x, y.T)
    lxx = np.diag(np.dot(x, x.T)).reshape((-1, 1))
    lyy = np.diag(np.dot(y, y.T)).reshape((1, -1))
    corr = lxy / (np.dot(np.sqrt(lxx), np.sqrt(lyy)) + eps)
    return corr

def cos_similarity(x: np.ndarray, y: np.ndarray, eps=1e-8):
    assert x.shape[1] == y.shape[1], "Different shape"
    xy = np.dot(x, y.T)
    norm_x = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    norm_y = np.sqrt(np.sum(y ** 2, axis=1, keepdims=True)).T
    cos_sim = xy / (norm_x * norm_y + eps)
    return cos_sim

def euclidean_similarity(x: np.ndarray, y: np.ndarray, gamma=1.0):
    assert x.shape[1] == y.shape[1], "Different shape"
    xy = np.dot(x, y.T)
    xx = np.diag(np.dot(x, x.T)).reshape((-1, 1))
    yy = np.diag(np.dot(y, y.T)).reshape((1, -1))
    distance = np.sqrt(np.maximum(xx + yy - 2*xy, 0))
    return np.exp(-gamma * distance)

def mahalanobis_similarity(x: np.ndarray, y: np.ndarray, eps=1e-8, gamma=1.0):
    assert x.shape[1] == y.shape[1], "Different shape"
    cov = np.cov(np.concatenate((x, y), axis=0).T) + eps * np.eye(x.shape[1])
    inv_cov = np.linalg.inv(cov)
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    mahalanobis_dist = np.sqrt(np.einsum('...i,ij,...j->...', diff, inv_cov, diff))
    similarity = np.exp(-gamma * mahalanobis_dist)
    return similarity

def load_data(data_path='BRCA.mat', type="BRCA"):
    data = sio.loadmat(data_path)
    # features1 = data['BRCA_Gene_Expression'].T
    # features2 = data['BRCA_Methy_Expression'].T
    # features3 = data['BRCA_Mirna_Expression'].T
    # labels = data['BRCA_clinicalMatrix'].reshape(-1)
    # indexes = data['BRCA_indexes'].flatten()
    features1 = data[f'{type}_Gene_Expression'].T
    features2 = data[f'{type}_Methy_Expression'].T
    features3 = data[f'{type}_Mirna_Expression'].T
    labels = data[f'{type}_clinicalMatrix'].reshape(-1)
    indexes = data[f'{type}_indexes'].flatten()
    logging.info("Shape of Gene Expression: %s", features1.shape)
    logging.info("Shape of Methylation Expression: %s", features2.shape)
    logging.info("Shape of miRNA Expression: %s", features3.shape)
    logging.info("Shape of labels: %s", labels.shape)
    logging.info("Shape of indexes: %s", indexes.shape)
    return features1, features2, features3, labels, indexes

def build_graph(features: np.ndarray, type: str="BRCA_gene", method: str='corr', f_name: str='./data2/'):
    features = StandardScaler().fit_transform(features)
    if method == 'corr':
        z =  abs(corr_x_y(features, features))
        logging.info("Correlation matrix shape: %s", z.shape)
    elif method == 'cos':
        z = cos_similarity(features, features)
        logging.info("Cosine similarity matrix shape: %s", z.shape)
    elif method == 'euclidean':
        z = euclidean_similarity(features, features)
        logging.info("Euclidean similarity matrix shape: %s", z.shape)
    elif method == 'mahalanobis':
        z = mahalanobis_similarity(features, features)
        logging.info("Mahalanobis similarity matrix shape: %s", z.shape)
    else:
        raise ValueError("Unknown method: %s" % method)
    # HYPER = HYPER_GENE if type == "BRCA_gene" else (HYPER_METHY if type == "BRCA_methy" else HYPER_MIRNA)
    HYPER = HYPER_GENE if type == "GBM_gene" else (HYPER_METHY if type == "GBM_methy" else HYPER_MIRNA)
    normalize_z = np.where(z > HYPER, 1, 0)
    sio.savemat(f_name + f'{type}_matrix.mat', {'normalize_corr': normalize_z})
    logging.info("Saved adjacency matrix for %s to %s", type, f_name + f'{type}_matrix.mat')
    return z
    
def build_new_graph(features: np.ndarray, type: str="BRCA_gene", method: str='corr', f_name: str='./data2/', weight=0.3, n_components: float=0.97):
    features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    if method == 'corr':
        z = abs(corr_x_y(features, features))
        z_pca =  abs(corr_x_y(pca_features, pca_features))
        logging.info("Correlation matrix shape: %s", z_pca.shape)
    elif method == 'cos':
        z = cos_similarity(features, features)
        z_pca = cos_similarity(pca_features, pca_features)
        logging.info("Cosine similarity matrix shape: %s", z_pca.shape)
    elif method == 'euclidean':
        z = euclidean_similarity(features, features)
        z_pca = euclidean_similarity(pca_features, pca_features)
        logging.info("Euclidean similarity matrix shape: %s", z_pca.shape)
    elif method == 'mahalanobis':
        z = mahalanobis_similarity(features, features)
        z_pca = mahalanobis_similarity(pca_features, pca_features)
        logging.info("Mahalanobis similarity matrix shape: %s", z_pca.shape)
    else:
        raise ValueError("Unknown method: %s" % method)
    HYPER = HYPER_GENE if type == "BRCA_gene" else (HYPER_METHY if type == "BRCA_methy" else HYPER_MIRNA)
    normalize_z = np.where((weight * z_pca + (1 - weight) * z) > HYPER, 1, 0)
    sio.savemat(f_name + f'{type}_matrix.mat', {'normalize_corr': normalize_z})
    logging.info("Saved adjacency matrix for %s to %s", type, f_name + f'{type}_matrix.mat')
    return z_pca

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_edges(file_path, node_dict):
    edge_index = []

    with open(file_path, "r") as f:
        for line in f:
            start, end = line.strip().split(',')

            start = str(start)
            end = str(end)
        
            if start in node_dict and end in node_dict:
                src = node_dict[start]
                dst = node_dict[end]

                edge_index.append([src, dst])
                edge_index.append([dst, src])

    edge_index = torch.LongTensor(edge_index).t().contiguous()

    return edge_index

def load_weighted_edges(file_path, node_dict):
    edge_index = []
    edge_weight = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            start, end, weight = line.strip().split(',')
            start = str(start)
            end = str(end)
            if start in node_dict and end in node_dict:
                weight = float(weight)
                src = node_dict[start]
                dst = node_dict[end]
                edge_index.append([src, dst])
                edge_index.append([dst, src])
                edge_weight.append(weight)
                edge_weight.append(weight)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_weight = torch.FloatTensor(edge_weight)
    return edge_index, edge_weight

def build_improved_graph(
    features: np.ndarray,
    type: str = "BRCA_gene",
    method: str = 'corr',
    f_name: str = './data2/PCA_BRCA/',
    weight: float = 0.3,
    n_components: float = 0.95,
    threshold_mode: str = 'adaptive', # 'adaptive' | 'percentile' | 'fixed'
    threshold_k: float = 1.1836,
    threshold_pct: float = 97.0,
    k_nearest: int = 100,
    k_min_nearest:int = 3,
) -> np.ndarray:
    scaler = StandardScaler()
    normal_features = scaler.fit_transform(features)

    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(normal_features)

    sim_fn = {
        'corr':        lambda x: np.abs(corr_x_y(x, x)),
        'cos':         lambda x: cos_similarity(x, x),
        'euclidean':   lambda x: euclidean_similarity(x, x),
        'mahalanobis': lambda x: mahalanobis_similarity(x, x),
    }.get(method)
    if sim_fn is None:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: corr, cos, euclidean, mahalanobis")
    
    z_raw = sim_fn(normal_features)
    z_pca = sim_fn(pca_features)
    logging.info("Mean similarity — raw: %.4f | pca: %.4f", z_raw.mean(), z_pca.mean())
    z_raw = np.clip(z_raw, 0.0, 1.0)
    z_pca = np.clip(z_pca, 0.0, 1.0)

    z_fused = weight * z_pca + (1.0 - weight) * z_raw

    off_diag = z_fused[~np.eye(z_fused.shape[0], dtype=bool)]
    # off_diag = z_fused.copy()

    if threshold_mode == 'adaptive':
        threshold = off_diag.mean() + threshold_k * off_diag.std()
        logging.info("Adaptive threshold: %.4f (mean=%.4f, std=%.4f, k=%.1f)",
                     threshold, off_diag.mean(), off_diag.std(), threshold_k)
    elif threshold_mode == 'percentile':
        threshold = np.percentile(off_diag, threshold_pct)
        logging.info("Percentile threshold (%.0f%%): %.4f", threshold_pct, threshold)
    else:
        HYPER = HYPER_GENE if type == "BRCA_gene" else (HYPER_METHY if type == "BRCA_methy" else HYPER_MIRNA)
        threshold = HYPER
        logging.info("Fixed threshold (HYPER): %.4f", threshold)

    normalize_z = np.zeros_like(z_fused, dtype=int)
    zero_degrees = []
    statistic_degrees = []
    for i in range(z_fused.shape[0]):
        row = z_fused[i].copy()
        # row[i] = -1
        top_k = np.argsort(row)[::-1][:k_nearest]
        # top_k = np.argsort(row)[::-1]
        candidates = top_k[row[top_k] > threshold]
        normalize_z[i, candidates] = 1
        if np.sum(normalize_z[i]) == 1:
            zero_degrees.append(i)
        statistic_degrees.append(np.sum(normalize_z[i]))

    # normalize_z = np.maximum(normalize_z, normalize_z.T)
    normalize_z = np.minimum(normalize_z, normalize_z.T)
    # for i in range(len(statistic_degrees)):
    #     degree = statistic_degrees[i]
    #     if degree >= k_min_nearest: continue
    #     three_most_similar = np.argsort(z_fused[i])[::-1][:k_nearest + 1]
    #     three_most_similar = [three for three in three_most_similar if three != i and statistic_degrees[three] >= k_min_nearest]
    #     if len(three_most_similar) == 0: 
    #         # normalize_z[i, i] = 1
    #         continue
    #     number_to_add = min(k_min_nearest + 1 - degree, len(three_most_similar))
    #     # normalize_z[three_most_similar[:number_to_add], i] = 1
    #     normalize_z[i, three_most_similar[:number_to_add]] = 1
    # normalize_z = np.maximum(normalize_z, normalize_z.T)

    logging.info("Average degree: %.2f", np.sum(normalize_z) / normalize_z.shape[0])
    logging.info ("Min degree: %d", np.min(np.sum(normalize_z, axis=1)))
    logging.info ("Max degree: %d", np.max(np.sum(normalize_z, axis=1)))
    logging.info("Nodes with zero degree: %d", len(np.where(np.sum(normalize_z, axis=1) == 0)[0]))
    sio.savemat(f_name + f'{type}_matrix.mat', {'normalize_corr': normalize_z})
    logging.info("Saved adjacency matrix for %s to %s", type, f_name + f'{type}_matrix.mat')

    return z_pca


def convert_csv_to_mat(data_dir, name_subtype="BRCA"):
    labels_path = os.path.join(data_dir, "labels.csv")
    gene_path = os.path.join(data_dir, f"{name_subtype}_mRNA_top.csv")
    miRNA_path = os.path.join(data_dir, f"{name_subtype}_miRNA_top.csv")
    methy_path = os.path.join(data_dir, f"{name_subtype}_Methy_top.csv")
    
    gene_data = pd.read_csv(gene_path, index_col=0).T
    miRNA_data = pd.read_csv(miRNA_path, index_col=0).T
    methy_data = pd.read_csv(methy_path, index_col=0).T
    labels = pd.read_csv(labels_path).values.flatten()
    indexes = gene_data.index.values
    
    assert set(methy_data.index) == set(gene_data.index) == set(miRNA_data.index) == set(indexes)
    indexes_gene_dict = {idx: i for i, idx in enumerate(indexes)}
    gene_data.index = gene_data.index.map(indexes_gene_dict)
    methy_data.index = methy_data.index.map(indexes_gene_dict)
    miRNA_data.index = miRNA_data.index.map(indexes_gene_dict)
    
    save_mat_path = os.path.join(data_dir, f"{name_subtype}_labels.mat")
    sio.savemat(save_mat_path, {
        f'{name_subtype}_Gene_Expression': gene_data.values.T,
        f'{name_subtype}_Methy_Expression': methy_data.values.T,
        f'{name_subtype}_Mirna_Expression': miRNA_data.values.T,
        f'{name_subtype}_clinicalMatrix': labels.reshape(-1, 1),
        f'{name_subtype}_indexes': miRNA_data.index.values.reshape(-1, 1),
    })


def load_data_csv(data_folder ="./data2/BRCA_v2", save_mat_path=None):
    labels_path = os.path.join(data_folder, "labels.csv")
    labels_path = os.path.join(data_folder, "labels.csv")
    BRCA_gene_path = os.path.join(data_folder, "BRCA_mRNA_top.csv")
    BRCA_miRNA_path = os.path.join(data_folder, "BRCA_miRNA_top.csv")
    BRCA_Methy_path = os.path.join(data_folder, "BRCA_Methy_top.csv")
    features1 = pd.read_csv(BRCA_gene_path, index_col=0).T
    features2 = pd.read_csv(BRCA_Methy_path, index_col=0).T
    features3 = pd.read_csv(BRCA_miRNA_path, index_col=0).T
    labels = pd.read_csv(labels_path).values.flatten()
    indexes = features1.index.values
    logging.info("Shape of Gene Expression: %s", features1.shape)
    logging.info("Shape of Methylation Expression: %s", features2.shape)
    logging.info("Shape of miRNA Expression: %s", features3.shape)
    logging.info("Shape of labels: %s", labels.shape)
    logging.info("Shape of indexes: %s", indexes.shape)
    if save_mat_path is not None:
        sio.savemat(save_mat_path, {
            'BRCA_Gene_Expression': features1.values.T,
            'BRCA_Methy_Expression': features2.values.T,
            'BRCA_Mirna_Expression': features3.values.T,
            'BRCA_clinicalMatrix': labels.reshape(-1, 1),
            'BRCA_indexes': indexes.reshape(-1, 1),
        })
        logging.info("Saved .mat file to %s", save_mat_path)
    return features1.values, features2.values, features3.values, labels, indexes

def build_percentile_graph(
    features: np.ndraay,
    type: str = "BRCA",
    method: str = 'corr',
    f_name: str = "./data2/PER_BRCA/",
    weight: float = 0.3,
    n_components: float = 0.95,
    threshold_pct: float = 97.0,
    threshold_std_k: float = 2.0,
    k_nearest: int = 100,
):
    os.makedirs(f_name, exist_ok=True)
    
    scaler = StandardScaler()
    normal_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(normal_features)

    sim_fn = {
        'corr':        lambda x: np.abs(corr_x_y(x, x)),
        'cos':         lambda x: cos_similarity(x, x),
        'euclidean':   lambda x: euclidean_similarity(x, x),
        'mahalanobis': lambda x: mahalanobis_similarity(x, x),
    }.get(method)
    if sim_fn is None:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: corr, cos, euclidean, mahalanobis")
    
    z_raw = sim_fn(normal_features)
    z_pca = sim_fn(pca_features)
    logging.info("Mean similarity — raw: %.4f | pca: %.4f", z_raw.mean(), z_pca.mean())
    z_raw = np.clip(z_raw, 0.0, 1.0)
    z_pca = np.clip(z_pca, 0.0, 1.0)

    z_fused = weight*z_pca + (1.0 - weight)*z_raw
    off_diag = z_fused[~np.eye(z_fused.shape[0], dtype=bool)]

    threshhold = np.percentile(off_diag, threshold_pct)
    logging.info("Percentile threshold (%.1f%%): %.4f", threshold_pct, threshhold)

    normalize_z = np.zeros_like(z_fused, dtype=int)
    for i in range(z_fused.shape[0]):
        row = z_fused[i].copy()
        candidates = row > threshhold
        # dynamic_k = np.mean(row) + threshold_std_k * np.std(row)
        dynamic_k = np.percentile(row, 50)        
        candidates = candidates & (row > dynamic_k)
        normalize_z[i, candidates] = 1
    normalize_z = np.minimum(normalize_z, normalize_z.T)
    logging.info("Average degree: %.2f", np.sum(normalize_z) / normalize_z.shape[0])
    logging.info ("Min degree: %d", np.min(np.sum(normalize_z, axis=1)))
    logging.info ("Max degree: %d", np.max(np.sum(normalize_z, axis=1)))
    logging.info("Nodes with zero degree: %d", len(np.where(np.sum(normalize_z, axis=1) == 0)[0]))
    sio.savemat(f_name + f'{type}_matrix.mat', {'normalize_corr': normalize_z})
    logging.info("Saved adjacency matrix for %s to %s", type, f_name + f'{type}_matrix.mat')

def build_weighted_graph(
    features: np.ndrarray,
    type: str = "BRCA",
    f_name: str = "./data2/WEI_BRCA/",
    method: str = 'corr',
    weight: float = 0.3,
    n_components: float = 0.95,
    threshold_pct: float = 97.0,
    threshold_std_k: float = 2.0,
):
    os.makedirs(f_name, exist_ok=True)
    scaler = StandardScaler()
    normal_features = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(normal_features)
    sim_fn = {
        'corr':        lambda x: corr_x_y(x, x),
        'cos':         lambda x: cos_similarity(x, x),
        'euclidean':   lambda x: euclidean_similarity(x, x),
        'mahalanobis': lambda x: mahalanobis_similarity(x, x),
    }.get(method)
    if sim_fn is None:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: corr, cos, euclidean, mahalanobis")
    z_raw = sim_fn(normal_features)
    z_pca = sim_fn(pca_features)
    z_fused = weight*z_pca + (1.0 - weight)*z_raw
    
    off_diag = z_fused[~np.eye(z_fused.shape[0], dtype=bool)]
    threshold = np.percentile(off_diag, threshold_pct)
    logging.info("Threshold (%.0f%%): %.4f", threshold_pct, threshold)

    normalize_z = np.zeros_like(z_fused, dtype=float)
    for i in range(z_fused.shape[0]):
        row = z_fused[i].copy()
        row[i] = 0
        candidates = np.abs(row) > threshold
        normalize_z[i, candidates] = row[candidates]
    normalize_z = np.minimum(normalize_z, normalize_z.T)
    logging.info("Average degree: %.2f", np.sum(normalize_z > 0) / normalize_z.shape[0])
    logging.info ("Min degree: %d", np.min(np.sum(normalize_z > 0, axis=1)))
    logging.info ("Max degree: %d", np.max(np.sum(normalize_z > 0, axis=1)))
    logging.info("Negative weights: %.4f", np.percentile(z_fused[z_fused < 0], threshold_pct))
    sio.savemat(f_name + f'{type}_matrix.mat', {'normalize_corr': normalize_z})
    logging.info("Nodes with zero degree: %d", len(np.where(np.sum(normalize_z > 0, axis=1) == 0)[0]))
    return normalize_z

if __name__ == '__main__':
    # features1, features2, features3, labels, indexes = load_data()
    # build_improved_graph(features1, type="BRCA_gene", threshold_mode="percentile", method='corr')
    # build_improved_graph(features2, type="BRCA_methy", threshold_mode="percentile", method='corr')
    # build_improved_graph(features3, type="BRCA_mirna", threshold_mode="percentile", method='corr')
    # features1, features2, features3, labels, indexes = load_data_csv(save_mat_path="./BRCA_v5_labels.mat") 
    features1, features2, features3, labels, indexes = load_data("BRCA.mat", type="BRCA")
    # build_graph(features1, f_name = './data2/GBM/', type="GBM_gene", method='corr')
    # build_graph(features2, f_name = './data2/GBM/', type="GBM_methy", method='corr')
    # build_graph(features3, f_name = './data2/GBM/', type="GBM_mirna", method='corr')
    # build_improved_graph(features1, f_name = './data2/PCA_BRCA/', type="BRCA_gene", threshold_mode="percentile", method='corr', threshold_pct=99.5)
    # build_improved_graph(features2, f_name = './data2/PCA_BRCA/', type="BRCA_methy", threshold_mode="percentile", method='corr', threshold_pct=99.5)
    # build_improved_graph(features3, f_name = './data2/PCA_BRCA/', type="BRCA_mirna", threshold_mode="percentile", method='corr', threshold_pct=99.5)
    normalize_features1 = build_weighted_graph(features1, f_name = './data2/WEI_BRCA/', type="BRCA_gene", method='corr', threshold_pct=99)
    normalize_features2 = build_weighted_graph(features2, f_name = './data2/WEI_BRCA/', type="BRCA_methy", method='corr', threshold_pct=99)
    normalize_features3 = build_weighted_graph(features3, f_name = './data2/WEI_BRCA/', type="BRCA_mirna", method='corr', threshold_pct=99)
    print(normalize_features1)