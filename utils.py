import numpy as np
import logging
import scipy.io as sio
import random
from sklearn.decomposition import PCA
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
HYPER_GENE = 0.7
HYPER_METHY = 0.75
HYPER_MIRNA = 0.25

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

def load_data(data_path='BRCA.mat'):
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
    HYPER = HYPER_GENE if type == "BRCA_gene" else (HYPER_METHY if type == "BRCA_methy" else HYPER_MIRNA)
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

            start = int(start)
            end = int(end)

            if start in node_dict and end in node_dict:
                src = node_dict[start]
                dst = node_dict[end]

                edge_index.append([src, dst])
                edge_index.append([dst, src])

    edge_index = torch.LongTensor(edge_index).t().contiguous()

    return edge_index
if __name__ == '__main__':
    features1, features2, features3, labels, indexes = load_data()
    build_new_graph(features1, type="BRCA_gene", method='corr')
    build_new_graph(features2, type="BRCA_methy", method='corr')
    build_new_graph(features3, type="BRCA_mirna", method='corr')