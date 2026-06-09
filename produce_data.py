import numpy as np
import os
import logging
# from .utils import load_data, build_graph
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    Gene_Expression = sio.loadmat(BASE_DIR / 'data2' / 'PCA_BRCA_v5' / 'BRCA_gene_matrix.mat')['normalize_corr']
    Methy_Expression = sio.loadmat(BASE_DIR / 'data2' / 'PCA_BRCA_v5' / 'BRCA_methy_matrix.mat')['normalize_corr']
    Mirna_Expression = sio.loadmat(BASE_DIR / 'data2' / 'PCA_BRCA_v5' / 'BRCA_mirna_matrix.mat')['normalize_corr']

    edges1 = sp.coo_matrix(Gene_Expression)
    logging.info("Number of edges in Gene adjacency matrix: %s", edges1.data.shape[0])
    file_write_obj = open(BASE_DIR / 'data2' / 'PCA_BRCA_v5' / "edges_gene_brca_v5.csv", 'w+')
    for id in np.arange(edges1.data.shape[0]):
        file_write_obj.writelines(np.str_(edges1.row[id]))
        file_write_obj.write(',')
        file_write_obj.writelines(np.str_(edges1.col[id]))
        file_write_obj.write('\n')
    file_write_obj.close()

    edges2=sp.coo_matrix(Methy_Expression)
    logging.info("Number of edges in Methylation adjacency matrix: %s", edges2.data.shape[0])
    file_write_obj = open(BASE_DIR / 'data2' / 'PCA_BRCA_v5' / "edges_methy_brca_v5.csv", 'w+')
    for id in np.arange(edges2.data.shape[0]):
        file_write_obj.writelines(np.str_(edges2.row[id]))
        file_write_obj.write(',')
        file_write_obj.writelines(np.str_(edges2.col[id]))
        file_write_obj.write('\n')
    file_write_obj.close()

    edges3=sp.coo_matrix(Mirna_Expression)
    logging.info("Number of edges in miRNA adjacency matrix: %s", edges3.data.shape[0])
    file_write_obj = open(BASE_DIR / 'data2' / 'PCA_BRCA_v5' / "edges_mirna_brca_v5.csv", 'w+')
    for id in np.arange(edges3.data.shape[0]):
        file_write_obj.writelines(np.str_(edges3.row[id]))
        file_write_obj.write(',')
        file_write_obj.writelines(np.str_(edges3.col[id]))
        file_write_obj.write('\n')
    file_write_obj.close()

