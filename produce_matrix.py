import numpy as np
import os
import logging
# from .utils import load_data, build_graph
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
from utils import load_data, build_improved_graph, build_percentile_graph, build_weighted_graph, build_percentile_graph_vfinal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    features1, features2, features3, labels, indexes = load_data("GBM.mat", type="GBM")
    # build_graph(features1, f_name = './data2/OV/', type="OV_gene", method='corr')
    # build_graph(features2, f_name = './data2/OV/', type="OV_methy", method='corr')
    # build_graph(features3, f_name = './data2/OV/', type="OV_mirna", method='corr')
    build_percentile_graph_vfinal(features1, f_name = './data2/PER_GBM_/', type="GBM_gene", method='corr', threshold_pct=97.5)
    build_percentile_graph_vfinal(features2, f_name = './data2/PER_GBM_/', type="GBM_methy", method='corr', threshold_pct=98)
    build_percentile_graph_vfinal(features3, f_name = './data2/PER_GBM_/', type="GBM_mirna", method='corr', threshold_pct=98)
    # build_weighted_graph(features1, f_name = './data2/WEI_OV_/', type="OV_gene", method='corr', threshold_pct=99)
    # build_weighted_graph(features2, f_name = './data2/WEI_OV_/', type="OV_methy", method='corr', threshold_pct=99)
    # build_weighted_graph(features3, f_name = './data2/WEI_OV_/', type="OV_mirna", method='corr', threshold_pct=99)
    Gene_Expression = sio.loadmat(BASE_DIR / 'data2' / 'PER_GBM_/' / 'GBM_gene_matrix.mat')['normalize_corr']
    Methy_Expression = sio.loadmat(BASE_DIR / 'data2' / 'PER_GBM_/' / 'GBM_methy_matrix.mat')['normalize_corr']
    Mirna_Expression = sio.loadmat(BASE_DIR / 'data2' / 'PER_GBM_/' / 'GBM_mirna_matrix.mat')['normalize_corr']

    edges1 = sp.coo_matrix(Gene_Expression)
    logging.info("Number of edges in Gene adjacency matrix: %s", edges1.data.shape[0])
    file_write_obj = open(BASE_DIR / 'data2' / 'PER_GBM_/' / "edges_gene_gbm.csv", 'w+')
    for id in np.arange(edges1.data.shape[0]):
        file_write_obj.writelines(np.str_(edges1.row[id]))
        file_write_obj.write(',')
        file_write_obj.writelines(np.str_(edges1.col[id]))
        file_write_obj.write('\n')
    file_write_obj.close()

    edges2=sp.coo_matrix(Methy_Expression)
    logging.info("Number of edges in Methylation adjacency matrix: %s", edges2.data.shape[0])
    file_write_obj = open(BASE_DIR / 'data2' / 'PER_GBM_/' / "edges_methy_gbm.csv", 'w+')
    for id in np.arange(edges2.data.shape[0]):
        file_write_obj.writelines(np.str_(edges2.row[id]))
        file_write_obj.write(',')
        file_write_obj.writelines(np.str_(edges2.col[id]))
        file_write_obj.write('\n')
    file_write_obj.close()

    edges3=sp.coo_matrix(Mirna_Expression)
    logging.info("Number of edges in miRNA adjacency matrix: %s", edges3.data.shape[0])
    file_write_obj = open(BASE_DIR / 'data2' / 'PER_GBM_/' / "edges_mirna_gbm.csv", 'w+')
    for id in np.arange(edges3.data.shape[0]):
        file_write_obj.writelines(np.str_(edges3.row[id]))
        file_write_obj.write(',')
        file_write_obj.writelines(np.str_(edges3.col[id]))
        file_write_obj.write('\n')
    file_write_obj.close()

