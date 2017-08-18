import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

from configs.classification_config import ClassConfig
import utils.data_io as data_io
from utils.utils import convert_seg

def main():

    # load default configuration
    cfg = ClassConfig().parse()
    print cfg.name

    result_path = os.path.join(cfg.result_root, cfg.name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if cfg.model == 'logistic':
        from models.logistic_regression import LGModel
        model = LGModel(cfg.PCA_dim)
    elif cfg.model == 'svm':
        from models.svm import SVMModel
        model = SVMModel(cfg.PCA_dim)
    elif cfg.model == 'knn':
        from models.knn import KNNModel
        model = KNNModel(cfg.PCA_dim, cfg.n_neighbors)

    else:
        raise NotImplementedError

    # training stage
    if cfg.isTrain:
    
        # load data
        print ("Loading data ...")
        data, _ = data_io.load_data_list(cfg, cfg.train_session, cfg.modality_X, cfg.X_feat)
        label = data_io.load_annotation_list(cfg, cfg.train_session)
        #temporary operation, join two labels
        label = np.max(label, axis=1)

        index = label>0
        data = data[index]
        label = label[index]

        model.train(data, label)
        model.save_model(result_path) 

    # Testing stage
    print "Testing..."
    if not model.feasibility:
        model.load_model(result_path)

    result = {}
    for session_id in cfg.test_session:
        print session_id
        data = data_io.load_data(cfg, session_id, cfg.modality_X, cfg.X_feat)
        pred = model.predict(data)
        result[session_id] = pred

    pkl.dump(result, open(os.path.join(result_path, 'result.pkl'), 'w'))


if __name__ == "__main__":

    main()
