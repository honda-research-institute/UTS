import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

from configs.cluster_config import ClusterConfig
import utils.data_io as data_io
from utils.utils import convert_seg


def main():

    # load default configuration
    cfg = ClusterConfig().parse()
    print cfg.name

    result_path = os.path.join(cfg.result_root, cfg.name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if cfg.model == 'kmeans':

        from models.kmeans import KMeansModel
        model = KMeansModel()

        # training stage
        if cfg.isTrain:
        
            # load data
            print ("Loading data ...")
            data, _ = data_io.load_data_list(cfg, cfg.train_session, cfg.modality_X, cfg.X_feat)

            model.train(data, cfg.K)
            model.save_model(result_path) 

        # Testing stage
        if not model.feasibility:
            model.load_model(result_path)

        result = {}
        result_seg = {}
        for session_id in cfg.test_session:
            data = data_io.load_data(cfg, session_id, cfg.modality_X, cfg.X_feat)
            result[session_id] = model.predict(data)
            s, G = convert_seg(result[session_id])
            result_seg[session_id] = {}
            result_seg[session_id]['s'] = s
            result_seg[session_id]['G'] = G

        pkl.dump(result, open(os.path.join(result_path, 'result.pkl'), 'w'))
        pkl.dump(result_seg, open(os.path.join(result_path, 'result_seg.pkl'), 'w'))

    if cfg.model == 'kts':

        from models.kts import KTSModel
        model = KTSModel(is_clustered=cfg.is_clustered, K=cfg.K, D=cfg.D)

        result = {}
        result_seg = {}
        for session_id in cfg.test_session:
            print "Session id: " + session_id
            data = data_io.load_data(cfg, session_id, cfg.modality_X, cfg.X_feat)
            cps, label = model.train_and_predict(data, cfg.m)

            _, G = convert_seg(label)

            result[session_id] = label
            result_seg[session_id] = {}
            result_seg[session_id]['s'] = cps    # keep the original kts segments
            result_seg[session_id]['G'] = G


        pkl.dump(result, open(os.path.join(result_path, 'result.pkl'), 'w'))
        pkl.dump(result_seg, open(os.path.join(result_path, 'result_seg.pkl'), 'w'))



if __name__ == "__main__":

    main()
