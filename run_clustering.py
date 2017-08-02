import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

from configs.cluster_config import ClusterConfig
import utils.data_io as data_io


def main():

    # load default configuration
    cfg = ClusterConfig()
    print cfg.name

    if cfg.method == 'kmeans':

        from models.kmeans import KMeansModel
        model = KMeansModel()

        # training stage
        if cfg.is_Train:
        
            # load data
            print ("Loading data ...")
            data, _ = data_io.load_data_list(cfg)

            model.train(data, cfg.K)
            model.save_model(cfg.result_root, cfg.name) 

        else:
            model.load_model(cfg.result_root, cfg.name)

            result = {}
            for session_id in cfg.test_list:
                data = data_io.load_data(cfg, session_id)
                result[session_id] = model.predict(data)

            pkl.dump(result, open(os.path.join(cfg.result_root, 'result_'+cfg.name+'.pkl'), 'w'))

    if cfg.method == 'kts':

        from models.kts import KTSModel
        model = KTSModel(is_clustered=True, K=cfg.K, D=cfg.D)

        result = {}
        for session_id in cfg.test_list:
            data = data_io.load_data(cfg, session_id)
            _, result[session_id] = model.train_and_predict(data, cfg.m)

        pkl.dump(result, open(os.path.join(cfg.result_root, 'result_'+cfg.name+'.pkl'), 'w'))



if __name__ == "__main__":

    main()
