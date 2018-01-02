"""
Build PCA model using background videos in training set
"""

import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

from sklearn.cluster import KMeans

import sys
sys.path.append('../configs')
from base_config import BaseConfig

import mkl
mkl.get_max_threads()

def main():
    cfg = BaseConfig().parse()

    feat_back = []
    skip_session = cfg.val_session+cfg.test_session
    for session_id in cfg.all_session:
        if session_id in skip_session:
            continue
        print session_id

        # load annotations
#        label = pkl.load(open(os.path.join(cfg.annotation_root, session_id+'/annotations.pkl'), 'r'))
#        label = np.sum(label, axis=1)

        # load features
        feat = h5py.File(os.path.join(cfg.sensor_root, session_id+'/feat_norm.h5'), 'r')['feats'][:]
#        feat_back.append(feat[label==0])    # using only background videos
        feat_back.append(feat)

    feat_back = np.vstack(feat_back)

    # KMeans
    K = 20

    print ("KMeans for %d data, from %d dims to %d clusters" % (feat_back.shape[0], feat_back.shape[1], K))
    kmeans = KMeans(n_clusters=K)
    start = time.time()
    kmeans.fit(feat_back)
    end = time.time()
    print ("KMeans Time: %d secs" % (end-start))

    pkl.dump(kmeans, open(os.path.join(cfg.result_root, 'kmeans_featnorm_%d.pkl' % K), 'w'))
    print ("Saved kmeans_featnorm_%d.pkl" % K)

    # transform original features
    for session_id in cfg.all_session:
        print session_id

        # load features
        feat = h5py.File(os.path.join(cfg.sensor_root, session_id+'/feat_norm.h5'), 'r')['feats'][:]
        idx = kmeans.predict(feat)

        new_feat = np.zeros((feat.shape[0], K), dtype='float32')
        for i in range(new_feat.shape[0]):
            new_feat[i,idx[i]] = 1

        pdb.set_trace()
        with h5py.File(os.path.join(cfg.sensor_root, session_id+'/feat_cluster'+str(K)+'.h5'),'w') as fout:
            fout.create_dataset(name='feats', data=new_feat, dtype='float32')
        

if __name__ == '__main__':
    main()
