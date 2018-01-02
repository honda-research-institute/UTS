"""
Video Retrieval pipeline
"""

import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

from configs.retrieval_config import RetrievalConfig
from sources.encoder import Encoder
from sources.retriever import Retriever

def main():
    cfg = RetrievalConfig().parse()
    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    encoder = Encoder(cfg)
    retriever = Retriever(cfg.method)

    if cfg.modality_X == 'camera':
        data_root = cfg.video_root
    else:
        data_root = cfg.sensor_root

    # load train + val set as target videos
    feat_target = []
    label_target = []
    for session_id in cfg.train_session+cfg.val_session:
        label = pkl.load(open(os.path.join(cfg.annotation_root, session_id+'/annotations_seg.pkl'), 'r'))
        s = label['s'][cfg.event]
        G = label['G'][cfg.event]

#        feat = h5py.File(os.path.join(data_root, session_id+'/'+cfg.X_feat+'.h5'), 'r')['feats'][:, [1,2,3,4,5,6,7]]
        feat = h5py.File(os.path.join(data_root, session_id+'/'+cfg.X_feat+'.h5'), 'r')['feats'][:]
        for i, g in enumerate(G):
            # include background videos
            feat_target.append(encoder.encode(feat[s[i]:s[i+1],:]))
            label_target.append(g)

    feat_target = np.vstack(feat_target)
    label_target = np.vstack(label_target)

    print ("%d target videos in database with dim %d" % (feat_target.shape[0], feat_target.shape[1]))

    # load query videos and retrieve targe videos
    retriever.load_data(feat_target, label_target)

    result = []
    ap_all = 0; count = 0
    for session_id in cfg.test_session:
        label = pkl.load(open(os.path.join(cfg.annotation_root, session_id+'/annotations_seg.pkl'), 'r'))
        s = label['s'][cfg.event]
        G = label['G'][cfg.event]

#        feat = h5py.File(os.path.join(data_root, session_id+'/'+cfg.X_feat+'.h5'), 'r')['feats'][:, [1,2,3,4,5,6,7]]
        feat = h5py.File(os.path.join(data_root, session_id+'/'+cfg.X_feat+'.h5'), 'r')['feats'][:]
        for i, g in enumerate(G):
            if g > 0:
                ap, dist, _ = retriever.retrieve_and_evaluate(encoder.encode(feat[s[i]:s[i+1],:]), g)
                if ap > 0:    # if event exists in database
                    result.append((session_id, cfg.event, i, g, dist))
#                    print ((session_id, cfg.event, i, g, ap))

                    ap_all += ap
                    count += 1

    print "Average AP: %f for %d queries" % ((ap_all/count), count)
    pkl.dump({'result':result, 'label_target':label_target}, 
            open(os.path.join(cfg.result_root, 'result_'+cfg.name+'.pkl'), 'w'))

if __name__ == "__main__":
    main()
