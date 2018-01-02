import os
import numpy, scipy.io
import pickle as pkl

import sys
sys.path.append('../configs')
from base_config import BaseConfig

def main():
    cfg = BaseConfig().parse()

    for session_id in cfg.train_session+cfg.val_session+cfg.test_session:
        print session_id

        # load annotations
        label = pkl.load(open(os.path.join(cfg.annotation_root, session_id+'/annotations.pkl'), 'r'))
        scipy.io.savemat(os.path.join(cfg.annotation_root, session_id+'/annotations.mat'), mdict={'label':label})


        label_seg = pkl.load(open(os.path.join(cfg.annotation_root, session_id+'/annotations_seg.pkl'), 'r'))
        s = label_seg['s']
        G = label_seg['G']
        scipy.io.savemat(os.path.join(cfg.annotation_root, session_id+'/annotations_seg.mat'), mdict={'s':s, 'G':G})

if __name__ == "__main__":
    main()
