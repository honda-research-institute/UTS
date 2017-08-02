"""
Script for model training / testing
"""

import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

import tensorflow as tf
from tqdm import tqdm
from basicLSTM import basicLSTM


def load_data(args, session_ids):

    allsessions = []

    if args.can:
        output_path = args.root + 'general_sensors'
        for session_id in session_ids:
            d = pkl.load(open(path.join(output_path, "general_sensors_{0}.pkl").format(session_id), 'r'))
            allsessions.append(d)

        data = np.vstack(allsessions)
    
    if args.camera:
        output_path = args.root + 'camera'
        for session_id in session_ids:
            fin = h5py.File(path.join(output_path, "{0}/feats.h5").format(session_id), 'r')
            d = fin['feats'][:]
            allsessions.append(d)

        data = np.vstack(allsessions)


class ModelRunner(obj):

    def __init__(self, data, cfg):
        """
        Preparation for data reading
        Fill a queue with preprocessed data
        """

        self.cfg = cfg
        self.batch_size = cfg.batch_size

        self.queue = tf.FIFOQueue(dtypes=[[tf.float32]], capacity=self.batch_size * 3)



def main(args):

    # load session ids
    session_list = open(root+'session_list', 'r')
    session_ids = session_list.read().strip().split('\n')
    session_ids = [''.join(s.split('-')[:-1]) for s in session_ids]


    if args.train_stage:
        # load data into memory
        print ("Loading data ...")

        data = load_data(args, session_ids)

        train(data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for model training/testing')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', dest='train_stage', action='store_true',
                       help='Training')
    group.add_argument('--test', dest='test_stage', action='store_true',
                       help='Testing')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--can', dest='can', action='store_true',
                       help='CANbus')
    group2.add_argument('--camera', dest='camera', action='store_true',
                       help='camera')

    args = parser.parse_args()

    main(args)
