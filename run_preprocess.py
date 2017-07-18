"""
Script for data preprocessing
"""

import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb


from configs.base_config import BaseConfig
from preprocess import AnnotationReader


def main(args):

    # load default configuration
    cfg = BaseConfig()

    if args.annotation:
        annotation_reader = AnnotationReader(cfg.DATA_ROOT)
        annotation_reader.parse_all_annotation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for model training/testing')

#    group = parser.add_mutually_exclusive_group(required=True)
#    group.add_argument('--train', dest='train_stage', action='store_true',
#                       help='Training')
#    group.add_argument('--test', dest='test_stage', action='store_true',
#                       help='Testing')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--can', dest='can', action='store_true',
                       help='CANbus')
    group2.add_argument('--camera', dest='camera', action='store_true',
                       help='camera')
    group2.add_argument('--annotation', dest='annotation', action='store_true',
                       help='prepare annotation')


    args = parser.parse_args()

    main(args)
