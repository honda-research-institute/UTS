"""
Default configurations for retrieval pipeline
"""

from .base_config import BaseConfig
import argparse

class RetrievalConfig(BaseConfig):
    def __init__(self):
        super(RetrievalConfig, self).__init__()

        self.parser.add_argument('--encoder', type=str, default='pool',
                       help='method for sequence encoding, e.g. pool (average pooling) | concat | feat | pred')
        self.parser.add_argument('--method', type=str, default='cos',
                       help='method for similarity matching, e.g. cos')

        self.parser.add_argument('--event', type=int, default=1, 
                help='0: Stimulate oriented events; 1: Goal oriented events')
        self.parser.add_argument('--model_name', type=str, default='debug',
                       help='model name')
        self.parser.add_argument('--snapshot_num', type=int, default=-1,
                help='The model snapshot used for feature extraction, -1 for the latest model')

        self.parser.add_argument('--gpu', type=str, default=None,
                help='Set CUDA_VISIBLE_DEVICES')
        self.parser.add_argument('--modality_X', type=str, default='camera',
                help='Modality X: e.g. camera or can')
