"""
Default configurations for model training 
"""

from .base_config import BaseConfig
import argparse

class TrainConfig(BaseConfig):
    def __init__(self):
        super(TrainConfig, self).__init__()

        self.parser.add_argument('--model', type=str, default='seq2seq_recon',
                       help='Model name for training, e.g. seq2seq_recon | ')

        self.parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
        self.parser.add_argument('--max_time', type=int, default=15,
                       help='Maximum sequence length')
        self.parser.add_argument('--n_predict', type=int, default=1,
                       help='The length of prediction sequence')
        self.parser.add_argument('--n_hidden', type=int, default=200,
                       help='The size of hidden layer')
        self.parser.add_argument('--no_reverse', action='store_true',
                       help='If use no_reverse, the reconstruction sequence is not reversed')
        self.parser.add_argument('--no_shuffle', action='store_true',
                       help='If use no_shuffle, the data will not be shuffle when training')
        self.parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='rmsprop',
                       help='Optimizer used for training, e.g., rmsprop, adam')
        self.parser.add_argument('--n_input', type=int, default=0,
                       help='The dimension of input feature')
        self.parser.add_argument('--n_output', type=int, default=0,
                       help='The dimension of output feature')
        self.parser.add_argument('--modality_X', type=str, default='camera',
                help='Modality X: e.g. camera or can')
        self.parser.add_argument('--modality_Y', type=str, default='can',
                help='Modality Y: e.g. camera or can')

        self.parser.add_argument('--isTrain', action='store_true',
                help='Is training phase.')
        self.parser.add_argument('--continue_train', action='store_true',
                help='Is continue training.')
        self.parser.add_argument('--snapshot_num', type=int, default=-1,
                help='The model snapshot used for feature extraction, -1 for the latest model')
        self.parser.add_argument('--n_epochs', type=int, default=10,
                help='Number of epochs for training')
        self.parser.add_argument('--iter_mode', type=str, default='recon',
                help='Iteration mode: recon | pred')
        self.parser.add_argument('--gpu', type=str, default=None,
                help='Set CUDA_VISIBLE_DEVICES')

