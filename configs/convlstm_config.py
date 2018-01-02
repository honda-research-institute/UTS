"""
Default configurations for model training 
"""

from base_config import BaseConfig
import argparse

class ConvLSTMConfig(BaseConfig):
    def __init__(self):
        super(ConvLSTMConfig, self).__init__()

        self.parser.add_argument('--event', type=int, default=1,
                       help='which event')
        self.parser.add_argument('--sampling', type=str, default='random',
                help='sampling strategy: eg. random| uniform')
        self.parser.add_argument('--is_classify', action='store_true',
                help='if specify, use cross entropy loss.')
        self.parser.add_argument('--margin', type=float, default=1.,
                       help='margin')
        self.parser.add_argument('--cluster_name', type=str, default=None,
                help='name of clustering model')

        self.parser.add_argument('--input_keep_prob', type=float, default=1.,
                       help='input_keep_prob')
        self.parser.add_argument('--output_keep_prob', type=float, default=1.,
                       help='output_keep_prob')
        self.parser.add_argument('--n_threads', type=int, default=1,
                       help='number of threads for loading data in parallel')
        self.parser.add_argument('--buffer_size', type=int, default=10,
                       help='buffer_size')

        self.parser.add_argument('--n_epochs', type=int, default=10,
                help='Number of epochs for training')
        self.parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
        self.parser.add_argument('--max_time', type=int, default=90,
                       help='Maximum sequence length')
        self.parser.add_argument('--n_predict', type=int, default=1,
                       help='The length of prediction sequence')
        self.parser.add_argument('--n_hidden', type=int, default=200,
                       help='The size of hidden layer')
        self.parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer used for training, e.g., rmsprop, adam')
        self.parser.add_argument('--n_input', type=int, default=1536,
                       help='The number of channels of input feature')
        self.parser.add_argument('--n_h', type=int, default=8,
                       help='Height of input feature')
        self.parser.add_argument('--n_w', type=int, default=8,
                       help='Width of input feature')
        self.parser.add_argument('--n_C', type=int, default=20,
                       help='The number of output channels')
        self.parser.add_argument('--n_output', type=int, default=1,
                       help='The dimension of output feature')
        self.parser.add_argument('--modality_X', type=str, default='camera',
                help='Modality X: e.g. camera or can')
        self.parser.add_argument('--modality_Y', type=str, default=None,
                help='Modality Y: e.g. camera or can, if None is specified, predict event labels')
        self.parser.add_argument('--is_conditioned', action='store_true',
                help='if specified, use Conditional decoder.')
        self.parser.add_argument('--data_mode', type=str, default='pool',
                help='Data mode for dealing with sensor data: pool | next')


        self.parser.add_argument('--isTrain', action='store_true',
                help='Is training phase.')
        self.parser.add_argument('--continue_train', action='store_true',
                help='Is continue training.')
        self.parser.add_argument('--snapshot_num', type=int, default=-1,
                help='The model snapshot used for feature extraction, -1 for the latest model')
        self.parser.add_argument('--gpu', type=str, default=None,
                help='Set CUDA_VISIBLE_DEVICES')

