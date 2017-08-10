"""
Default configurations for clustering algorithms
"""

from .base_config import BaseConfig
import argparse

class ClusterConfig(BaseConfig):
    def __init__(self):
        super(ClusterConfig, self).__init__()

        self.parser.add_argument('--model', type=str, default='kmeans',
                       help='Model name for clustering, e.g. kmeans | kts')

        self.parser.add_argument('--isTrain', action='store_true',
                help='Is training phase.')
        self.parser.add_argument('--modality_X', type=str, default='camera',
                help='Modality X: e.g. camera or can')
        self.parser.add_argument('--X_feat', type=str, default='feat_fc',
                help='Feature name to use for modality X: feat_fc | recon_camera')
        self.parser.add_argument('--Y_feat', type=str, default='feat',
                help='Feature name to use for modality Y: feat | recon_can')

        self.parser.add_argument('--K', type=int, default=10, 
                help='Number of clusters')
        self.parser.add_argument('--PCA_dim', type=int, default=0, 
                help='Whether to use PCA and the dimensions to keep')

        # for kts
        self.parser.add_argument('--is_clustered', action='store_true',
                help='Whether to perform clustering after KTS')
        self.parser.add_argument('--D', type=int, default=100, 
                help='dimension for BoW')
        self.parser.add_argument('--m', type=int, default=100, 
                help='Number of change points for KTS')
