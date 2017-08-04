"""
Default configurations for clustering algorithms
"""

from .base_config import BaseConfig
import argparse

class ClusterConfig(BaseConfig):
    def __init__(self):
        super(ClusterConfig, self).__init__()



        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--train', dest='train_stage', action='store_true',
                       help='Training')
        group.add_argument('--test', dest='train_stage', action='store_false',
                       help='Testing')

        group2 = parser.add_mutually_exclusive_group(required=True)
        group2.add_argument('--can', dest='can', action='store_true',
                       help='CANbus')
        group2.add_argument('--camera', dest='camera', action='store_true',
                       help='camera')

        parser.add_argument('--K', dest='K', default=10, type=int,
                       help='Number of clusters')
        parser.add_argument('--PCA', dest='PCA_dim', default=0, type=int,
                       help='Whether to use PCA and the dimensions to keep')
        parser.add_argument('--method', dest='method', default='kmeans', type=str,
                       help='Method for clustering')
        parser.add_argument('--D', dest='D', default=100, type=int,
                       help='dimension for BoW')
        parser.add_argument('--m', dest='m', default=100, type=int,
                       help='change points for KTS')


        args = parser.parse_args()


        if args.name == 'debug':
            print "="*79
            print "Warning!! You're using the debug name"
            print "="*79

        self.is_Train = args.train_stage
        if args.can:
            self.modality = 'can'
        elif args.camera:
            self.modality = 'camera'
        self.K = args.K
        self.PCA = args.PCA_dim
        self.method = args.method
        self.name = args.name+'_'+self.modality
        self.D = args.D
        self.m = args.m

        self.test_list = ['201704151140']    # for example
        #self.test_list = ['201704141145']    # for example
