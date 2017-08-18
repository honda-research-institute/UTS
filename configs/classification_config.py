"""
Default configurations for clustering algorithms
"""

from .base_config import BaseConfig
import argparse

class ClassConfig(BaseConfig):
    def __init__(self):
        super(ClassConfig, self).__init__()

        self.parser.add_argument('--model', type=str, default='logistic',
                       help='Model name for classification. e.g. logistic')

        self.parser.add_argument('--isTrain', action='store_true',
                help='Is training phase.')
        self.parser.add_argument('--modality_X', type=str, default='camera',
                help='Modality X: e.g. camera or can')

