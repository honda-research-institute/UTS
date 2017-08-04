"""
Default configurations for Evaluation
"""

from .base_config import BaseConfig
import argparse

class EvaluateConfig(BaseConfig):
    def __init__(self):
        super(EvaluateConfig, self).__init__()

        self.parser.add_argument('--result_path', help='path to the result file')
        self.parser.add_argument('--method', help='hungarian')
