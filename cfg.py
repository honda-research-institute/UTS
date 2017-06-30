"""
Defalut configuration for models
"""

class Config():
    def __init__(self):

        self.root = "/home/xyang/project/data/"
        self.logdir = "/home/xyang/project/result/"

        self.model_params = {
                "batch_size": 30,
                "n_input": 8,
                "n_hidden": 50,
                "n_steps": 15,
                "learning_rate": 1.0e-4,
                "n_epochs": 20,
                }
