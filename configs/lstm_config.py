"""
Default configurations for clustering algorithms
"""

from .base_config import BaseConfig
import argparse

class LSTMConfig(BaseConfig):
    def __init__(self):
        super(LSTMConfig, self).__init__()

        parser = argparse.ArgumentParser()

        parser.add_argument('--name', type=str, default='lstm_debug',
                        help='name of this experiment')

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
        group2.add_argument('--both', dest='both', action='store_true',
                       help='cross modality')

        parser.add_argument('--seq2seq', dest='seq2seq', action='store_true',
                       help='Whether seq2seq model')
        parser.add_argument('--reconstruct', dest='reconstruct', default=True, type=bool,
                       help='Whether reconstruct')
        parser.add_argument('--predict', dest='predict', default=False, type=bool,
                       help='Whether predict')
        parser.add_argument('--reverse', dest='reverse', default=True, type=bool,
                       help='Whether to reverse reconstruction sequence')

        parser.add_argument('--n_input', dest='n_input', default=1537, type=int,
                       help='dimensionality of input')
        parser.add_argument('--batch_size', dest='batch_size', default=10, type=int,
                       help='batch size')
        parser.add_argument('--n_steps', dest='n_steps', default=10, type=int,
                       help='length of encoded sequence')
        parser.add_argument('--n_predict', dest='n_predict', default=5, type=int,
                       help='length of decoded sequence')
        parser.add_argument('--n_hidden', dest='n_hidden', default=200, type=int,
                       help='size of hidden layer')
        parser.add_argument('--n_epochs', dest='n_epochs', default=10, type=int,
                       help='number of epochs')
        parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4,
                       help='learning rate')


        args = parser.parse_args()


        if args.name == 'lstm_debug':
            print "="*79
            print "Warning!! You're using the debug name"
            print "="*79

        self.is_Train = args.train_stage
        if args.can:
            self.modality = 'can'
        elif args.camera:
            self.modality = 'camera'
        elif args.both:
            self.modality = 'both'
        self.name = args.name+'_'+self.modality

        self.seq2seq = args.seq2seq
        self.reconstruct = args.reconstruct
        self.reverse = args.reverse
        self.predict = args.predict

        self.n_input = args.n_input
        self.batch_size = args.batch_size
        self.n_steps = args.n_steps
        self.n_predict = args.n_predict
        self.n_hidden = args.n_hidden
        self.n_epochs = args.n_epochs
        self.learning_rate = args.learning_rate

#        self.test_list = ['201704151140']    # for example
        self.session_list = ['201704141145']    # for example
