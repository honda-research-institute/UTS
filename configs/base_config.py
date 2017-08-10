"""
Basic configurations
"""
import os
import argparse

class BaseConfig(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--name', type=str, default='debug',
                        help='name of this experiment')
        self.parser.add_argument('--train_session', type=str, default='all',
                help='session id list for training, e.g. 201704151140,201704141145, use "all" for all sessions')
        self.parser.add_argument('--test_session', type=str, default='all',
                help='session id list for test, e.g. 201704151140,201704141145, use "all" for all sessions')
        self.parser.add_argument('--silent_mode', action='store_true',
                help='Silent mode, no printing')

    def parse(self):
        args = self.parser.parse_args()

        args.UTS_ROOT = '/home/xyang/UTS/'
        args.DATA_ROOT = '/home/xyang/UTS/Data'
        
        args.video_root = os.path.join(args.DATA_ROOT, 'camera/')
        args.sensor_root = os.path.join(args.DATA_ROOT, 'sensor/')
        args.annotation_root = os.path.join(args.DATA_ROOT, 'annotation/')
        args.result_root = os.path.join(args.DATA_ROOT, 'result/')

        if args.train_session == 'all':
            args.train_session = load_session_list(os.path.join(args.DATA_ROOT, 'session_list.txt'))
        elif args.train_session[-3:] == 'txt':
            args.train_session = load_session_list(os.path.join(args.DATA_ROOT, args.train_session))
        else:
            args.train_session = args.train_session.split(',')
        if args.test_session == 'all':
            args.test_session = load_session_list(os.path.join(args.DATA_ROOT, 'test_session.txt'))
        else:
            args.test_session = args.test_session.split(',')

        return args

def load_session_list(path):
    with open(path, 'r') as fin:
        session_ids = fin.read().strip().split('\n')
#        session_ids = [''.join(s.split('-')[:-1]) for s in session_ids]

    return session_ids
