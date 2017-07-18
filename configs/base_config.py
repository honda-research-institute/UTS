"""
Basic configurations
"""
import os

class BaseConfig(object):
    def __init__(self):

        self.UTS_ROOT = '/home/xyang/UTS/'
        self.DATA_ROOT = '/home/xyang/UTS/Data'
        
        self.video_root = os.path.join(self.DATA_ROOT, 'camera/')
        self.sensor_root = os.path.join(self.DATA_ROOT, 'general_sensors/')
        self.annotation_root = os.path.join(self.DATA_ROOT, 'annotation/')
        self.result_root = os.path.join(self.DATA_ROOT, 'result/')

#        self.session_list = load_session_list(os.path.join(self.DATA_ROOT, 'session_list'))
        self.session_list = ['201704151140']    # for example

def load_session_list(path):
    with open(path, 'r') as fin:
        session_ids = fin.read().strip().split('\n')
        session_ids = [''.join(s.split('-')[:-1]) for s in session_ids]

    return session_ids
