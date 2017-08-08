import os
from data_io import load_data

cfg = type('', (), {})()
cfg.DATA_ROOT = '/home/xyang/UTS/Data'
cfg.video_root = cfg.DATA_ROOT + '/camera/'
cfg.sensor_root = cfg.DATA_ROOT + '/general_sensors/'

path = os.path.join(cfg.DATA_ROOT, 'session_list')

with open(path, 'r') as fin:
    session_ids = fin.read().strip().split('\n')
    session_ids = [''.join(s.split('-')[:-1]) for s in session_ids]

for i, session_id in enumerate(session_ids):
    d1 = load_data(cfg, session_id, 'camera')
    d2 = load_data(cfg, session_id, 'can')

    print i
    if not d1.shape[0] == d2.shape[0]:
        print session_id, d1.shape[0], d2.shape[0]
