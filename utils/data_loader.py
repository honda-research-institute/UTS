from os import path
import numpy as np
import pickle as pkl
import h5py

def load_data(cfg, session_id):

    if cfg.modality == 'can':
        d = pkl.load(open(path.join(cfg.sensor_root, "{0}/feats.pkl").format(session_id), 'r'))
#            d = pkl.load(open(path.join(output_path, "lstm_feat_{0}.pkl").format(session_id), 'r'))
    
    if cfg.modality == 'camera':
        fin = h5py.File(path.join(cfg.video_root, "{0}/feats.h5").format(session_id), 'r')
        d = fin['feats'][:]

    return d

def load_data_list(cfg, session_list=None):

    allsessions = []
    if session_list is None:
        session_list = cfg.session_list

    for session_id in session_list:
        d = load_data(cfg, session_id)
        allsessions.append(d)

    data = np.vstack(allsessions)

    return data

