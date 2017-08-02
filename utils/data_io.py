from os import path
import numpy as np
import pickle as pkl
import h5py

def save_data(data, cfg, session_id, name=None):
    if name is None:
        name = cfg.name

    if cfg.modality == 'can':
        pkl.dump(data, open(path.join(cfg.sensor_root, "{0}/{1}.pkl").format(session_id, name), 'w'))
    
    if cfg.modality == 'camera':
        with h5py.File(path.join(cfg.video_root, "{0}/{1}.pkl").format(session_id, name), 'w') as fout:
            fout.create_dataset('feats', data=data, dtype='float32')


def load_data(cfg, session_id, name='feats'):

    if cfg.modality == 'can':
        d = pkl.load(open(path.join(cfg.sensor_root, "{0}/{1}.pkl").format(session_id,name), 'r'))
#            d = pkl.load(open(path.join(output_path, "lstm_feat_{0}.pkl").format(session_id), 'r'))
    
    if cfg.modality == 'camera':
        fin = h5py.File(path.join(cfg.video_root, "{0}/{1}.h5").format(session_id,name), 'r')
        d = fin['feats'][:]

    if cfg.modality == 'both':
        fin = h5py.File(path.join(cfg.video_root, "{0}/{1}.h5").format(session_id,name), 'r')
        d1 = fin['feats'][:]
        d2 = pkl.load(open(path.join(cfg.sensor_root, "{0}/{1}.pkl").format(session_id,name), 'r'))
        d = (d1, d2)

    return d

def load_data_list(cfg, session_list=None, name='feats'):

    allsessions = []
    vid = []
    if session_list is None:
        session_list = cfg.session_list

    for i, session_id in enumerate(session_list):
        d = load_data(cfg, session_id, name=name)
        allsessions.append(d)
        vid.append(np.ones(d.shape[0]) * i)

    if cfg.modality == 'both':
        d1 = [d[0] for d in allsessions]
        d2 = [d[1] for d in allsessions]
        data = (np.vstack(d1), np.vstack(d2))
    else:
        data = np.vstack(allsessions)

    vid = np.hstack(vid)

    return data, vid

