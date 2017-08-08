from os import path
import numpy as np
import pickle as pkl
import h5py
import utils

class CreateDataset(object):
    """
    Build dataset
    """

    def __init__(self, cfg):
        print "Loading Data..."

        self.X, self.vid_X = load_data_list(cfg, cfg.train_session, cfg.modality_X)
        self.Y, self.vid_Y = load_data_list(cfg, cfg.train_session, cfg.modality_Y)
        self.iter_mode = cfg.iter_mode
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.shuffle = not cfg.no_shuffle
        self.reverse = not cfg.no_reverse
        self.batch_size = cfg.batch_size
        self.max_time = cfg.max_time
        self.n_predict = cfg.n_predict

        self.N = self.X.shape[0]
        self.X_dim = self.X.shape[1]
        self.Y_dim = self.Y.shape[1]

    def get_iterator(self):
        if self.iter_mode == 'recon':
            iterator = utils.recon_minibatch(self.X, self.vid_X, self.Y, 
                    self.batch_size, self.max_time, self.shuffle, self.reverse)
        if self.iter_mode == 'pred':
            iterator = utils.pred_minibatch(self.X, self.vid_X, self.Y, 
                    self.batch_size, self.max_time, self.n_predict, self.shuffle)
        else:
            raise ValueError("Iter_mode %s not recognized" % self.iter_mode)

        return iterator

    def print_config(self):
        print "="*79

        print "Dataset configurations:"
        print "Modality X: %s, Modality Y: %s, Iteration mode: %s" % (self.modality_X,
                self.modality_Y, self.iter_mode)
        print "N: ", self.N
        print "X_dim: ", self.X_dim
        print "Y_dim: ", self.Y_dim

        print "shuffle: ", self.shuffle
        print "reverse: ", self.reverse
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "n_predict: ", self.n_predict

        print "="*79


def save_feat(data, cfg, session_id, name=None):
    """
    Save new features extracted by pre-trained model for modality X
    """

    if name is None:
        name = cfg.name

    if cfg.modality_X == 'can':
        pkl.dump(data, open(path.join(cfg.sensor_root, "{0}/{1}.pkl").format(session_id, name), 'w'))
    
    if cfg.modality_X == 'camera':
        with h5py.File(path.join(cfg.video_root, "{0}/{1}.h5").format(session_id, name), 'w') as fout:
            fout.create_dataset('feats', data=data, dtype='float32')


def load_data(cfg, session_id, modality, name='feats'):

    if modality == 'can':
        d = pkl.load(open(path.join(cfg.sensor_root, "{0}/{1}.pkl").format(session_id,name), 'r'))
#            d = pkl.load(open(path.join(output_path, "lstm_feat_{0}.pkl").format(session_id), 'r'))
    
    if modality == 'camera':
        fin = h5py.File(path.join(cfg.video_root, "{0}/{1}.h5").format(session_id,name), 'r')
        d = fin['feats'][:]

    return d

def load_data_list(cfg, session_list, modality, name='feats'):

    allsessions = []
    vid = []

    for i, session_id in enumerate(session_list):
        d = load_data(cfg, session_id, modality, name=name)
        allsessions.append(d)
        vid.append(np.ones(d.shape[0]) * i)

    data = np.vstack(allsessions)
    vid = np.hstack(vid)

    return data, vid

