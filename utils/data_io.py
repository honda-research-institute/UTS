from os import path
import os
import numpy as np
import pickle as pkl
import h5py
import utils
import tensorflow as tf
import pdb

class DataGeneratorSensorSeq(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Feature + Sensor for seq2seq
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.is_classify = cfg.is_classify
        self.n_epochs = cfg.n_epochs
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        else:
            self.Y_root = cfg.sensor_root

        self.sessions = cfg.train_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.session_len = []
        for session_id in self.sessions:
            with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
                self.session_len.append(fin['feats'].shape[0])

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        session_list = []
        frame_list = []
        len_list = []

        session_len = np.asarray(self.session_len, dtype='float32')
        N = int(np.sum(session_len)) * 1 / self.max_time    # roughly cover whole dataset with stride of 1/2 * max_time
        N = (N // self.batch_size) * self.batch_size
        session_len = session_len / np.sum(session_len)    # normalize to prob
        # randomly sample from multinomial distribution
        exp_multinomial = np.random.multinomial(N, session_len)

        sessions = np.asarray(self.sessions)
        for i in range(len(self.session_len)):
            session_temp = sessions[np.ones((exp_multinomial[i],),dtype='int32') * i]
            frame_temp = np.zeros((exp_multinomial[i],),dtype='int32')
            len_temp = np.zeros((exp_multinomial[i],),dtype='int32')
            for j in range(exp_multinomial[i]):
                # randomly sample a batch of end frames  [:frame_temp[j]]
                frame_temp[j] = np.random.randint(10, self.session_len[i]-1)
                len_temp[j] = min(self.max_time, frame_temp[j]-1)

            session_list.append(session_temp)
            frame_list.append(frame_temp)
            len_list.append(len_temp)

        session_list= np.hstack(session_list)
        frame_list = np.hstack(frame_list)
        len_list = np.hstack(len_list)
        self.data_size = session_list.shape[0]
        idx = np.arange(self.data_size)
        np.random.shuffle(idx)

        dataset = Dataset.from_tensor_slices((session_list[idx], 
                                              frame_list[idx],
                                              len_list[idx]))
        dataset = dataset.map(lambda session_id, frame, seq_len: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame, seq_len],
                                    [tf.float32, tf.int32 if self.is_classify else tf.float32, 
                                     tf.int32, tf.int32 if self.is_classify else tf.float32])),
                              num_threads = self.n_threads,
                              output_buffer_size = self.buffer_size*self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame, seq_len):
        # load features according to session_id and frame indices
        X_feats = np.zeros((self.max_time,)+self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats[:seq_len] = fin['feats'][frame-seq_len:frame]

        if not self.is_classify:
            Y_feats = np.zeros((self.max_time, self.n_output), dtype='float32')
            p0 = np.zeros((self.n_output,), dtype='float32')
        with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
            Y_feats[:seq_len] = fin['feats'][frame-seq_len:frame]
            p0 = fin['feats'][frame-seq_len-1]

        return X_feats, Y_feats, seq_len, p0

    def print_config(self):
        print "="*79

        print "Dataset configurations: DataGeneratorSensorSeq"
        print "n_epochs: ", self.n_epochs
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat

        print "="*79

class DataGeneratorSensorSeqTrimmed(object):
    """
    Feature + Sensor, Seq2seq
    Use label as segmentation
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.event = cfg.event
        self.is_classify = cfg.is_classify
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.sampling = cfg.sampling
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain:
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session
            self.sampling = 'random'

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.data = self.create_dataset()

    def create_dataset(self):

        session_list = []
        frame_list = []
        label_list = []
        for session_id in self.sessions:
            # load annotations
            label = pkl.load(open(os.path.join(self.annotation_root, session_id+'/annotations_seg.pkl'), 'r'))
            s = label['s'][self.event]
            G = label['G'][self.event]

            for i, g in enumerate(G):
                l = s[i+1] - s[i]
                session_list.append(session_id)
                label_list.append(g)
                if l >= self.max_time:
                    frame_list.append((s[i+1]-self.max_time, s[i+1]))
                else:
                    frame_list.append((s[i], s[i+1]))
        session_list = np.asarray(session_list)
        frame_list = np.asarray(frame_list, dtype='int32')
        label_list = np.asarray(label_list, dtype='int32')

        # construct data_list according to sampling strategy
        idx_list = []
        indices = np.arange(len(session_list))

        for i in range(len(session_list) // self.batch_size):

            if self.sampling == 'uniform':
                label_batch = np.random.randint(0, np.max(label_list), self.batch_size)
                idx_batch = np.zeros((self.batch_size,),dtype='int32')
                for j in range(self.batch_size):
                    idx = np.where(label_list==label_batch[j])[0]
                    idx = idx[np.random.randint(0, len(idx))]
                    idx_batch[j] = idx
            else:
                idx_batch = indices[i*self.batch_size:(i+1)*self.batch_size]

            idx_list.append(idx_batch)
        idx_list = np.hstack(idx_list)
        self.data_size = idx_list.shape[0]

        dataset = Dataset.from_tensor_slices((session_list[idx_list], 
                                              frame_list[idx_list],
                                              label_list[idx_list]))
        # fix doc issue according to https://github.com/tensorflow/tensorflow/issues/11786
        dataset = dataset.map(lambda session_id, frame, label: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame, label],
                                    [tf.float32, tf.int32 if self.is_classify else tf.float32, 
                                     tf.int32, tf.int32 if self.is_classify else tf.float32])),
                              num_threads = self.n_threads,
                              output_buffer_size = self.buffer_size*self.batch_size)
        if self.isTrain:
            dataset = dataset.shuffle(buffer_size=1000)
        return dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame, label):
        # load features according to session_id and frame indices
        X_feats = np.zeros((self.max_time,)+self.dim, dtype='float32')
        l = frame[1] - frame[0]
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats[:l] = fin['feats'][frame[0]:frame[1]]

        if not self.is_classify:
            Y_feats = np.zeros((self.max_time, self.n_output), dtype='float32')
            p0 = np.zeros((self.n_output,), dtype='float32')
        with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
            Y_feats[:l] = fin['feats'][frame[0]:frame[1]]
            p0 = fin['feats'][frame[0]-1]

        return X_feats, Y_feats, l, p0

    def print_config(self):
        print "="*79

        print "Dataset configurations: DataGeneratorSensorSeqTrimmed"
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "event: ", self.event
        print "sampling: ", self.sampling

        print "="*79

class DataGeneratorSensorTrimmed(object):
    """
    Feature + Sensor
    Use label as segmentation
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.event = cfg.event
        self.is_classify = cfg.is_classify
        self.n_epochs = cfg.n_epochs
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.sampling = cfg.sampling
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain:
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session
            self.sampling = 'random'
            self.n_epochs = 1

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.data = self.create_dataset()

    def create_dataset(self):

        session_list = []
        frame_list = []
        label_list = []
        for session_id in self.sessions:
            # load annotations
            label = pkl.load(open(os.path.join(self.annotation_root, session_id+'/annotations_seg.pkl'), 'r'))
            s = label['s'][self.event]
            G = label['G'][self.event]

            for i, g in enumerate(G):
                l = s[i+1] - s[i]
                session_list.append(session_id)
                label_list.append(g)
                if l >= self.max_time:
                    frame_list.append((s[i+1]-self.max_time, s[i+1]))
                else:
                    frame_list.append((s[i], s[i+1]))
        session_list = np.asarray(session_list)
        frame_list = np.asarray(frame_list, dtype='int32')
        label_list = np.asarray(label_list, dtype='int32')

        # construct data_list according to sampling strategy
        idx_list = []
        indices = np.arange(len(session_list))

        for i in range(len(session_list) // self.batch_size):

            if self.sampling == 'uniform':
                label_batch = np.random.randint(0, np.max(label_list), self.batch_size)
                idx_batch = np.zeros((self.batch_size,),dtype='int32')
                for j in range(self.batch_size):
                    idx = np.where(label_list==label_batch[j])[0]
                    idx = idx[np.random.randint(0, len(idx))]
                    idx_batch[j] = idx
            else:
                idx_batch = indices[i*self.batch_size:(i+1)*self.batch_size]

            idx_list.append(idx_batch)
        idx_list = np.hstack(idx_list)
        self.data_size = idx_list.shape[0]

        dataset = Dataset.from_tensor_slices((session_list[idx_list], 
                                              frame_list[idx_list],
                                              label_list[idx_list]))
        # fix doc issue according to https://github.com/tensorflow/tensorflow/issues/11786
        dataset = dataset.map(lambda session_id, frame, label: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame, label],
                                    [tf.float32, tf.int32 if self.is_classify else tf.float32, 
                                     tf.int32])),
                              num_threads = self.n_threads,
                              output_buffer_size = self.buffer_size*self.batch_size)
        if self.isTrain:
            dataset = dataset.shuffle(buffer_size=1000)
        return dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame, label):
        # load features according to session_id and frame indices
        X_feats = np.zeros((self.max_time,)+self.dim, dtype='float32')
        l = frame[1] - frame[0]
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats[:l] = fin['feats'][frame[0]:frame[1]]

        with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
#            Y_feats = fin['feats'][frame+1]
             
            # for predicting pooled sensor
            Y_feats = np.mean(fin['feats'][frame[0]:frame[1]], 0)

        return X_feats, Y_feats, l

    def print_config(self):
        print "="*79

        print "Dataset configurations: DataGeneratorSensorTrimmed"
        print "n_epochs: ", self.n_epochs
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "event: ", self.event
        print "sampling: ", self.sampling

        print "="*79


class DataGeneratorSensorRank(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Feature + Sensor for ranking loss
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.data_mode = cfg.data_mode

        self.cluster = pkl.load(open(os.path.join(cfg.result_root, cfg.cluster_name), 'r'))
        print "Cluster %s loaded" % cfg.cluster_name

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain or cfg.continue_train:
            self.isTrain = True
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.session_len = []
        for session_id in self.sessions:
            with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
                self.session_len.append(fin['feats'].shape[0])

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        session_list = []
        frame_list = []
        len_list = []

        session_len = np.asarray(self.session_len, dtype='float32')
        N = int(np.sum(session_len)) * 2 / self.max_time    # roughly cover whole dataset with stride of 1/2 * max_time
        N = (N // self.batch_size) * self.batch_size
        session_len = session_len / np.sum(session_len)    # normalize to prob
        # randomly sample from multinomial distribution
        exp_multinomial = np.random.multinomial(N, session_len)

        sessions = np.asarray(self.sessions)
        for i in range(len(self.session_len)):
            session_temp = sessions[np.ones((exp_multinomial[i],),dtype='int32') * i]
            frame_temp = np.zeros((exp_multinomial[i],),dtype='int32')
            len_temp = np.zeros((exp_multinomial[i],),dtype='int32')
            for j in range(exp_multinomial[i]):
                # randomly sample a batch of end frames  [:frame_temp[j]]
                frame_temp[j] = np.random.randint(10, self.session_len[i]-1)
                len_temp[j] = min(self.max_time, frame_temp[j]-1)

            session_list.append(session_temp)
            frame_list.append(frame_temp)
            len_list.append(len_temp)

        session_list= np.hstack(session_list)
        frame_list = np.hstack(frame_list)
        len_list = np.hstack(len_list)
        self.data_size = session_list.shape[0]
        idx = np.arange(self.data_size)
        idx1 = np.random.permutation(idx)
        idx2 = np.random.permutation(idx1)

        dataset = tf.data.Dataset.from_tensor_slices((session_list[idx1],session_list[idx2],
                                              frame_list[idx1],frame_list[idx2],
                                              len_list[idx1], len_list[idx2]))
        dataset = dataset.map(lambda session_id1, session_id2, frame1, frame2, seq_len1, seq_len2: tuple(tf.py_func(
                                    self._input_parser, [session_id1, session_id2, frame1, frame2, seq_len1, seq_len2],
                                    [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, session_id1, session_id2, frame1, frame2, seq_len1, seq_len2):
        # load features according to session_id and frame indices
        X_feats1 = np.zeros((self.max_time,)+self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id1+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats1[:seq_len1] = fin['feats'][frame1-seq_len1:frame1]

        X_feats2 = np.zeros((self.max_time,)+self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id2+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats2[:seq_len2] = fin['feats'][frame2-seq_len2:frame2]

        with h5py.File(os.path.join(self.Y_root, session_id1+'/'+self.Y_feat+'.h5'), 'r') as fin:
            if self.data_mode == 'next':
                Y_feats1 = fin['feats'][frame1+1]
            elif self.data_mode == 'pool':
                # for predicting pooled sensor
                Y_feats1 = np.mean(fin['feats'][frame1-seq_len1:frame1], 0)
        with h5py.File(os.path.join(self.Y_root, session_id2+'/'+self.Y_feat+'.h5'), 'r') as fin:
            if self.data_mode == 'next':
                Y_feats2 = fin['feats'][frame2+1]
            elif self.data_mode == 'pool':
                # for predicting pooled sensor
                Y_feats2 = np.mean(fin['feats'][frame2-seq_len2:frame2], 0)
        c1 = self.cluster.predict(Y_feats1.reshape(1,-1))
        c2 = self.cluster.predict(Y_feats2.reshape(1,-1))

        return X_feats1, X_feats2, np.int32(c1==c2), seq_len1, seq_len2

    def print_config(self):
        print "="*79

        print "Dataset configurations: DataGeneratorSensorRank"
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat
        print "isTrain: ", self.isTrain
        print "Number of sessions: %d" % len(self.sessions)

        print "="*79


class SeqGenerator(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Sequence feature + Sensor / label
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.is_classify = cfg.is_classify
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.event = cfg.event
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output
        self.annotation_root = cfg.annotation_root

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        elif cfg.modality_Y == 'label':
            self.Y_root = cfg.annotation_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain or cfg.continue_train:
            self.isTrain = True
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.session_len = []
        for session_id in self.sessions:
            with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
                self.session_len.append(fin['feats'].shape[0])

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        session_list = []
        frame_list = []
        sessions = np.asarray(self.sessions)
        for i in xrange(len(self.sessions)):
            start_idx = []
            seed = np.random.randint(self.max_time)
            for j in range(seed, self.session_len[i]-self.max_time, self.max_time):
                start_idx.append(j)

            session_list.append(sessions[np.ones((len(start_idx),),dtype='int32')*i])
            frame_list.append(np.asarray(start_idx))

        session_list= np.hstack(session_list)
        frame_list = np.hstack(frame_list)

        self.data_size = session_list.shape[0] // self.batch_size * self.batch_size
        idx = np.arange(self.data_size)
        if self.isTrain:
            np.random.shuffle(idx)

        dataset = tf.data.Dataset.from_tensor_slices((session_list[idx], 
                                              frame_list[idx]))
        dataset = dataset.map(lambda session_id, frame: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame],
                                    [tf.float32, tf.float32, tf.int32])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame):
        # load features according to session_id and frame indices
        X_feats = np.zeros((self.max_time,)+self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats = fin['feats'][frame:frame+self.max_time]

        Y_feats = np.zeros((self.max_time, self.n_output), dtype='float32')
        if self.modality_Y == 'label':
            label = pkl.load(open(os.path.join(self.Y_root, session_id+'/annotations.pkl'), 'r'))
            for i in range(self.max_time):
                Y_feats[i, label[frame+i, self.event]] = 1
        else:
            with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
                if self.n_output == 6:
                    Y_feats = fin['feats'][frame:frame+self.max_time, [0,1,2,3,4,7]]
                else:
                    Y_feats = fin['feats'][frame:frame+self.max_time]

        seq_len = np.ones((1,), dtype='int32') * self.max_time
        return X_feats, Y_feats, seq_len

    def print_config(self):
        print "="*79

        print "Dataset configurations: SeqGenerator"
        print "batch_size: ", self.batch_size
        print "data_size: ", self.data_size
        print "max_time: ", self.max_time
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "event: ", self.event
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat
        print "isTrain: ", self.isTrain
        print "Number of sessions: %d" % len(self.sessions)

        print "="*79


class SeqGeneratorTrimmed(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Sequence feature + Sensor / label
    Background frames are reduced according to labels
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.is_classify = cfg.is_classify
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.event = cfg.event
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output
        self.annotation_root = cfg.annotation_root

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        elif cfg.modality_Y == 'label':
            self.Y_root = cfg.annotation_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain or cfg.continue_train:
            self.isTrain = True
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        session_list = []
        frame_list = []
        sessions = np.asarray(self.sessions)
        for i in xrange(len(self.sessions)):
            label = pkl.load(open(os.path.join(self.annotation_root, self.sessions[i]+'/annotations.pkl'), 'r'))[:, self.event]

            start_idx = []
            seed = np.random.randint(self.max_time)
            for j in range(seed, label.shape[0]-self.max_time, self.max_time):
                if np.sum(label[j:j+self.max_time]) > 0:
                    start_idx.append(j)

            session_list.append(sessions[np.ones((len(start_idx),),dtype='int32')*i])
            frame_list.append(np.asarray(start_idx, dtype='int32'))

        session_list= np.hstack(session_list)
        frame_list = np.hstack(frame_list)

        self.data_size = session_list.shape[0] // self.batch_size * self.batch_size
        idx = np.arange(self.data_size)
        if self.isTrain:
            np.random.shuffle(idx)

        dataset = tf.data.Dataset.from_tensor_slices((session_list[idx], 
                                              frame_list[idx]))
        dataset = dataset.map(lambda session_id, frame: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame],
                                    [tf.float32, tf.float32, tf.int32])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame):
        # load features according to session_id and frame indices
        X_feats = np.zeros((self.max_time,)+self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats = fin['feats'][frame:frame+self.max_time]

        Y_feats = np.zeros((self.max_time, self.n_output), dtype='float32')
        if self.modality_Y == 'label':
            label = pkl.load(open(os.path.join(self.Y_root, session_id+'/annotations.pkl'), 'r'))
            for i in range(self.max_time):
                Y_feats[i, label[frame+i, self.event]] = 1
        else:
            with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
                if self.n_output == 6:
                    Y_feats = fin['feats'][frame:frame+self.max_time, [0,1,2,3,4,7]]
                else:
                    Y_feats = fin['feats'][frame:frame+self.max_time]

        seq_len = np.ones((1,), dtype='int32') * self.max_time
        return X_feats, Y_feats, seq_len

    def print_config(self):
        print "="*79

        print "Dataset configurations: SeqGeneratorTrimmed"
        print "batch_size: ", self.batch_size
        print "data_size: ", self.data_size
        print "max_time: ", self.max_time
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "event: ", self.event
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat
        print "isTrain: ", self.isTrain
        print "Number of sessions: %d" % len(self.sessions)

        print "="*79

class FrameGeneratorRankTrimmed(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Frame feature + Sensor / label, for ranking loss
    Background frames are reduced according to labels
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.is_classify = cfg.is_classify
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.event = cfg.event
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output
#        self.keep_background = kwargs.get('keep_background', cfg.keep_background)
        self.annotation_root = cfg.annotation_root

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        elif cfg.modality_Y == 'label':
            self.Y_root = cfg.annotation_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain or cfg.continue_train:
            self.isTrain = True
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.session_len = []
        for session_id in self.sessions:
            with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
                self.session_len.append(fin['feats'].shape[0])

        session_list = []
        frame_list = []
        label_list = []
        sessions = np.asarray(self.sessions)
        for i in xrange(len(self.sessions)):
            label = pkl.load(open(os.path.join(self.annotation_root, self.sessions[i]+'/annotations.pkl'), 'r'))[:, self.event]
            # remove the background frames that far away from events
            mask = np.zeros((label.shape[0],), dtype=bool)
            for j in range(label.shape[0]):
                if np.sum(label[max(0, j-15):min(j+15,label.shape[0])]) > 0:
                    mask[j] = True

#            pdb.set_trace()
            session_list.append(sessions[np.ones((mask.sum(),),dtype='int32')*i])
            frame_list.append(np.arange(self.session_len[i], dtype='int32')[mask])

            if self.modality_Y == 'label':
                label = label[mask]
                temp = np.zeros((mask.sum(), 13),dtype='float32')
                for i in range(temp.shape[0]):
                    temp[i, label[i]] = 1
                label_list.append(temp)
            elif self.modality_Y == 'can':
                with h5py.File(os.path.join(self.Y_root, self.sessions[i]+'/'+self.Y_feat+'.h5'), 'r') as fin:
                    label_list.append(fin['feats'][mask])


        self.session_list= np.hstack(session_list)
        self.frame_list = np.hstack(frame_list)
        self.label_list = np.concatenate(label_list, axis=0)

        print "Building label pool..."
        label_pool = {}
        for i in range(self.label_list.shape[1]):
            label_pool[i] = []
        for i in range(self.label_list.shape[0]):
            label_pool[self.label_list[i].argmax()].append((self.session_list[i],
                                                            self.frame_list[i]))
        self.label_pool = label_pool
        self.label_nums = len(self.label_pool.keys())

        self.label_distribution = self.label_list.sum(axis=0) / float(self.label_list.sum())

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        sample_list1 = []
        sample_list2 = []
        match_list = []
        for i in range(self.label_nums):
            cur_set = self.label_pool[i]
            ops_idx = np.arange(self.label_nums)[np.arange(self.label_nums)!=i]
            p = self.label_distribution[ops_idx] / self.label_distribution[ops_idx].sum()

            for j in range(len(cur_set)-1, -1, -1):
                # similar pairs
                sample_list1.append(cur_set[j])
                sample_list2.append(cur_set[np.random.randint(len(cur_set))])
                match_list.append(1)

                # dissimliar pairs
                sample_list1.append(cur_set[j])
                ops_set = self.label_pool[np.random.choice(ops_idx,1,p=p)[0]]
                sample_list2.append(ops_set[np.random.randint(len(ops_set))])
                match_list.append(0)

        self.data_size = len(match_list) // self.batch_size * self.batch_size

        sample_list1 = np.asarray(sample_list1)
        sample_list2 = np.asarray(sample_list2)
        match_list = np.asarray(match_list, dtype='int32')

        # random shuffling if is_traing
        if self.isTrain:
            idx = np.arange(0, self.data_size, 2)
            np.random.shuffle(idx)
            idx = idx.reshape(-1,1)
            idx = np.concatenate((idx,idx+1), axis=1)
            idx = idx.reshape(-1,1)
            idx = np.squeeze(idx)   # very important
        else:
            idx = np.arange(self.data_size)

        dataset = tf.data.Dataset.from_tensor_slices((sample_list1[idx], sample_list2[idx], match_list[idx]))
        dataset = dataset.map(lambda s1,s2,y: tuple(tf.py_func(
                                    self._input_parser, [s1,s2,y],
                                    [tf.float32, tf.float32, tf.int32])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, s1, s2, y):
        # load features according to session_id and frame indices
        X_feats1 = np.zeros(self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, s1[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats1 = fin['feats'][int(s1[1])]

        X_feats2 = np.zeros(self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, s2[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats2 = fin['feats'][int(s2[1])]

        return X_feats1, X_feats2, y

    def print_config(self):
        print "="*79

        print "Dataset configurations: FrameGeneratorRankTrimmed"
        print "batch_size: ", self.batch_size
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "event: ", self.event
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat
        print "isTrain: ", self.isTrain
        print "Number of sessions: %d" % len(self.sessions)

        print "="*79


class FrameGeneratorTrimmed(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Frame feature + Sensor
    Background frames are reduced according to labels
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.is_classify = cfg.is_classify
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.event = cfg.event
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output
#        self.keep_background = kwargs.get('keep_background', cfg.keep_background)
        self.annotation_root = cfg.annotation_root

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        elif cfg.modality_Y == 'label':
            self.Y_root = cfg.annotation_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain or cfg.continue_train:
            self.isTrain = True
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.session_len = []
        for session_id in self.sessions:
            with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
                self.session_len.append(fin['feats'].shape[0])

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        session_list = []
        frame_list = []
        label_list = []
        sessions = np.asarray(self.sessions)
        for i in xrange(len(self.sessions)):
            label = pkl.load(open(os.path.join(self.annotation_root, self.sessions[i]+'/annotations.pkl'), 'r'))[:, self.event]
            # remove the background frames that far away from events
            mask = np.zeros((label.shape[0],), dtype=bool)
            for j in range(label.shape[0]):
                if np.sum(label[max(0, j-30):min(j+30,label.shape[0])]) > 0:
                    mask[j] = True

#            pdb.set_trace()
            session_list.append(sessions[np.ones((mask.sum(),),dtype='int32')*i])
            frame_list.append(np.arange(self.session_len[i], dtype='int32')[mask])
            label_list.append(label[mask])

        session_list= np.hstack(session_list)
        frame_list = np.hstack(frame_list)
        label_list = np.hstack(label_list)

#        # reduce background frames according to keep_background
#        if self.keep_background < 1:
#            idx_background = np.where(label_list == 0)[0]
#            np.random.shuffle(idx_background)
#            idx_remove = idx_background[int(self.keep_background*idx_background.shape[0]):]
#            mask = np.ones(session_list.shape, dtype=bool)
#            mask[idx_remove] = False
#            session_list = session_list[mask]
#            frame_list = frame_list[mask]
#            label_list = label_list[mask]

        self.data_size = session_list.shape[0] // self.batch_size * self.batch_size
        idx = np.arange(self.data_size)
        if self.isTrain:
            np.random.shuffle(idx)

        dataset = tf.data.Dataset.from_tensor_slices((session_list[idx], 
                                              frame_list[idx]))
        dataset = dataset.map(lambda session_id, frame: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame],
                                    [tf.float32, tf.float32])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame):
        # load features according to session_id and frame indices
        X_feats = np.zeros(self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats = fin['feats'][frame]

        Y_feats = np.zeros((self.n_output,), dtype='float32')
        if self.modality_Y == 'label':
            label = pkl.load(open(os.path.join(self.Y_root, session_id+'/annotations.pkl'), 'r'))
            Y_feats[label[frame, self.event]] = 1
        else:
            with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
                if self.n_output == 6:
                    Y_feats = fin['feats'][frame, [0,1,2,3,4,7]]
                else:
                    Y_feats = fin['feats'][frame]

        return X_feats, Y_feats

    def print_config(self):
        print "="*79

        print "Dataset configurations: FrameGeneratorTrimmed"
        print "batch_size: ", self.batch_size
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "event: ", self.event
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat
        print "isTrain: ", self.isTrain
        print "Number of sessions: %d" % len(self.sessions)

        print "="*79

class FrameGenerator(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Frame feature + Sensor
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.is_classify = cfg.is_classify
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.event = cfg.event
        self.X_feat = cfg.X_feat
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        elif cfg.modality_Y == 'label':
            self.Y_root = cfg.annotation_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain or cfg.continue_train:
            self.isTrain = True
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.session_len = []
        for session_id in self.sessions:
            with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
                self.session_len.append(fin['feats'].shape[0])

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        session_list = []
        frame_list = []
        sessions = np.asarray(self.sessions)
        for i in xrange(len(self.sessions)):
            session_list.append(sessions[np.ones((self.session_len[i],),dtype='int32')*i])
            frame_list.append(np.arange(self.session_len[i], dtype='int32'))

        session_list= np.hstack(session_list)
        frame_list = np.hstack(frame_list)
        self.data_size = session_list.shape[0] // self.batch_size * self.batch_size
        idx = np.arange(self.data_size)
        if self.isTrain:
            np.random.shuffle(idx)

        dataset = tf.data.Dataset.from_tensor_slices((session_list[idx], 
                                              frame_list[idx]))
        dataset = dataset.map(lambda session_id, frame: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame],
                                    [tf.float32, tf.float32])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame):
        # load features according to session_id and frame indices
        X_feats = np.zeros(self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats = fin['feats'][frame]

        Y_feats = np.zeros((self.n_output,), dtype='float32')
        if self.modality_Y == 'label':
            label = pkl.load(open(os.path.join(self.Y_root, session_id+'/annotations.pkl'), 'r'))
            Y_feats[label[frame, self.event]] = 1
        else:
            with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
                if self.n_output == 6:
                    Y_feats = fin['feats'][frame, [0,1,2,3,4,7]]
                else:
                    Y_feats = fin['feats'][frame]

        return X_feats, Y_feats

    def print_config(self):
        print "="*79

        print "Dataset configurations: FrameGenerator"
        print "batch_size: ", self.batch_size
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "event: ", self.event
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat
        print "isTrain: ", self.isTrain
        print "Number of sessions: %d" % len(self.sessions)

        print "="*79

class DataGeneratorSensor(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Feature + Sensor
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.is_classify = cfg.is_classify
        self.modality_X = cfg.modality_X
        self.modality_Y = cfg.modality_Y
        self.X_feat = cfg.X_feat
        self.event = cfg.event
        self.Y_feat = cfg.Y_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size
        self.n_output = cfg.n_output
        self.data_mode = cfg.data_mode

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root
        if cfg.modality_Y == 'camera':
            self.Y_root = cfg.video_root
        elif cfg.modality_Y == 'label':
            self.Y_root = cfg.annotation_root
        else:
            self.Y_root = cfg.sensor_root

        if self.isTrain or cfg.continue_train:
            self.isTrain = True
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.session_len = []
        for session_id in self.sessions:
            with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
                self.session_len.append(fin['feats'].shape[0])

        self.create_dataset()

    def create_dataset(self):
        print "Reinitialize dataset ..."

        session_list = []
        frame_list = []
        len_list = []

        session_len = np.asarray(self.session_len, dtype='float32')
        N = int(np.sum(session_len)) * 2 / self.max_time    # roughly cover whole dataset with stride of 1/2 * max_time
        N = (N // self.batch_size) * self.batch_size
        session_len = session_len / np.sum(session_len)    # normalize to prob
        # randomly sample from multinomial distribution
        exp_multinomial = np.random.multinomial(N, session_len)

        sessions = np.asarray(self.sessions)
        for i in range(len(self.session_len)):
            session_temp = sessions[np.ones((exp_multinomial[i],),dtype='int32') * i]
            frame_temp = np.zeros((exp_multinomial[i],),dtype='int32')
            len_temp = np.zeros((exp_multinomial[i],),dtype='int32')
            for j in range(exp_multinomial[i]):
                # randomly sample a batch of end frames  [:frame_temp[j]]
                frame_temp[j] = np.random.randint(10, self.session_len[i]-1)
                len_temp[j] = min(self.max_time, frame_temp[j]-1)

            session_list.append(session_temp)
            frame_list.append(frame_temp)
            len_list.append(len_temp)

        session_list= np.hstack(session_list)
        frame_list = np.hstack(frame_list)
        len_list = np.hstack(len_list)
        self.data_size = session_list.shape[0]
        idx = np.arange(self.data_size)
        np.random.shuffle(idx)

        dataset = tf.data.Dataset.from_tensor_slices((session_list[idx], 
                                              frame_list[idx],
                                              len_list[idx]))
        dataset = dataset.map(lambda session_id, frame, seq_len: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame, seq_len],
                                    [tf.float32, tf.float32, tf.int32])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
#        if self.isTrain:
#            dataset = dataset.shuffle(buffer_size=1000)
        self.data = dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame, seq_len):
        # load features according to session_id and frame indices
        X_feats = np.zeros((self.max_time,)+self.dim, dtype='float32')
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            X_feats[:seq_len] = fin['feats'][frame-seq_len:frame]

        Y_feats = np.zeros((self.n_output,), dtype='float32')
        if self.modality_Y == 'label':
            label = pkl.load(open(os.path.join(self.Y_root, session_id+'/annotations.pkl'), 'r'))
            Y_feats[label[frame,self.event]] = 1
        else:
            with h5py.File(os.path.join(self.Y_root, session_id+'/'+self.Y_feat+'.h5'), 'r') as fin:
                if self.data_mode == 'next':
                    Y_feats = fin['feats'][frame+1]
             
                elif self.data_mode == 'pool':
                    # for predicting pooled sensor
                    Y_feats = np.mean(fin['feats'][frame-seq_len:frame], 0)

        return X_feats, Y_feats, seq_len

    def print_config(self):
        print "="*79

        print "Dataset configurations: DataGeneratorSensor"
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "modality_X: ", self.modality_X
        print "modality_Y: ", self.modality_Y
        print "X_feat: ", self.X_feat
        print "Y_feat: ", self.Y_feat
        print "isTrain: ", self.isTrain
        print "Number of sessions: %d" % len(self.sessions)

        print "="*79



class DataGeneratorBasic(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.
    Feature + Label
    """

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.isTrain = kwargs.get('isTrain', cfg.isTrain)
        self.event = cfg.event
        self.n_epochs = cfg.n_epochs
        self.sampling = cfg.sampling
        self.X_feat = cfg.X_feat
        self.n_threads = cfg.n_threads
        self.buffer_size = cfg.buffer_size

        self.annotation_root = cfg.annotation_root
        if cfg.modality_X == 'camera':
            self.X_root = cfg.video_root
        else:
            self.X_root = cfg.sensor_root

        if self.isTrain:
            self.sessions = cfg.train_session
        else:
            self.sessions = cfg.val_session
            self.sampling = 'random'
            self.n_epochs = 1

        # determine feature shape
        with h5py.File(os.path.join(self.X_root, self.sessions[0]+'/'+self.X_feat+'.h5'), 'r') as fin:
            feat = fin['feats'][0]
            self.dim = feat.shape    # e.g, (1536,) for feat_fc, (8,8,1536) for feat_conv

        self.data = self.create_dataset()

    def create_dataset(self):

        session_list = []
        frame_list = []
        label_list = []
        for session_id in self.sessions:
            # load annotations
            label = pkl.load(open(os.path.join(self.annotation_root, session_id+'/annotations_seg.pkl'), 'r'))
            s = label['s'][self.event]
            G = label['G'][self.event]

            for i, g in enumerate(G):
                l = s[i+1] - s[i]
                session_list.append(session_id)
                label_list.append(g)
                if l >= self.max_time:
                    frame_list.append((s[i+1]-self.max_time, s[i+1]))
                else:
                    frame_list.append((s[i], s[i+1]))
        session_list = np.asarray(session_list)
        frame_list = np.asarray(frame_list, dtype='int32')
        label_list = np.asarray(label_list, dtype='int32')

        # construct data_list according to sampling strategy
        idx_list = []
        indices = np.arange(len(session_list))

        for i in range(len(session_list) // self.batch_size):

            if self.sampling == 'uniform':
                label_batch = np.random.randint(0, np.max(label_list), self.batch_size)
                idx_batch = np.zeros((self.batch_size,),dtype='int32')
                for j in range(self.batch_size):
                    idx = np.where(label_list==label_batch[j])[0]
                    idx = idx[np.random.randint(0, len(idx))]
                    idx_batch[j] = idx
            else:
                idx_batch = indices[i*self.batch_size:(i+1)*self.batch_size]

            idx_list.append(idx_batch)
        idx_list = np.hstack(idx_list)
        self.data_size = idx_list.shape[0]

        dataset = tf.data.Dataset.from_tensor_slices((session_list[idx_list], 
                                              frame_list[idx_list],
                                              label_list[idx_list]))
        # fix doc issue according to https://github.com/tensorflow/tensorflow/issues/11786
        dataset = dataset.map(lambda session_id, frame, label: tuple(tf.py_func(
                                    self._input_parser, [session_id, frame, label],
                                    [tf.float32, frame_list.dtype, label_list.dtype])),
                              num_parallel_calls = self.n_threads)
        dataset = dataset.prefetch(self.buffer_size * self.batch_size)
        if self.isTrain:
            dataset = dataset.shuffle(buffer_size=100)
        return dataset.batch(self.batch_size)

    def _input_parser(self, session_id, frame, label):
        # load features according to session_id and frame indices
        feats = np.zeros((self.max_time,)+self.dim, dtype='float32')
        l = frame[1] - frame[0]
        with h5py.File(os.path.join(self.X_root, session_id+'/'+self.X_feat+'.h5'), 'r') as fin:
            feats[:l] = fin['feats'][frame[0]:frame[1]]

        return feats, label, l

    def print_config(self):
        print "="*79

        print "Dataset configurations:"
        print "n_epochs: ", self.n_epochs
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "data_size: ", self.data_size
        print "n_threads: ", self.n_threads
        print "buffer_size: ", self.buffer_size
        print "event: ", self.event
        print "sampling: ", self.sampling

        print "="*79



class CreateDataset(object):
    """
    Build dataset
    """

    def __init__(self, cfg):
        print "Loading Data..."

        self.X, self.vid_X = load_data_list(cfg, cfg.train_session, cfg.modality_X, cfg.X_feat)
        self.Y, self.vid_Y = load_data_list(cfg, cfg.train_session, cfg.modality_Y, cfg.Y_feat)
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
        elif self.iter_mode == 'pred':
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
        with h5py.File(path.join(cfg.sensor_root, "{0}/{1}.h5").format(session_id, name), 'w') as fout:
            fout.create_dataset('feats', data=data, dtype='float32')
    
    if cfg.modality_X == 'camera':
        with h5py.File(path.join(cfg.video_root, "{0}/{1}.h5").format(session_id, name), 'w') as fout:
            fout.create_dataset('feats', data=data, dtype='float32')


def load_data(cfg, session_id, modality, name=None):

    if modality == 'can':
        fin = h5py.File(path.join(cfg.sensor_root, "{0}/{1}.h5").format(session_id,name), 'r')
        d = fin['feats'][:]
    
    if modality == 'camera':
        fin = h5py.File(path.join(cfg.video_root, "{0}/{1}.h5").format(session_id,name), 'r')
        d = fin['feats'][:]

    return d

def load_data_list(cfg, session_list, modality, name=None):

    allsessions = []
    vid = []

    for i, session_id in enumerate(session_list):
        d = load_data(cfg, session_id, modality, name=name)
        allsessions.append(d)
        vid.append(np.ones(d.shape[0]) * i)

    data = np.vstack(allsessions)
    vid = np.hstack(vid)

    return data, vid

def load_annotation_list(cfg, session_list):

    allsessions = []

    for i, session_id in enumerate(session_list):
        label = pkl.load(open(path.join(cfg.annotation_root, "{}/annotations.pkl".format(session_id)), 'r'))
        allsessions.append(label)

    annotation = np.concatenate(allsessions, axis=0)

    return annotation

