import numpy as np
import os
import pickle as pkl
import sys
import pdb
import tensorflow as tf

class Encoder(object):
    def __init__(self, cfg):
#            pca_path='/home/xyang/UTS/Data/results/PCA_featfc_200.pkl'):
        self.method = cfg.encoder

        if self.method == "feat" or self.method == "pred":

            # load configurations accroding to model name
            result_path = os.path.join(cfg.result_root, cfg.model_name)
            model_cfg = pkl.load(open(os.path.join(result_path, 'configs.pkl'), 'r'))
            model_cfg.batch_size = 1
            model_cfg.input_keep_prob, model_cfg.output_keep_prob = 1, 1

            # load model according to model name
            model_type = cfg.model_name.split('_')[1]
    
            if model_type == "convlstm":
                from models.conv_lstm import ConvLSTM as Model
                x = tf.placeholder(tf.float32, (1, model_cfg.max_time, model_cfg.n_h, model_cfg.n_w, model_cfg.n_input))
                y = tf.placeholder(tf.float32, (1, model_cfg.n_output))
                seq_len = tf.placeholder(tf.int32, (1,))
            elif model_type == "convuntrimmedlstm":
                from models.conv_untrimmed_lstm import ConvUntrimmedLSTM as Model
                x = tf.placeholder(tf.float32, (1, model_cfg.max_time, model_cfg.n_h, model_cfg.n_w,model_cfg.n_input))
                y = tf.placeholder(tf.float32, (1, model_cfg.max_time, model_cfg.n_output))
                seq_len = tf.placeholder(tf.int32, (1,))
            elif model_type == "convbilstm":
                from models.conv_bilstm import ConvBiLSTM as Model
            elif model_type == "convtsn":
                from models.conv_tsn import ConvTSN as Model
            else:
                raise NotImplementedError
    
            self.model = Model(model_cfg)
            self.model.print_config()
            self.model.build_network(x,y,seq_len)

            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
            saver = tf.train.Saver()
            if cfg.snapshot_num == -1:
                model_path = tf.train.latest_checkpoint(result_path)
            else:
                model_path = os.path.join(result_path, cfg.model_name+'-'+str(cfg.snapshot_num))
            saver.restore(self.sess, model_path)
            print "Model: "+model_path+" is restored"

        elif self.method == "rank":

            # load model according to model name
            model_type = cfg.model_name.split('_')[0]
    
            from models.conv_lstm_rank import ConvLSTMRank as Model

            # load configurations accroding to model name
            result_path = os.path.join(cfg.result_root, cfg.model_name)
            model_cfg = pkl.load(open(os.path.join(result_path, 'configs.pkl'), 'r'))
            model_cfg.batch_size = 1
            model_cfg.input_keep_prob, model_cfg.output_keep_prob = 1, 1
            self.model = Model(model_cfg)
            self.model.print_config()

            x1 = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.max_time, model_cfg.n_h, model_cfg.n_w,model_cfg.n_input))
            x2 = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.max_time, model_cfg.n_h, model_cfg.n_w,model_cfg.n_input))
            y = tf.placeholder(tf.int32, (model_cfg.batch_size, 1))   # adjusted shape for data loader
            seq_len1 = tf.placeholder(tf.int32, (model_cfg.batch_size,))
            seq_len2 = tf.placeholder(tf.int32, (model_cfg.batch_size,))
            self.model.build_network(x1, x2, y,seq_len1, seq_len2)

            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
            saver = tf.train.Saver()
            if cfg.snapshot_num == -1:
                model_path = tf.train.latest_checkpoint(result_path)
            else:
                model_path = os.path.join(result_path, cfg.model_name+'-'+str(cfg.snapshot_num))
            saver.restore(self.sess, model_path)
            print "Model: "+model_path+" is restored"


    def encode(self, feats):
        """
        Encode a feature sequence into a fixed length representation
        Input:
            feats -- T * input_dim
        Output:
            output -- 1 * output_dim
        """

        if self.method == "avgpool":
            output = self.avgpool_encode(feats)
        elif self.method == "maxpool":
            output = self.maxpool_encode(feats)
        elif self.method == "concat":
            output = self.concat_encode(feats)
        elif self.method == "feat":
            output = self.feat_encode(feats)
        elif self.method == "pred":
            output = self.pred_encode(feats)
        elif self.method == "rank":
            output = self.rank_encode(feats)
        else:
            raise NotImplementedError

        return output

    def avgpool_encode(self, feats):
        output = np.mean(feats, axis=0)
        output = output.reshape(1, -1)

        # L2 normalization
        norm = np.linalg.norm(output)
        if norm > 0:
            output /= norm
        return output.reshape(1, -1)

    def maxpool_encode(self, feats):
        output = np.max(feats, axis=0)
        output = output.reshape(1, -1)

        # L2 normalization
        norm = np.linalg.norm(output)
        if norm > 0:
            output /= norm
        return output.reshape(1, -1)

    def concat_encode(self, feats):
        output = feat.reshape(1, -1)
        # L2 normalization
        norm = np.linalg.norm(output)
        if norm > 0:
            output /= norm
        return output

    def feat_encode(self, feats):
        feat = np.zeros((1, self.model.max_time)+feats.shape[1:], dtype='float32')
        if feats.shape[0] >= self.model.max_time:
            feat[0] = feats[-self.model.max_time:]
            length = self.model.max_time
        else:
            feat[0,:feats.shape[0],:,:,:] = feats
            length = feats.shape[0]

        output = self.sess.run(self.model.feat, feed_dict={
                                self.model.x: feat,
                                self.model.seq_len: np.array([length], dtype='int32')})

        # average pooling if not only output last timestep
        if len(output.shape)>2 and output.shape[1] == self.model.max_time:
            output = np.mean(output, axis=1)

        # L2 normalization
        output /= np.linalg.norm(output)
        return output.reshape(1, -1)

    def pred_encode(self, feats):
        feat = np.zeros((1, self.model.max_time)+feats.shape[1:], dtype='float32')
        if feats.shape[0] >= self.model.max_time:
            feat[0] = feats[-self.model.max_time:]
            length = self.model.max_time
        else:
            feat[0,:feats.shape[0],:,:,:] = feats
            length = feats.shape[0]

        output = self.sess.run(self.model.pred, feed_dict={
                                self.model.x: feat,
                                self.model.seq_len: np.array([length], dtype='int32')})

        # average pooling if not only output last timestep
        if len(output.shape)>2 and output.shape[1] == self.model.max_time:
            output = np.mean(output, axis=1)

        # L2 normalization
        output /= np.linalg.norm(output)
        return output.reshape(1, -1)

    def rank_encode(self, feats):
        feat = np.zeros((1, self.model.max_time)+feats.shape[1:], dtype='float32')
        if feats.shape[0] >= self.model.max_time:
            feat[0] = feats[-self.model.max_time:]
            length = self.model.max_time
        else:
            feat[0,:feats.shape[0],:,:,:] = feats
            length = feats.shape[0]

        output = self.sess.run(self.model.feat1, feed_dict={
                                self.model.x1: feat,
                                self.model.seq_len1: np.array([length], dtype='int32')})

        # L2 normalization
        output /= np.linalg.norm(output)
        return output.reshape(1, -1)
