import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb
import tensorflow as tf
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../')
from configs.convlstm_config import ConvLSTMConfig
from models.conv_lstm import ConvLSTM
from models.conv_seq2seq import ConvSeq2seq

def main():
    cfg = ConvLSTMConfig().parse()
    cfg.batch_size = 1
    print cfg.name
    result_path = os.path.join(cfg.result_root, cfg.name)

    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        
    x = tf.placeholder(tf.float32, (1, cfg.max_time, cfg.n_h, cfg.n_w,cfg.n_input))
    y = tf.placeholder(tf.float32, (1, cfg.max_time, cfg.n_output))
    seq_len = tf.placeholder(tf.int32, (1,))
#    p0 = tf.placeholder(tf.float32, (1, cfg.n_output))

#    model = ConvLSTM(cfg)
    model = ConvSeq2seq(cfg)
    model.print_config()
    model.build_network(x, y, seq_len)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        print result_path
        latest_checkpoint = tf.train.latest_checkpoint(result_path)
        print "Loading pretrained model %s" % latest_checkpoint
        saver.restore(sess, latest_checkpoint)

        for session_id in ['201703061323']:
#        for session_id in cfg.val_session+cfg.test_session:
            print session_id

            X_feat = 'feat_conv'
            with h5py.File(os.path.join(cfg.video_root, session_id+'/'+X_feat+'.h5'), 'r') as fin:
                X_feats = fin['feats'][:]
            Y_feat = 'feat_norm'
            with h5py.File(os.path.join(cfg.sensor_root, session_id+'/'+Y_feat+'.h5'), 'r') as fin:
                Y_feats = fin['feats'][:]

            pred = []
            for i in range(5, X_feats.shape[0]+1, 10):
                start = max(1, i-cfg.max_time)
                length = i - start
                feat = np.zeros((1, cfg.max_time, cfg.n_h,cfg.n_w,cfg.n_input), dtype='float32')
                feat[0,:length,:,:,:] = X_feats[start:i]
                l = np.zeros((1,), dtype='int32')
                l[0] = length
#                p0_temp = np.zeros((1,cfg.n_output), dtype='float32')
#                p0_temp[0] = Y_feats[start-1]
    
                p_temp = sess.run(model.pred, feed_dict={
                                            x: feat,
                                            seq_len: l})
#                                            p0: p0_temp})
                p = np.zeros((1, X_feats.shape[0], cfg.n_output), dtype='float32')
                p[0,start:i,:] = p_temp
                pred.append(p)

            pred = np.concatenate(pred, axis=0)
            pred = np.sum(pred, 0) / 3

            with h5py.File(os.path.join(cfg.sensor_root, session_id+'/pred_'+cfg.name+'.h5'), 'w') as fout:
                fout.create_dataset('feats', data=pred, dtype='float32')

if __name__ == '__main__':
    main()
