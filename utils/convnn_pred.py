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
from configs.retrieval_config import RetrievalConfig

def main():
    cfg = RetrievalConfig().parse()
    model_type = cfg.model_name.split('_')[0]
    result_path = os.path.join(cfg.result_root, cfg.model_name)

    if model_type == "convnn":
        from models.conv_nn import ConvNN as Model
    else:
        raise NotImplementedError

    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model_cfg = pkl.load(open(os.path.join(result_path, 'configs.pkl'), 'r'))
    model_cfg.batch_size = 1
    model_cfg.input_keep_prob, model_cfg.output_keep_prob = 1, 1
    model_cfg.isTrain, model_cfg.continue_train = False, False
    model_cfg.val_session = cfg.val_session
    model = Model(model_cfg)
    model.print_config()

    x = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.n_h, model_cfg.n_w, model_cfg.n_input))
    y = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.n_output))
    model.build_network(x,y)
        
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        if cfg.snapshot_num == -1:
            model_path = tf.train.latest_checkpoint(result_path)
        else:
            model_path = os.path.join(result_path, cfg.model_name+'-'+str(cfg.snapshot_num))
        saver.restore(sess, model_path)
        print "Model: "+model_path+" is restored"

        for session_id in cfg.train_session:#+cfg.val_session+cfg.test_session:
            print session_id

            with h5py.File(os.path.join(cfg.video_root, session_id+'/'+model_cfg.X_feat+'.h5'), 'r') as fin:
                X_feats = fin['feats'][:]

            feat = []
            pred = []
            for i in range(X_feats.shape[0]):
                f, p = sess.run([model.feat, model.pred], feed_dict={
                                            x: X_feats[i].reshape(model_cfg.batch_size, model_cfg.n_h, model_cfg.n_w, model_cfg.n_input)})

#                temp = np.zeros((1, model_cfg.n_output), dtype='float32')
#                temp[0, p] = 1
#                pred.append(temp)
                feat.append(f)
                pred.append(p)

            feat = np.concatenate(feat, axis=0)
            pred = np.concatenate(pred, axis=0)

            with h5py.File(os.path.join(cfg.video_root, session_id+'/feat_'+model_cfg.name+'-'+str(cfg.snapshot_num)+'.h5'), 'w') as fout:
                fout.create_dataset('feats', data=feat, dtype='float32')
            with h5py.File(os.path.join(cfg.video_root, session_id+'/pred_'+model_cfg.name+'-'+str(cfg.snapshot_num)+'.h5'), 'w') as fout:
                fout.create_dataset('feats', data=pred, dtype='float32')

if __name__ == '__main__':
    main()
