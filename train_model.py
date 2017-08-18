import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb
from tqdm import tqdm

import tensorflow as tf
from configs.train_config import TrainConfig
from models.models import create_model
from utils.data_io import CreateDataset, load_data, load_data_list, save_feat
from utils.utils import iterate_minibatch


def main():

    # load configuration
    cfg = TrainConfig().parse()
    print cfg.name

    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = create_model(cfg)
    model.print_config()

    result_path = os.path.join(cfg.result_root, cfg.name)

    if cfg.isTrain:
        dataset = CreateDataset(cfg)
        dataset.print_config()

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
    
            init = tf.global_variables_initializer()
            writer = tf.summary.FileWriter(result_path, sess.graph)
            sess.run(init)
            saver = tf.train.Saver()

            if not cfg.silent_mode:
                iters = cfg.n_epochs * (dataset.N // cfg.batch_size) * cfg.batch_size
                pbar = tqdm(total=iters, dynamic_ncols=True)
    
            event_timer = 0
            for epoch in range(1, cfg.n_epochs+1):
    
                iterator = dataset.get_iterator()    # reinitialize iterator
                for batch in iterator:
    
                    train_loss, _ = sess.run([model.loss, model.optimizer], feed_dict={
                                            model.x: batch['x_batch'],
                                            model.y: batch['y_batch'],
                                            model.in_len: batch['in_len'],
                                            model.out_len: batch['out_len']})
    
                    summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss",
                                                                simple_value=train_loss),])
                    writer.add_summary(summary, event_timer)
                    event_timer += cfg.batch_size
    
                    if not cfg.silent_mode:
                        description = "Epoch:{0}, train_loss: {1}".format(epoch, train_loss)
                        pbar.set_description(description)
                        pbar.update(cfg.batch_size)
                
                saver.save(sess, os.path.join(result_path, cfg.name), global_step=epoch)
    
            if not cfg.silent_mode:
                pbar.close()
                print ("Training model done!")
            
    else:
        # extract features using pretrained model
        with tf.Session() as sess:
    
            saver = tf.train.Saver()
            if cfg.snapshot_num == -1:
                saver.restore(sess, tf.train.latest_checkpoint(result_path))
            else:
                raise NotImplementedError
    
            for session_id in cfg.train_session:   # extract feature for all sessions
                print "Session: ", session_id
                x = load_data(cfg, session_id, cfg.modality_X, cfg.X_feat)
    
                feat = []
                for batch in iterate_minibatch(x, cfg.batch_size, cfg.max_time, shuffle=False):
                    feat_batch = sess.run(model.feat, feed_dict={
                                    model.x: batch['x_batch'],
                                    model.in_len: batch['seq_batch']})
                    feat.append(feat_batch[batch['seq_batch'] > 0, :])
    
                new_feat = np.vstack(feat)
    
                save_feat(new_feat, cfg, session_id)
    
            if not cfg.silent_mode:
                print ("Feature extraction done!")


if __name__ == "__main__":

    main()
