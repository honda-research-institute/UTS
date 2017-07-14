"""
Script for model training / testing
"""

import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb

import tensorflow as tf
from tqdm import tqdm
from basicLSTM import basicLSTM

from cfg import Config


def load_data(args, session_ids):

    print ("Loading data ...")

    allsessions = []
    vid = []

    if args.can:
        output_path = args.root + 'general_sensors'
        for i, session_id in enumerate(session_ids):
            d = pkl.load(open(os.path.join(output_path, "general_sensors_{0}.pkl").format(session_id), 'r'))
            allsessions.append(d)
            vid.append(np.ones(d.shape[0]) * i)

    
    if args.camera:
        output_path = args.root + 'camera'
        for i, session_id in enumerate(session_ids):
            fin = h5py.File(os.path.join(output_path, "{0}/feats.h5").format(session_id), 'r')
            d = fin['feats'][:]
            allsessions.append(d)
            vid.append(np.ones(d.shape[0]) * i)

    data = np.vstack(allsessions)
    vid = np.hstack(vid)

    return data, vid

def save_data(args, data, session_id):
    """
    save features of one session
    """

    if args.can:
        output_path = args.root + 'general_sensors'
        pkl.dump(data, open(os.path.join(output_path, "lstm_feat_{0}.pkl").format(session_id), 'w'))
    
    if args.camera:
        output_path = args.root + 'camera'
        pkl.dump(data, open(os.path.join(output_path, "{0}/lstm_feat.h5").format(session_id), 'w'))



def iterate_minibatch(x, vid, batch_size, n_steps, shuffle=False):
    """
    Iterator for creating batch data
    x.shape = [N, dim]

    return shape:
    x_batch.shape = [batch_size, n_step, dim]
    y_batch.shape = [batch_size, dim]
    """
    
    valid = vid[n_steps:] == vid[:-n_steps]
    indices = np.where(valid)[0]

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, indices.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i: i+batch_size]

        temp = []
        for j in range(n_steps):
            temp.append(np.expand_dims(x[excerpt + j, :], axis=1))    # add axis for n_step

        yield np.concatenate(temp, axis=1), x[excerpt + n_steps, :]


class ModelRunner():

    def __init__(self, cfg):

        self.cfg = cfg

        self.batch_size = self.cfg.model_params['batch_size']
        self.n_input = self.cfg.model_params['n_input']
        self.n_hidden = self.cfg.model_params['n_hidden']
        self.n_steps = self.cfg.model_params['n_steps']
        self.n_epochs = self.cfg.model_params['n_epochs']
        self.learning_rate = self.cfg.model_params['learning_rate']

        # initialize a LSTM model
        self.lstm_model = basicLSTM(self.batch_size, self.n_steps,
                                self.n_input, self.n_hidden)

    def train(self, data, vid, silent_mode=False):

        with tf.Session() as sess:

            # initialize a LSTM model
            lstm_model = self.lstm_model
            
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(lstm_model.cost)

            init = tf.global_variables_initializer()

            writer = tf.summary.FileWriter('../result/', sess.graph)
            sess.run(init)

            saver = tf.train.Saver()

            if not silent_mode:
                iters = self.n_epochs * (data.shape[0] // self.batch_size) * self.batch_size
                pbar = tqdm(total=iters, dynamic_ncols=True)


            # start training

            event_timer = 0
            for epoch in range(1, self.n_epochs + 1):

                count = 0
                loss_sum = 0
                # train for one epoch
                for x_batch, y_batch in iterate_minibatch(data, vid,
                                batch_size=self.batch_size, n_steps=self.n_steps, shuffle=True):

                    train_loss, _ = sess.run([lstm_model.cost, optimizer], feed_dict={
                                            lstm_model.x: x_batch,
                                            lstm_model.y: y_batch})
                    loss_sum += train_loss
                    count += 1

                    summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss",
                                                                simple_value=train_loss), ])
                    writer.add_summary(summary, event_timer)
                    event_timer += self.batch_size

                    if not silent_mode:
                        description = "Epoch:{0}, train_loss: {1}".format(epoch, train_loss)
                        pbar.set_description(description)
                        pbar.update(self.batch_size)

                saver.save(sess, "../result/model", global_step=epoch)

                if not silent_mode:
                    print ("Epoch:{0}, average Loss:{1}".format(epoch, loss_sum/count))

            if not silent_mode:
                pbar.close()
                print ("Training done!")


    def extract_feat(self, data, vid, snapshot_num="-1", silent_mode=False):
        """
        Extract features using pretrained model
        """
    
        with tf.Session() as sess:
    
            # initialize a LSTM model
            lstm_model = self.lstm_model
                
            saver = tf.train.Saver()
            if snapshot_num == "-1":
                saver.restore(sess, tf.train.latest_checkpoint("../result"))
            else:
                pass
    
            if not silent_mode:
                pbar = tqdm(total=data.shape[0], dynamic_ncols=True)
    
    
            feat = []
            for x_batch, y_batch in iterate_minibatch(data, vid,
                            batch_size=self.batch_size, n_steps=self.n_steps, shuffle=False):
    
                feat.append(sess.run(lstm_model.hidden, feed_dict={
                                    lstm_model.x: x_batch,
                                    lstm_model.y: y_batch}))
    
                if not silent_mode:
                    pbar.update(self.batch_size)
    
            if not silent_mode:
                pbar.close()
                print ("Feature extraction done!")
    
            return np.vstack(feat)


def main(args):

    # load default configuration
    cfg = Config()

    args.root = cfg.root

    # load session ids
    session_list = open('/home/xyang/project/data/session_list', 'r')
    session_ids = session_list.read().strip().split('\n')
    session_ids = [''.join(s.split('-')[:-1]) for s in session_ids]


    if args.train_stage:
        # load data into memory
        data, vid = load_data(args, session_ids)

        # model training
        model_runner = ModelRunner(cfg)
        model_runner.train(data, vid, silent_mode=args.silent)

    if args.test_stage:

        model_runner = ModelRunner(cfg)

        # extract features for each session
        for session_id in session_ids:
            data, vid = load_data(args, [session_id])

            feat = model_runner.extract_feat(data, vid, 
                    snapshot_num=args.snapshot_num, silent_mode=args.silent)

            save_data(args, feat, session_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for model training/testing')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', dest='train_stage', action='store_true',
                       help='Training')
    group.add_argument('--test', dest='test_stage', action='store_true',
                       help='Testing')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--can', dest='can', action='store_true',
                       help='CANbus')
    group2.add_argument('--camera', dest='camera', action='store_true',
                       help='camera')

    parser.add_argument('--silent', dest='silent', action='store_true')
    parser.add_argument('--snapshot_num', dest='snapshot_num', type=str, default="-1")

    args = parser.parse_args()

    main(args)
