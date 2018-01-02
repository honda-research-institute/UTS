"""
ConvNN to predict sensor value
"""

import pickle as pkl
import h5py
import os
import numpy as np
import argparse
import time
import pdb
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../')
from configs.train_config import TrainConfig
from utils.data_io import FrameGenerator, FrameGeneratorTrimmed
from utils.utils import compute_framelevel_ap

def main():

    cfg = TrainConfig().parse()
    print cfg.name
    result_path = os.path.join(cfg.result_root, cfg.name)

    if cfg.model_type == "convnn":
        from models.conv_nn import ConvNN as Model
    else:
        raise NotImplementedError

    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


    # load data
    data_root = '/home/xyang/Downloads/cifar-10-batches-py/'
    with h5py.File(data_root+'cifar10_conv_train.h5', 'r') as fin:
        data = fin['feats'][:-1]
        label = fin['label'][:]
    with h5py.File(data_root+'cifar10_conv_test.h5', 'r') as fin:
        test_data = fin['feats'][:]
        test_label = fin['label'][:]
    print data.shape
    print test_data.shape

    if cfg.isTrain or cfg.continue_train:
        # Place data loading and preprocessing on the cpu
    
        # Train the model
        x = tf.placeholder(tf.float32, (cfg.batch_size, cfg.n_h, cfg.n_w,cfg.n_input))
        y = tf.placeholder(tf.float32, (cfg.batch_size, cfg.n_output))
    
    
        # Train the model
        model = Model(cfg)
        model.print_config()
        model.build_network(x, y)
    
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
    
            init = tf.global_variables_initializer()
            writer = tf.summary.FileWriter(result_path, sess.graph)
            # save configures
            pkl.dump(cfg, open(os.path.join(result_path, 'configs.pkl'), 'w'))
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=cfg.n_epochs)

            # print # of model parameters
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= int(dim)
                total_parameters += variable_parameters
            print ("Total number of parameters: %d" % total_parameters)

            epoch_start = 1
            if cfg.continue_train:
                if cfg.snapshot_num == -1:
                    model_path = tf.train.latest_checkpoint(result_path)
                else:
                    model_path = os.path.join(result_path, cfg.name+'-'+str(cfg.snapshot_num))
                saver.restore(sess, model_path)
                print "Model: "+model_path+" is restored"

                # set epoch start index
                epoch_start = int(model_path.split('-')[-1]) + 1
    
            if not cfg.silent_mode:
                iters = (cfg.n_epochs-epoch_start+1) * (data.shape[0] // cfg.batch_size) * cfg.batch_size
                pbar = tqdm(total=iters, dynamic_ncols = True)
    
            past_val_loss = []    # for early stopping
            event_timer = (epoch_start-1) * cfg.batch_size
            idx = np.arange(data.shape[0] // cfg.batch_size * cfg.batch_size)
            for epoch in range(epoch_start, cfg.n_epochs+1):
                np.random.shuffle(idx)

                pred = []
                y_truth = []
                for i in range(0, data.shape[0], cfg.batch_size):
                    x_batch = data[idx[i:i+cfg.batch_size]]
                    y_batch = label[idx[i:i+cfg.batch_size]]

                    train_loss, train_cost, train_regularizor, temp_pred, _ = sess.run([model.loss, model.cost, model.regularizor, model.pred, model.optimizer], feed_dict={
                                                                x: x_batch,
                                                                y: y_batch})

                    y_truth.append(y_batch)
                    pred.append(temp_pred)

                    if cfg.is_classify:
                        acc = accuracy_score(y_batch.argmax(axis=1), temp_pred.argmax(axis=1))
                    else:
                        acc = 0

                    summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss",
                                                                simple_value=train_loss),
                                                    tf.Summary.Value(tag="train_cost",
                                                                simple_value=train_cost),
                                                    tf.Summary.Value(tag="train_regularizor",
                                                                simple_value=train_regularizor),
                                                    tf.Summary.Value(tag="train_acc",
                                                                simple_value=acc)])
                    writer.add_summary(summary, event_timer)
                    event_timer += cfg.batch_size


                    if not cfg.silent_mode:
                        description = "Epoch:{0}, train_loss: {1}".format(epoch, train_loss)
                        pbar.set_description(description)
                        pbar.update(cfg.batch_size)

                print "Epoch %d done!" % epoch

                pred = np.concatenate(pred, axis=0)
                y_truth = np.concatenate(y_truth, axis=0)
                if cfg.is_classify:
                    mAP, _ = compute_framelevel_ap(y_truth, pred, background=True if cfg.modality_Y=='label' else False)
                else:
                    mAP = 0
                print "\n"
                print "Training mAP: ", mAP
                summary = tf.Summary(value=[tf.Summary.Value(tag="train_mAP",
                                                            simple_value=mAP),])
                writer.add_summary(summary, event_timer)



                val_loss = []
                pred = []
                y_truth = []
                for i in range(0, test_data.shape[0], cfg.batch_size):
                    x_batch = test_data[i:i+cfg.batch_size]
                    y_batch = test_label[i:i+cfg.batch_size]
                    temp_loss, temp_pred = sess.run([model.cost, model.pred], feed_dict={
                                                    x: x_batch,
                                                    y: y_batch})
                    val_loss.append(temp_loss)
                    pred.append(temp_pred)
                    y_truth.append(y_batch)

                val_loss = np.asarray(val_loss)
                pred = np.concatenate(pred, axis=0)
                y_truth = np.concatenate(y_truth, axis=0)

                val_loss = np.mean(val_loss)
                if cfg.is_classify:
                    acc = accuracy_score(y_truth.argmax(axis=1), pred.argmax(axis=1))
                    mAP, ap = compute_framelevel_ap(y_truth, pred, background=True if cfg.modality_Y=='label' else False)
                else:
                    acc = 0
                    mAP = 0
                print "\n"
                print "Validation Loss: ", val_loss
                print "Validation Accuracy: ", acc
                print "Validation mAP: ", mAP
                for i,a in ap:
                    print ("Label %d: %f" % (i, a))

                summary = tf.Summary(value=[tf.Summary.Value(tag="validation_loss",
                                                            simple_value=val_loss),
                                            tf.Summary.Value(tag="validation_acc",
                                                            simple_value=acc),
                                            tf.Summary.Value(tag="validation_mAP",
                                                            simple_value=mAP)])
                writer.add_summary(summary, event_timer)

                saver.save(sess, os.path.join(result_path, cfg.name), global_step=epoch)

            if not cfg.silent_mode:
                pbar.close()
                print ("Training model done!")
            

if __name__ == "__main__":
    main()
