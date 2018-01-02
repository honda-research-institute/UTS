"""
ConvLSTM encoder to predict sensor value
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
from utils.data_io import SeqGenerator, SeqGeneratorTrimmed
from utils.utils import compute_framelevel_ap

def main():

    cfg = TrainConfig().parse()
    print cfg.name
    result_path = os.path.join(cfg.result_root, cfg.name)

    if cfg.model_type == "convuntrimmedlstm":
        from models.conv_untrimmed_lstm import ConvUntrimmedLSTM as Model
    else:
        raise NotImplementedError

    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if cfg.isTrain or cfg.continue_train:
        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            if cfg.trimmed:
                train_data = SeqGeneratorTrimmed(cfg)
            else:
                train_data = SeqGenerator(cfg)
            train_data.print_config()
            val_data = SeqGenerator(cfg, isTrain=False)
            val_data.print_config()
    
            # create an reinitializable iterator given the dataset structure
            train_iterator = tf.data.Iterator.from_structure(train_data.data.output_types,
                                            train_data.data.output_shapes)
            next_train = train_iterator.get_next()
            val_iterator = tf.data.Iterator.from_structure(val_data.data.output_types,
                                                   val_data.data.output_shapes)
            next_val = val_iterator.get_next()
        
            training_init_op = train_iterator.make_initializer(train_data.data)
            validation_init_op = val_iterator.make_initializer(val_data.data)
    
        # Train the model
        x = tf.placeholder(tf.float32, (cfg.batch_size, cfg.max_time, cfg.n_h, cfg.n_w,cfg.n_input))
        y = tf.placeholder(tf.float32, (cfg.batch_size, cfg.max_time, cfg.n_output))
        seq_len = tf.placeholder(tf.int32, (cfg.batch_size,))
    
    
        # Train the model
        model = Model(cfg)
        model.print_config()
        model.build_network(x, y, seq_len)
    
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
                iters = (cfg.n_epochs-epoch_start+1) * (train_data.data_size // cfg.batch_size) * cfg.batch_size
                pbar = tqdm(total=iters, dynamic_ncols = True)
    
            past_val_loss = []    # for early stopping
            event_timer = (epoch_start-1) * cfg.batch_size
            for epoch in range(epoch_start, cfg.n_epochs+1):
                # initialize the data loader at each epoch
                train_data.create_dataset()
                sess.run(training_init_op)

                pred = []
                y_truth = []
                while True:
                    try:
                        x_batch, y_batch, len_batch = sess.run(next_train)
                        train_loss, train_cost, train_regularizor, temp_pred, _ = sess.run([model.loss, model.cost, model.regularizor, model.pred, model.optimizer], feed_dict={
                                                                x: x_batch,
                                                                y: y_batch,
                                                                seq_len: np.squeeze(len_batch)})
                        pred.append(temp_pred.reshape(-1,cfg.n_output))
                        y_truth.append(y_batch.reshape(-1,cfg.n_output))

                        if cfg.is_classify:
                            acc = accuracy_score(y_batch.reshape(-1, cfg.n_output).argmax(axis=1),
                                                 temp_pred.reshape(-1,cfg.n_output).argmax(axis=1))
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

                    except tf.errors.OutOfRangeError:
                        print "Epoch %d done!" % epoch
                        break

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

                # Validation
                sess.run(validation_init_op)

                val_loss = []
                pred = []
                y_truth = []
                while True:
                    try:
                        x_batch, y_batch, len_batch = sess.run(next_val)
                        temp_loss, temp_pred = sess.run([model.cost, model.pred], feed_dict={
                                                    x: x_batch,
                                                    y: y_batch,
                                                    seq_len: np.squeeze(len_batch)})
                        val_loss.append(temp_loss)
                        pred.append(temp_pred.reshape(-1,cfg.n_output))
                        y_truth.append(y_batch.reshape(-1,cfg.n_output))
                    except tf.errors.OutOfRangeError:
                        break

                val_loss = np.asarray(val_loss)
                pred = np.concatenate(pred, axis=0)
                y_truth = np.concatenate(y_truth, axis=0)

                val_loss = np.mean(val_loss)
                if cfg.is_classify:
                    acc = accuracy_score(y_truth.argmax(axis=1), pred.argmax(axis=1))
                    mAP, _ = compute_framelevel_ap(y_truth, pred, background=True if cfg.modality_Y=='label' else False)
                else:
                    acc = 0
                    mAP = 0
                print "Validation Loss: ", val_loss
                print "Validation Accuracy: ", acc
                print "Validation mAP: ", mAP

                summary = tf.Summary(value=[tf.Summary.Value(tag="validation_loss",
                                                            simple_value=val_loss),
                                            tf.Summary.Value(tag="validation_acc",
                                                            simple_value=acc),
                                            tf.Summary.Value(tag="validation_mAP",
                                                            simple_value=mAP)])
                writer.add_summary(summary, event_timer)

                saver.save(sess, os.path.join(result_path, cfg.name), global_step=epoch)
        
#                # Early stopping
#                past_val_loss.append(val_loss)
#                if len(past_val_loss) > 4:
#                    past_val_loss = past_val_loss[-4:]
#                    if past_val_loss[1]>past_val_loss[0] and past_val_loss[2]>past_val_loss[1] and past_val_loss[3]>past_val_loss[2]:
#                        print ("Early Stopping!")
#                        break

                
            if not cfg.silent_mode:
                pbar.close()
                print ("Training model done!")

    else:
        model_cfg = pkl.load(open(os.path.join(result_path, 'configs.pkl'), 'r'))
        model_cfg.input_keep_prob, model_cfg.output_keep_prob = 1, 1
        model_cfg.isTrain, model_cfg.continue_train = False, False
        model_cfg.val_session = cfg.val_session
        model = Model(model_cfg)
        model.print_config()

        x = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.max_time, model_cfg.n_h, model_cfg.n_w, model_cfg.n_input))
        y = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.max_time, model_cfg.n_output))
        seq_len = tf.placeholder(tf.int32, (model_cfg.batch_size,))
        model.build_network(x,y,seq_len)

        with tf.device('/cpu:0'):
            if cfg.trimmed:
                val_data = SeqGeneratorTrimmed(model_cfg, isTrain=False)
            else:
                val_data = SeqGenerator(model_cfg, isTrain=False)
            val_data.print_config()
    
            # create an reinitializable iterator given the dataset structure
            val_iterator = tf.data.Iterator.from_structure(val_data.data.output_types,
                                                   val_data.data.output_shapes)
            next_val = val_iterator.get_next()
        
            validation_init_op = val_iterator.make_initializer(val_data.data)

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            if cfg.snapshot_num == -1:
                model_path = tf.train.latest_checkpoint(result_path)
            else:
                model_path = os.path.join(result_path, cfg.name+'-'+str(cfg.snapshot_num))
            saver.restore(sess, model_path)
            print "Model: "+model_path+" is restored"

            sess.run(validation_init_op)

            val_loss = []
            pred = []
            y_truth = []
            while True:
                try:
                    x_batch, y_batch, len_batch = sess.run(next_val)
                    temp_val, temp_y, temp_pred = sess.run([model.loss, model.y, model.pred], feed_dict={
                                                x: x_batch,
                                                y: y_batch,
                                                seq_len: np.squeeze(len_batch)})
                    y_truth.append(y_batch.reshape(-1,model_cfg.n_output))
                    val_loss.append(temp_val)
                    pred.append(temp_pred.reshape(-1,model_cfg.n_output))
                except tf.errors.OutOfRangeError:
                    break

            val_loss = np.asarray(val_loss)
            y_truth = np.concatenate(y_truth, axis=0)
            pred = np.concatenate(pred, axis=0)
            if model_cfg.is_classify:
                acc = accuracy_score(y_truth.argmax(axis=1), pred.argmax(axis=1))
                mAP, ap = compute_framelevel_ap(y_truth, pred, background=True if model_cfg.modality_Y=='label' else False)
            else:
                acc = 0
                mAP = 0
            print "Loss: ", np.mean(val_loss)
            print "Accuracy: ", acc
            print "mAP: ", mAP
            for i,a in ap:
                print ("Label %d: %f" % (i, a))

if __name__ == "__main__":
    main()
