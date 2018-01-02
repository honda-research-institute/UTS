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
from utils.data_io import FrameGeneratorRankTrimmed

def main():

    cfg = TrainConfig().parse()
    print cfg.name
    result_path = os.path.join(cfg.result_root, cfg.name)

    if cfg.model_type == "convnn_rank":
        from models.conv_nn_rank import ConvNNRank as Model
    else:
        raise NotImplementedError

    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if cfg.isTrain or cfg.continue_train:
        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            train_data = FrameGeneratorRankTrimmed(cfg)
            train_data.print_config()
            val_data = FrameGeneratorRankTrimmed(cfg, isTrain=False)
    
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
        x1 = tf.placeholder(tf.float32, (cfg.batch_size, cfg.n_h, cfg.n_w,cfg.n_input))
        x2 = tf.placeholder(tf.float32, (cfg.batch_size, cfg.n_h, cfg.n_w,cfg.n_input))
        y = tf.placeholder(tf.float32, (cfg.batch_size,))
    
    
        # Train the model
        model = Model(cfg)
        model.print_config()
        model.build_network(x1,x2, y)
    
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
                if epoch > epoch_start:
                    train_data.create_dataset()
                sess.run(training_init_op)

                while True:
                    try:
                        x1_batch, x2_batch, y_batch = sess.run(next_train)
                        train_loss, train_cost, train_regularizor, d, _ = sess.run([model.loss, model.cost, model.regularizor, model.d, model.optimizer], feed_dict={
                                                                x1: x1_batch,
                                                                x2: x2_batch,
                                                                y: y_batch})

                        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss",
                                                                simple_value=train_loss),
                                                    tf.Summary.Value(tag="train_cost",
                                                                simple_value=train_cost),
                                                    tf.Summary.Value(tag="train_regularizor",
                                                                simple_value=train_regularizor)])
                        writer.add_summary(summary, event_timer)
                        event_timer += cfg.batch_size


                        if not cfg.silent_mode:
                            description = "Epoch:{0}, train_loss: {1}".format(epoch, train_loss)
                            pbar.set_description(description)
                            pbar.update(cfg.batch_size)

                    except tf.errors.OutOfRangeError:
                        print "Epoch %d done!" % epoch
                        break

                # Validation
                sess.run(validation_init_op)

                val_loss = []
                while True:
                    try:
                        x1_batch, x2_batch, y_batch = sess.run(next_val)
                        temp_loss = sess.run(model.cost, feed_dict={
                                                    x1: x1_batch,
                                                    x2: x2_batch,
                                                    y: y_batch})
                        val_loss.append(temp_loss)
                    except tf.errors.OutOfRangeError:
                        break

                val_loss = np.asarray(val_loss)
                val_loss = np.mean(val_loss)
                print "\n"
                print "Validation Loss: ", val_loss

                summary = tf.Summary(value=[tf.Summary.Value(tag="validation_loss",
                                                            simple_value=val_loss),])
                writer.add_summary(summary, event_timer)

                saver.save(sess, os.path.join(result_path, cfg.name), global_step=epoch)

                # Early stopping
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

        x1 = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.n_h, model_cfg.n_w, model_cfg.n_input))
        x2 = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.n_h, model_cfg.n_w, model_cfg.n_input))
        y = tf.placeholder(tf.float32, (model_cfg.batch_size, model_cfg.n_output))
        model.build_network(x1,x2,y)

        with tf.device('/cpu:0'):
            val_data = FrameGeneratorRankTrimmed(model_cfg, isTrain=False)
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
            while True:
                try:
                    x1_batch, x2_batch, y_batch = sess.run(next_val)
                    temp_val = sess.run(model.loss, feed_dict={
                                                x1: x1_batch,
                                                x2: x2_batch,
                                                y: y_batch})
                    val_loss.append(temp_val)
                except tf.errors.OutOfRangeError:
                    break

            val_loss = np.asarray(val_loss)
            print "Loss: ", np.mean(val_loss)
            

if __name__ == "__main__":
    main()
