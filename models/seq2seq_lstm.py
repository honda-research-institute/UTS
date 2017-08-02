import tensorflow as tf
from utils.utils import iterate_minibatch1, iterate_minibatch3
from tqdm import tqdm
import numpy as np

'''
Sequence-to-sequence LSTM for future prediction and (or) history reconstruction
'''

class Seq2seqLSTM():
    def __init__(self, batch_size, n_steps, n_input, n_hidden, n_output=None,
            reverse=True, reconstruct=True, n_predict=0):

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reverse = reverse
        self.reconstruct = reconstruct
        self.n_output1 = None    # for reconstruction
        self.n_output2 = None    # for prediction
        self.n_predict = n_predict

        if reconstruct:
            self.n_output1 = self.n_input
        if n_output is not None:
            self.n_output2 = n_output

        # define variables and weights
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.n_input])

        if self.n_output1 is not None:
            if reverse:
                self.y1 = tf.reverse(self.x, axis=[1])
            else:
                self.y1 = self.x
            W_ho1 = tf.get_variable(name="W_ho1", shape=[self.n_hidden, self.n_output1],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=True)
            b_o1 = tf.get_variable(name="b_o1", shape=[self.n_output1],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
        if self.n_output2 is not None:
            self.y2 = tf.placeholder(tf.float32, [self.batch_size, self.n_predict, self.n_output2])
            W_ho2 = tf.get_variable(name="W_ho2", shape=[self.n_hidden, self.n_output2],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=True)
            b_o2 = tf.get_variable(name="b_o2", shape=[self.n_output2],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

        # Encoder
        encoder_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, self.x, dtype=tf.float32)

        self.feat = encoder_outputs[-1]

        # Decoder

        # Reconstruction
        if self.n_output1 is not None:
            state1 = encoder_final_state
            decoder_cell1 = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)
            pred1 = []

            for i in range(self.n_steps):
                if i == 0:
                    curr_input1 = tf.zeros([self.batch_size, self.n_output1])

                output1, state1 = decoder_cell1(curr_input1, state1)

                # current output as next input
                pred = tf.matmul(output1, W_ho1) + b_o1
                pred1.append(pred)
                curr_input1 = tf.identity(pred)
                #tf.stop_gradient(curr_input1)

#                if i == 0:
#                    curr_input1 = tf.get_variable(name="curr_input1", shape=[self.batch_size, self.n_output1], initializer=tf.zeros_initializer(), trainable=False)
#
#                output1, state1 = decoder_cell1(curr_input1, state1)
#
#                # current output as next input
#                pred = tf.matmul(output1, W_ho1) + b_o1
#                pred1.append(pred)
#                tf.assign(curr_input1,pred) 

            pred1 = tf.stack(pred1, axis=1)

        # Future prediction
        if self.n_output2 is not None:
            state2 = encoder_final_state
            decoder_cell2 = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)
            pred2 = []

            for i in range(self.n_predict):
                if i == 0:
                    curr_input2 = tf.zeros([self.batch_size, self.n_output2])

                output2, state2 = decoder_cell2(curr_input2, state2)

                # current output as next input
                pred = tf.matmul(output2, W_ho2), + b_o2
                pred2.append(pred)
                curr_input2 = tf.identity(pred)

            pred2 = tf.stack(pred2, axis=1)

        # Define loss
        self.loss = 0.0

        if self.n_output1 is not None:
            self.loss += tf.losses.mean_squared_error(
                    tf.reshape(self.y1, (self.batch_size*self.n_steps, self.n_output1)),
                    tf.reshape(pred1, (self.batch_size*self.n_steps, self.n_output1)))

        if self.n_output2 is not None:
            self.loss += tf.losses.mean_squared_error(
                    tf.reshape(self.y2, (self.batch_size*self.n_predict, self.n_output2)),
                    tf.reshape(pred2, (self.batch_size*self.n_predict, self.n_output2)))

    def init_train(self, **kwargs):

        self.n_epochs = kwargs.get('n_epochs', 10)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.optimizer = kwargs.get('optimizer', 'rmsprop')
        self.result_path = kwargs.get('result_path', '')

        if self.predict


    def train(self, X, vid, Y=None, silent_mode=False):
        """
        X - modality 1, can be trained with reconstruction and prediction
        Y - modality 2, can be used as prediction signal
        vid - video id for each sample
        """

        with tf.Session() as sess:

            if self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                raise NotImplementedError

            init = tf.global_variables_initializer()

            writer = tf.summary.FileWriter(self.result_path, sess.graph)
            sess.run(init)

            saver = tf.train.Saver()

            if not silent_mode:
                iters = self.n_epochs * (X.shape[0] // self.batch_size) * self.batch_size
                pbar = tqdm(total=iters, dynamic_ncols=True)


            # start training

            event_timer = 0
            for epoch in range(1, self.n_epochs + 1):

                count = 0
                loss_sum = 0
                # train for one epoch
                for x_batch, y_batch in iterate_minibatch3(X, vid, Y,
                                batch_size=self.batch_size, n_steps=self.n_steps,
                                n_predict=self.n_predict, shuffle=True):

                    train_loss, _ = sess.run([self.loss, optimizer], feed_dict={
                                            self.x: x_batch,
                                            self.y2: y_batch})
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

                saver.save(sess, self.result_path, global_step=epoch)

                if not silent_mode:
                    print ("Epoch:{0}, average Loss:{1}".format(epoch, loss_sum/count))

            if not silent_mode:
                pbar.close()
                print ("Training done!")


    def extract_feat(self, data, snapshot_num="-1", silent_mode=False):
        """
        Extract features using pretrained model
        """
    
        with tf.Session() as sess:
    
            saver = tf.train.Saver()
            if snapshot_num == "-1":
                saver.restore(sess, tf.train.latest_checkpoint(self.result_path))
            else:
                pass
    
            if not silent_mode:
                pbar = tqdm(total=data.shape[0], dynamic_ncols=True)
    
    
            feat = []
            for x_batch, _ in iterate_minibatch1(data, batch_size=self.batch_size, n_steps=self.n_steps, shuffle=False):
    
                feat.append(sess.run(self.feat, feed_dict={
                                    self.x: x_batch}))
    
                if not silent_mode:
                    pbar.update(self.batch_size)
    
            if not silent_mode:
                pbar.close()
                print ("Feature extraction done!")
    
            return np.vstack(feat)

