import tensorflow as tf
import numpy as np
from utils.utils import focal_loss

'''
Convolutional embedding + LSTM
'''

class ConvUntrimmedLSTM(object):
    def name(self):
        return "ConvUntrimmedLSTM"

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.max_time = kwargs.get('max_time', cfg.max_time)
        self.n_input = kwargs.get('n_input', cfg.n_input)
        self.n_w = kwargs.get('n_w', cfg.n_w)
        self.n_h = kwargs.get('n_h', cfg.n_h)
        self.n_C = kwargs.get('n_C', cfg.n_C)
        self.n_output = kwargs.get('n_output', cfg.n_output)
        self.n_hidden = kwargs.get('n_hidden', cfg.n_hidden)
        self.learning_rate = kwargs.get('learning_rate', cfg.learning_rate)
        self.optimizer_name = kwargs.get('optimizer', cfg.optimizer)
        self.is_classify = kwargs.get('is_classify', cfg.is_classify)
        self.focal_loss = kwargs.get('focal_loss', cfg.focal_loss)
        self.input_keep_prob = kwargs.get('input_keep_prob', cfg.input_keep_prob)
        self.output_keep_prob = kwargs.get('output_keep_prob', cfg.output_keep_prob)
        self.beta_l2 = kwargs.get('beta_l2', cfg.beta_l2)

        if self.n_output == 0:
            self.n_output = self.n_input

        # define variables
        self.W_emb = tf.get_variable(name="W_emb", shape=[1, 1, self.n_input, self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        self.W_ho = tf.get_variable(name="W_ho", shape=[self.n_hidden, self.n_output],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_o", shape=[self.n_output],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def build_network(self, x, y, seq_len):
        """
        Argument:
            x -- input features, [batch_size, max_time, n_h, n_w, n_input]
            y -- output label / features, [batch_size, max_time, n_output]
            seq_len -- length indicator, [batch_size, ]
        """

        self.x = x
        self.y = y
        self.seq_len = seq_len    # used for testing trimmed sequence

        def RNN(x):
            encoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

            dropout_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,
                                                     input_keep_prob=self.input_keep_prob,
                                                     output_keep_prob=self.output_keep_prob)

            encoder_outputs, _ = tf.nn.dynamic_rnn(
                    dropout_cell, x, self.seq_len, dtype=tf.float32)

            return encoder_outputs

        x_flat = tf.reshape(self.x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [self.batch_size, self.max_time, self.n_h*self.n_w*self.n_C])

        outputs = RNN(x_emb)
        # slice the valid output
        indices = tf.stack([tf.range(self.batch_size), self.seq_len-1], axis=1)
        self.feat = tf.gather_nd(outputs, indices)

        all_logits = tf.nn.xw_plus_b(tf.reshape(outputs, [-1, self.n_hidden]), self.W_ho, self.b_o)
        self.logits = tf.reshape(all_logits, [tf.shape(self.x)[0], tf.shape(self.x)[1], -1])

        # define loss
        if self.is_classify:
            if self.focal_loss:
                # use focal loss
                self.cost = focal_loss(labels=tf.reshape(self.y, [-1, self.n_output]), 
                                   logits=tf.reshape(self.logits,[-1, self.n_output]))
            else:
                if not self.y.dtype == tf.int32:
                    if len(self.y.shape) > 2:
                        self.y = tf.argmax(self.y, -1, output_type=tf.int32)
                    else:
                        self.y = tf.cast(self.y, tf.int32)
                self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,
                    logits=self.logits))

            self.pred = tf.nn.softmax(self.logits)
        else:
            self.cost = tf.losses.mean_squared_error(
                    self.y, self.logits)
            self.pred = self.logits

        # define optimizer
        if self.optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            raise NotImplementedError

        self.regularizor = 0
        for variable in tf.trainable_variables():
            if 'b' not in variable.name:
                self.regularizor += tf.nn.l2_loss(variable)
        self.loss = self.cost + self.beta_l2 * self.regularizor

        # gradient clipping
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
#        self.norm = tf.global_norm(gradients)
        gradients_clipped, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer = optimizer.apply_gradients(zip(gradients_clipped, variables))

    def print_config(self):
        print "="*77

        print "Model configurations: %s" % self.name()
        print "batch_size: ", self.batch_size
        print "max_time: ", self.max_time
        print "n_input: ", self.n_input
        print "n_w: ", self.n_w
        print "n_h: ", self.n_h
        print "n_C: ", self.n_C
        print "n_output: ", self.n_output
        print "n_hidden: ", self.n_hidden
        print "is_classify: ", self.is_classify
        print "focal_loss: ", self.focal_loss
        print "input_keep_prob: ", self.input_keep_prob
        print "output_keep_prob: ", self.output_keep_prob
        print "beta_l2: ", self.beta_l2

        print "learning_rate: ", self.learning_rate
        print "optimizer: ", self.optimizer_name

        print "="*77
