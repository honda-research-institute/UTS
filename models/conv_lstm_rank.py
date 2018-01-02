import tensorflow as tf
import numpy as np

'''
Convolutional embedding + LSTM
Ranking loss
'''

class ConvLSTMRank(object):
    def name(self):
        return "ConvLSTMRank"

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
        self.input_keep_prob = kwargs.get('input_keep_prob', cfg.input_keep_prob)
        self.output_keep_prob = kwargs.get('output_keep_prob', cfg.output_keep_prob)
        self.margin = kwargs.get('margin', cfg.margin)

        if self.n_output == 0:
            self.n_output = self.n_input


    def build_network(self, x1, x2, y, seq_len1, seq_len2):
        """
        Argument:
            x -- input features, [batch_size, max_time, n_h, n_w, n_input]
            y -- label, [batch_size, ]
            seq_len -- length indicator, [batch_size, ]
        """

        self.x1 = x1
        self.x2 = x2
        self.y = tf.cast(y, tf.float32)
        self.seq_len1 = seq_len1
        self.seq_len2 = seq_len2

        self.feat1 = self.conv_lstm(x1, seq_len1, reuse=False)
        self.feat2 = self.conv_lstm(x2, seq_len2, reuse=True)

        # contrastive loss
        d = tf.reduce_sum(tf.square(self.feat1 - self.feat2), 1)
        d_sqrt = tf.sqrt(d)

        loss = self.y * d + (1 - self.y) * tf.square(tf.maximum(0., self.margin - d_sqrt))
        self.loss = 0.5 * tf.reduce_mean(loss)

        # define optimizer
        if self.optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            raise NotImplementedError

        # gradient clipping
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
#        self.norm = tf.global_norm(gradients)
        gradients_clipped, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer = optimizer.apply_gradients(zip(gradients_clipped, variables))

    def conv_lstm(self, x, seq_len, reuse=False):
        def RNN(x, seq_len):
            encoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

            dropout_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,
                                                     input_keep_prob=self.input_keep_prob,
                                                     output_keep_prob=self.output_keep_prob)

            encoder_outputs, _ = tf.nn.dynamic_rnn(
                    dropout_cell, x, seq_len, dtype=tf.float32)

            # slice the valid output
            indices = tf.stack([tf.range(self.batch_size), seq_len-1], axis=1)
            return tf.gather_nd(encoder_outputs, indices)

        with tf.variable_scope("conv_lstm") as scope:
            if reuse:
                scope.reuse_variables()

            # define variables
            W_emb = tf.get_variable(name="W_emb", shape=[1, 1, self.n_input, self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)

            x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
            x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=W_emb,
                                            strides=[1, 1, 1, 1], padding="VALID",
                                            data_format="NHWC"))
            x_emb = tf.reshape(x_emb, [self.batch_size, self.max_time, self.n_h*self.n_w*self.n_C])
    
            return RNN(x_emb, seq_len)



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
        print "input_keep_prob: ", self.input_keep_prob
        print "output_keep_prob: ", self.output_keep_prob
        print "margin: ", self.margin

        print "learning_rate: ", self.learning_rate
        print "optimizer: ", self.optimizer_name

        print "="*77
