import tensorflow as tf
import numpy as np

'''
Convolutional embedding + bi-directional LSTM
'''

class ConvBiLSTM(object):
    def name(self):
        return "ConvBiLSTM"

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
        self.input_keep_prob = kwargs.get('input_keep_prob', cfg.input_keep_prob)
        self.output_keep_prob = kwargs.get('output_keep_prob', cfg.output_keep_prob)

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
            y -- output label / features, [batch_size, n_output]
            seq_len -- length indicator, [batch_size, ]
        """

        self.x = x
        self.y = y
        self.seq_len = seq_len

        def RNN(x):
            fw_cell = tf.contrib.rnn.LSTMCell(self.n_hidden/2, forget_bias=1.0)
            fw_dropout_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                                                     input_keep_prob=self.input_keep_prob,
                                                     output_keep_prob=self.output_keep_prob)

            bw_cell = tf.contrib.rnn.LSTMCell(self.n_hidden/2, forget_bias=1.0)
            bw_dropout_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,
                                                     input_keep_prob=self.input_keep_prob,
                                                     output_keep_prob=self.output_keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_dropout_cell, bw_dropout_cell, x, self.seq_len, dtype=tf.float32)

            # concatenation
            encoder_outputs = tf.concat(outputs, 2)

            # slice the valid output
            indices = tf.stack([tf.range(self.batch_size), self.seq_len-1], axis=1)
            return tf.gather_nd(encoder_outputs, indices)


        x_flat = tf.reshape(self.x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [self.batch_size, self.max_time, self.n_h*self.n_w*self.n_C])

        self.feat = RNN(x_emb)

        self.logits = tf.nn.xw_plus_b(self.feat, self.W_ho, self.b_o)

        # define loss
        if self.is_classify:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,
                logits=self.logits))
            self.pred = tf.argmax(self.logits, 1)
        else:
            self.loss = tf.losses.mean_squared_error(
                    self.y, self.logits)
            self.pred = self.logits

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
        print "input_keep_prob: ", self.input_keep_prob
        print "output_keep_prob: ", self.output_keep_prob

        print "learning_rate: ", self.learning_rate
        print "optimizer: ", self.optimizer_name

        print "="*77
