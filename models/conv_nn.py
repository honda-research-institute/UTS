import tensorflow as tf
import numpy as np
from utils.utils import focal_loss

'''
Convolutional embedding + NN
'''

class ConvNN(object):
    def name(self):
        return "ConvNN"

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
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
        self.output_keep_prob = kwargs.get('output_keep_prob', cfg.output_keep_prob)
        self.beta_l2 = kwargs.get('beta_l2', cfg.beta_l2)

        if self.n_output == 0:
            self.n_output = self.n_input

        # define variables
        self.W_emb = tf.get_variable(name="W_emb", shape=[1, 1, self.n_input, self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        self.W_h = tf.get_variable(name="W_h", shape=[self.n_C*self.n_w*self.n_h, self.n_hidden],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        self.b_h = tf.get_variable(name="b_h", shape=[self.n_hidden],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_ho = tf.get_variable(name="W_ho", shape=[self.n_hidden, self.n_output],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_o", shape=[self.n_output],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def build_network(self, x, y):
        """
        Argument:
            x -- input features, [batch_size, n_h, n_w, n_input]
            y -- output label / features, [batch_size, n_output]
        """

        self.x = x
        self.y = y

        x_emb = tf.nn.relu(tf.nn.conv2d(input=self.x, filter=self.W_emb,
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [self.batch_size, -1])

        self.feat = tf.nn.xw_plus_b(x_emb, self.W_h, self.b_h)
        h_drop = tf.nn.dropout(tf.nn.relu(self.feat), self.output_keep_prob)

        self.logits = tf.nn.xw_plus_b(h_drop, self.W_ho, self.b_o)

        # define loss
        if self.is_classify:
            if self.focal_loss:
                # use focal loss
                self.cost = focal_loss(labels=tf.reshape(self.y, [-1, self.n_output]), 
                                   logits=tf.reshape(self.logits,[-1, self.n_output]))
            else:
                if not self.y.dtype == tf.int32:
                    if len(self.y.shape) > 1:
                        self.y = tf.argmax(self.y, -1, output_type=tf.int32)
                    else:
                        self.y = tf.cast(self.y, tf.int32)
                self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,
                    logits=self.logits))
            self.pred = tf.nn.softmax(self.logits)
        else:
            self.pred = self.logits
            self.cost = tf.losses.mean_squared_error(
                    self.y, self.logits)

        self.regularizor = tf.nn.l2_loss(self.W_emb) + tf.nn.l2_loss(self.W_h) + tf.nn.l2_loss(self.W_ho)
        self.loss = self.cost + self.beta_l2 * self.regularizor

        # define optimizer
        if self.optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            raise NotImplementedError

        self.optimizer = optimizer.minimize(self.loss)

    def print_config(self):
        print "="*77

        print "Model configurations: %s" % self.name()
        print "batch_size: ", self.batch_size
        print "n_input: ", self.n_input
        print "n_w: ", self.n_w
        print "n_h: ", self.n_h
        print "n_C: ", self.n_C
        print "n_output: ", self.n_output
        print "n_hidden: ", self.n_hidden
        print "is_classify: ", self.is_classify
        print "focal_loss: ", self.focal_loss
        print "output_keep_prob: ", self.output_keep_prob
        print "beta_l2: ", self.beta_l2

        print "learning_rate: ", self.learning_rate
        print "optimizer: ", self.optimizer_name

        print "="*77
