import tensorflow as tf
import numpy as np

'''
Convolutional embedding + NN
'''

class ConvNNRank(object):
    def name(self):
        return "ConvNNRank"

    def __init__(self, cfg=None, **kwargs):

        self.batch_size = kwargs.get('batch_size', cfg.batch_size)
        self.n_input = kwargs.get('n_input', cfg.n_input)
        self.n_w = kwargs.get('n_w', cfg.n_w)
        self.n_h = kwargs.get('n_h', cfg.n_h)
        self.n_C = kwargs.get('n_C', cfg.n_C)
        self.n_hidden = kwargs.get('n_hidden', cfg.n_hidden)
        self.learning_rate = kwargs.get('learning_rate', cfg.learning_rate)
        self.optimizer_name = kwargs.get('optimizer', cfg.optimizer)
        self.beta_l2 = kwargs.get('beta_l2', cfg.beta_l2)
        self.margin = kwargs.get('margin', cfg.margin)


    def conv_nn(self, x, reuse=False):
        with tf.variable_scope("conv_nn") as scope:
            if reuse:
                scope.reuse_variables()

            # define variables
            W_emb = tf.get_variable(name="W_emb", shape=[1, 1, self.n_input, self.n_C],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=True)
            W_h = tf.get_variable(name="W_h", shape=[self.n_C*self.n_w*self.n_h, self.n_hidden],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=True)
            b_h = tf.get_variable(name="b_h", shape=[self.n_hidden],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
    
            x_emb = tf.nn.relu(tf.nn.conv2d(input=x, filter=W_emb,
                                            strides=[1, 1, 1, 1], padding="VALID",
                                            data_format="NHWC"))
            x_emb = tf.reshape(x_emb, [self.batch_size, -1])
    
            return tf.nn.xw_plus_b(x_emb, W_h, b_h)

    def build_network(self, x1, x2, y):
        """
        Argument:
            x -- input features, [batch_size, n_h, n_w, n_input]
            y -- [batch_size,], 1 for similar, 0 for dissimilar
        """

        self.x1 = x1
        self.x2 = x2
        self.y = tf.cast(y, tf.float32)

        self.feat1 = self.conv_nn(self.x1, reuse=False)
        self.feat2 = self.conv_nn(self.x2, reuse=True)

        # contrastive loss
        epsilon = 1e-8
        d = tf.reduce_sum(tf.square(self.feat1 - self.feat2), 1) + epsilon
        d_sqrt = tf.sqrt(d)
        self.d = d_sqrt

        cost = self.y * d + (1 - self.y) * tf.square(tf.maximum(0., self.margin - d_sqrt))
        self.cost = 0.5 * tf.reduce_mean(cost)

        self.regularizor = 0
        for variable in tf.trainable_variables():
            if 'b' not in variable.name:
                self.regularizor += tf.nn.l2_loss(variable)
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
        print "n_hidden: ", self.n_hidden
        print "beta_l2: ", self.beta_l2
        print "margin: ", self.margin

        print "learning_rate: ", self.learning_rate
        print "optimizer: ", self.optimizer_name

        print "="*77
