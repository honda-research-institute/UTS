import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import _transpose_batch_time

'''
Convolutional embedding + seq2seq
'''

class ConvSeq2seq(object):
    def name(self):
        return "ConvSeq2seq"

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
        self.seq_len = seq_len

        ###################### Encoder ###################

        def RNN(x):
            encoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

            dropout_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,
                                                     input_keep_prob=self.input_keep_prob,
                                                     output_keep_prob=self.output_keep_prob)

            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    dropout_cell, x, self.seq_len, dtype=tf.float32)

            # slice the valid output
            indices = tf.stack([tf.range(self.batch_size), self.seq_len-1], axis=1)
            return tf.gather_nd(encoder_outputs, indices), encoder_final_state


        x_flat = tf.reshape(self.x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [self.batch_size, self.max_time, self.n_h*self.n_w*self.n_C])

        self.feat, self.encoder_final_state = RNN(x_emb)

        ###################### Decoder ###################

        with tf.variable_scope('decoder'):
            decoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

        def loop_fn(time, cell_output, cell_state, loop_state):
            def get_next_input():
                if cell_state is None:
                    next_input = tf.zeros([self.batch_size, self.n_output], dtype=tf.float32)
                else:
                    next_input = tf.nn.xw_plus_b(cell_output, self.W_ho, self.b_o)
                return next_input
                
            emit_output = cell_output

            if cell_state is None:
                next_cell_state = self.encoder_final_state
            else:
                next_cell_state = cell_state

            elements_finished = (time >= self.seq_len)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                        finished,
                        lambda: tf.zeros([self.batch_size, self.n_output], dtype=tf.float32),
                        get_next_input)
            next_loop_state = None

            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn, scope='decoder')
        outputs = _transpose_batch_time(outputs_ta.stack())    # outputs and shape [batch_size, time ,output_dim]

        _, max_steps, _ = tf.unstack(tf.shape(outputs))
        self.logits = tf.nn.xw_plus_b(tf.reshape(outputs, (-1, self.n_hidden)), self.W_ho, self.b_o)
        self.y = self.y[:,:max_steps]

        # define loss
        if self.is_classify:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.y, (self.batch_size*max_steps,)),
                logits=self.logits))
            self.pred = tf.reshape(tf.argmax(self.logits, 1), (self.batch_size,max_steps,self.n_output))
        else:
            self.loss = tf.losses.mean_squared_error(
                    tf.reshape(self.y, (-1, self.n_output)),
                    self.logits)
            self.pred = tf.reshape(self.logits, (self.batch_size,max_steps,self.n_output))

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
        gradients_clipped, _ = tf.clip_by_global_norm(gradients, 5.0)
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
