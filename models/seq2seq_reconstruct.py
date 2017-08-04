import tensorflow as tf
from tqdm import tqdm
import numpy as np

'''
Sequence-to-sequence LSTM for history reconstruction
'''

class Seq2seqRecon(object):
    def name(self):
        return "Seq2seqRecon"

    def __init__(self, cfg):

        self.batch_size = cfg.batch_size
        self.max_time = cfg.max_time
        self.n_input = cfg.n_input
        self.n_hidden = cfg.n_hidden
        self.learning_rate = cfg.learning_rate
        self.optimizer_name = cfg.optimizer

        # define input and output
        self.x = tf.placeholder(tf.float32, (self.batch_size, self.max_time, self.n_input))
        self.y = tf.placeholder(tf.float32, (self.batch_size, self.max_time, self.n_input))
        self.seq_len = tf.placeholder(tf.int32, (self.batch_size,))

        # define variables
        self.W_ho = tf.get_variable(name="W_ho", shape=[self.n_hidden, self.n_input],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_o", shape=[self.n_input],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def build_network(self):

        # Encoder
        with tf.variable_scope('encoder'):
            encoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, self.x, self.seq_len, dtype=tf.float32, scope='encoder')

        # slice the valid output
        indices = tf.stack([tf.range(self.batch_size), self.seq_len-1], axis=1)
        self.feat = tf.gather_nd(self.encoder_outputs, indices)

        # Decoder
        # based on raw_rnn (see official reference)
        with tf.variable_scope('decoder'):
            decoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

        def loop_fn_initial():
            elements_finished = (-1 >= self.seq_len)
            next_input = tf.zeros([self.batch_size, self.n_input], dtype=tf.float32)
            next_cell_state = self.encoder_final_state
            emit_output = None
            next_loop_state = None

            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        def loop_fn_transition(time, cell_output, cell_state, loop_state):

            def linear_fc():
                return tf.matmul(cell_output, self.W_ho) + self.b_o

            emit_output = cell_output
            next_cell_state = cell_state

            elements_finished = (time >= self.seq_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                    finished,
                    lambda: tf.zeros([self.batch_size, self.n_input], dtype=tf.float32),
                    linear_fc)
            next_loop_state=None

            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_state is None:
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, cell_output, cell_state, loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn, scope='decoder')

        # fully connected layer to get the output
        pred = tf.TensorArray(dtype=tf.float32, size=self.max_time)
        for t in range(self.max_time):
            time = tf.constant(t, dtype=tf.int32)
            output = outputs_ta.read(time)
            pred = pred.write(time, tf.matmul(output, self.W_ho) + self.b_o)
            time += 1
        self.pred = pred.stack()

        # define loss
        self.loss = tf.losses.mean_squared_error(
                tf.reshape(self.y, (self.batch_size*self.max_time, self.n_input)),
                tf.reshape(self.pred, (self.batch_size*self.max_time, self.n_input)))

        # define optimizer
        if self.optimizer_name == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        else:
            raise NotImplementedError

