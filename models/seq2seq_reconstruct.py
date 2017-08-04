import tensorflow as tf
from utils.utils import iterate_minibatch1, iterate_minibatch3
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
        self.reverse = not cfg.no_reverse
        self.learning_rate = cfg.learning_rate
        self.optimizer_name = cfg.optimizer

        # define input and output
        self.x = tf.placeholder(tf.float32, (self.batch_size, self.max_time, self.n_input))
        self.y = self.x
        if reverse:
            self.y = tf.reverse(self.y, axis=[1])
        self.seq_len = tf.placeholder(tf.int32, (self.batch_size,))

        # define variables
        self.W_ho = tf.get_variable(name="W_ho", shape=[self.n_hidden, self.n_input],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_o", shape=[self.n_output],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def build_network(self):

        # Encoder
        encoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, self.x, self.seq_len, dtype=tf.float32)

        # slice the valid output
        indices = tf.stack([tf.range(self.batch_size), self.seq_len-1], axis=1)
        self.feat = tf.gather_nd(encoder_outputs, indices)

        # Decoder
        # based on raw_rnn (see official reference)
        decoder_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, forget_bias=1.0)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:    # time == 0
                next_cell_state = encoder_final_state
                emit_output = cell_output    # emit_output=None when time == 0
            else:
                next_cell_state = cell_state
                emit_output = tf.matmul(cell_output, self.W_ho) + b_o

            elements_finished = (time >= self.seq_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                    finished,
                    lambda: tf.zeros([self.batch_size, self.n_input], dtype=tf.float32),
                    lambda: emit_output)
            next_loop_state=None

            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        outputs_ta, final_state, _ = df.nn.raw_rnn(decoder_cell, loop_fn)
        self.pred = outputs_ta.stack()

        # define loss
        self.loss = tf.losses.mean_squared_error(
                tf.reshape(self.y, (self.batch_size*self.max_time, self.n_input)),
                tf.reshape(self.pred, (self.batch_size*self.max_time, self.n_input)))

        # define optimizer
        if self.optimizer_name == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        else:
            raise NotImplementedError

