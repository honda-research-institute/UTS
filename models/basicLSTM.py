import tensorflow as tf

'''
Basic LSTM for next timestamp prediction
'''

class basicLSTM():
    def __init__(self, batch_size, n_steps, n_input, n_hidden, n_output=None):

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        if n_output is None:
            self.n_output = self.n_input
        else:
            self.n_output = n_output


        # define variables and weights
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.n_input])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.n_input])

        W_ho = tf.get_variable(name="W_ho", shape=[self.n_hidden, self.n_output],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=True)
        b_o = tf.get_variable(name="b_o", shape=[self.n_output],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

        def RNN(x):

            x = tf.unstack(x, self.n_steps, 1)

            lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)

            outputs, _ = tf.nn.static_rnn(cell=lstm_cell,
                                            inputs=x,
                                            dtype=tf.float32)

            return outputs[-1]


        ############## Build the network ################

        self.hidden = RNN(self.x)

        self.pred = tf.matmul(self.hidden, W_ho) + b_o

        # define loss
        self.cost = tf.losses.mean_squared_error(self.y, self.pred)
