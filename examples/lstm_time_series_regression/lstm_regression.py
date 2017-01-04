"""
LSTM Model for Time Series Prediction/Regression
source: 'https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2.2/full_code.py'
2017/01/03
"""
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def batch_iterate(num_batchs, batch_size, num_steps):
    """
    Generate the mini batch about sin and cos function
    """
    start = 0
    for i in range(num_batchs):
        xo = np.arange(start, start+batch_size*num_steps).reshape(
                            [batch_size, num_steps])/(10.0*np.pi)
        x = np.sin(xo)
        y = np.cos(xo)
        start += num_steps
        yield (x[:, :, np.newaxis], y[:, :, np.newaxis], xo)

class LstmRegression(object):
    """
    A lstm class for time series prediction
    """
    def __init__(self, in_size, out_size, num_steps=20, cell_size=20, batch_size=50,
                    num_lstm_layers=2, keep_prob=0.5, is_training=True):
        """
        :param in_size: int, the dimension of input
        :param out_size: int, the dimension of output
        :param num_steps: int, the number of time steps
        :param cell_size: int, the size of lstm cell
        :param batch_size: int, the size of mini bacth
        :param num_lstm_layers: int, the number of lstm cells
        :param keep_prob: float, the keep probability of dropout layer
        :param is_training: bool, set True for training model, but False for test model
        """
        self.in_size = in_size
        self.out_size = out_size
        self.num_steps = num_steps
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.num_lstm_layers = num_lstm_layers
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.__build_model__()

    def __build_model__(self):
        """
        The inner method to construct the lstm model.
        """
        # Input and output placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.in_size])
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.out_size])

        # Add the first input layer
        with tf.variable_scope("input"):
            # Reshape x to 2-D tensor
            inputs = tf.reshape(self.x, shape=[-1, self.in_size])  #[batch_size*num_steps, in_size]
            W, b = self._get_weight_bias(self.in_size, self.cell_size)
            inputs = tf.nn.xw_plus_b(inputs, W, b, name="input_xW_plus_b")
        # Reshep to 3-D tensor
        inputs = tf.reshape(inputs, shape=[-1, self.num_steps, self.cell_size]) #[batch_size, num_steps, in_size]

        # Dropout the inputs
        if self.is_training and self.keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
        
        # Construct lstm cells
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        if self.is_training and self.keep_prob < 1.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.num_lstm_layers)
        # The initial state
        self.init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # Add the lstm layer
        with tf.variable_scope("LSTM"):
            outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.init_state)
        self.final_state = final_state

        # Add the output layer
        with tf.variable_scope("output"):
            output = tf.reshape(outputs, shape=[-1, self.cell_size])
            W, b = self._get_weight_bias(self.cell_size, self.out_size)
            output = tf.nn.xw_plus_b(output, W, b, name="output")
        
        self.pred = output
        losses = tf.nn.seq2seq.sequence_loss_by_example([tf.reshape(self.pred, [-1,])], [tf.reshape(self.y, [-1,])],
                                    [tf.ones([self.batch_size*self.num_steps])], average_across_timesteps=True,
                                    softmax_loss_function=self._ms_cost)
        self.cost = tf.reduce_sum(losses)/tf.to_float(self.batch_size)

    def _ms_cost(self, y_pred, y_target):
        """The quadratic cost function"""
        return 0.5*tf.square(y_pred - y_target)

    def _get_weight_bias(self, in_size, out_size):
        """
        Create weight and bias variables
        """
        weights = tf.get_variable("weight", shape=[in_size, out_size], 
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        biases = tf.get_variable("bias", shape=[out_size,], initializer=tf.constant_initializer(0.1))
        return weights, biases

if __name__ == "__main__":
    batch_size = 50
    in_size = 1
    out_size = 1
    cell_size = 10
    num_steps = 20
    lr = 0.002
    num_batchs = 200
    n_epochs = 10

    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=None):
            model = LstmRegression(in_size, out_size, num_steps=num_steps, cell_size=cell_size, 
                            batch_size=batch_size, num_lstm_layers=2, keep_prob=0.5, is_training=True)
        with tf.variable_scope("model", reuse=True):
            pred_model = LstmRegression(in_size, out_size, num_steps=num_steps, cell_size=cell_size, 
                            batch_size=batch_size, num_lstm_layers=2, keep_prob=1.0, is_training=False)
        
        train_op = tf.train.AdamOptimizer(lr).minimize(model.cost)
        tf.summary.scalar("cost", model.cost)
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("logs", sess.graph)
        sess.run(tf.global_variables_initializer())
        
        global_steps = 0
        state = sess.run(model.init_state)
        for epoch in range(n_epochs):
            losses = 0
            for x, y, xo in batch_iterate(num_batchs, batch_size, num_steps):
                _, cost, state = sess.run([train_op, model.cost, model.final_state], feed_dict={model.x: x,
                                            model.y: y, model.init_state: state})
                losses += cost/num_batchs
            print("Epoch {0}, cost {1}".format(epoch, losses))
        
        # The prediction
        plt.ion()
        plt.show()
        state = sess.run(pred_model.init_state)
        for x, y, xo in batch_iterate(num_batchs, batch_size, num_steps):
            pred, state = sess.run([pred_model.pred, pred_model.final_state], feed_dict={pred_model.x: x,
                                    pred_model.y: y, pred_model.init_state: state })

            # plotting
            plt.plot(xo[0, :], y[0].flatten(), 'r', xo[0, :], pred.flatten()[:num_steps], 'b--')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.3)
            
