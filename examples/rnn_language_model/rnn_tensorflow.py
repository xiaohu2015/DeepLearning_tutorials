"""
A simple RNN model implemented by Tensorflow
author: Ye Hu
2016/12/24
"""
import timeit
from datetime import datetime
import numpy as np
import tensorflow as tf

from input_data_rnn import get_data

class RNN_tf(object):
    """
    A RNN class for the language model
    """
    def __init__(self, inpt=None, word_dim=8000, hidden_dim=100, bptt_truncate=4):
        """
        :param inpt: tf.Tensor, the input tensor
        :param word_dim: int, the number of word in the input sentence
        :param hidden_dim: int, the size of hidden units
        :param bptt_truncate: int, (TO DO:)
        """
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        if inpt is None:
            inpt = tf.placeholder(tf.int32, shape=[None, ])
        self.x = inpt
        self.y = tf.placeholder(tf.int32, shape=[None, ])

        # Initialize the network parameters
        bounds = np.sqrt(1.0/self.word_dim)
        # Input weight matrix
        self.U = tf.Variable(tf.random_uniform([self.word_dim, self.hidden_dim], minval=-bounds, maxval=bounds), 
                             name="U")
        bounds = np.sqrt(1.0/self.hidden_dim)
        self.W = tf.Variable(tf.random_uniform([self.hidden_dim, self.hidden_dim], minval=-bounds, maxval=bounds),
                                name="W")         # old state weight matrix
        self.V = tf.Variable(tf.random_uniform([self.hidden_dim, self.word_dim], minval=-bounds, maxval=bounds),
                                name="V")         # the output weight matrix
        # Keep track of all parameters for training
        self.params = [self.U, self.W, self.V]
        # Build the model
        self.__model_build__()
    
    def __model_build__(self):
        """
        A private method to build the RNN model
        """
        # The inner function for forward propagation
        def forward_propagation(s_t_prv, x_t):
            s_t = tf.nn.tanh(tf.slice(self.U, [x_t, 0], [1, -1]) + tf.matmul(s_t_prv, self.W))
            return s_t
        # Use scan function to get the hidden state of all times
        s = tf.scan(forward_propagation, self.x, initializer=tf.zeros([1, self.hidden_dim]))  # [seq_len, 1, hidden_dim]
        s = tf.squeeze(s)  # [seq_len, hidden_dim]
        # The output
        o_wx = tf.matmul(s, self.V)
        o = tf.nn.softmax(o_wx)
        # The right prediction
        self.prediction = tf.argmax(o, axis=1)
        # The cost for training
        self.cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(o_wx, self.y))
        self.loss = self.cost / tf.cast(tf.size(self.x), tf.float32)

        

def train_rnn_with_sgd(sess, model, X_train, y_train, learning_rate=0.005, n_epochs=100,
                       evaluate_loss_after=5):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.cost, var_list=model.params)
    N = len(X_train)  # number of training examples
    print("Start training...")
    start_time = timeit.default_timer()
    for epoch in range(n_epochs):
        # If output the loss for all training examples
        if epoch % evaluate_loss_after == 0:
            losses = 0
            for i in range(N):
                losses += sess.run(model.loss, feed_dict={model.x: X_train[i], model.y: y_train[i]})
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("\t{0}:Loss after Epoch {1} is {2}".format(time, epoch, losses/N))
        # Traing each by each
        for i in range(N):
            sess.run(train_op, feed_dict={model.x: X_train[i], model.y: y_train[i]})
    end_time = timeit.default_timer()
    print("Finished!")
    print("Time elapsed {0} minutes.".format((end_time-start_time)/60.0))

if __name__ == "__main__":
    np.random.seed(10)
    tf.set_random_seed(1111)
    vocabulary_size = 8000
    X_train, y_train = get_data(vocabulary_size=vocabulary_size)
    
    with tf.Session() as sess:
        model = RNN_tf(inpt=None, word_dim=8000, hidden_dim=100)
        sess.run(tf.global_variables_initializer())
        train_rnn_with_sgd(sess, model, X_train[:1000], y_train[:1000], n_epochs=10, evaluate_loss_after=1)