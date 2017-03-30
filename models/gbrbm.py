"""
Restricted Boltzmann Machines (RBM)
author: Ye Hu
2016/12/18
"""
import timeit
import numpy as np
import tensorflow as tf
from rbm import RBM


class GBRBM(RBM):
    """
    Gaussian-binary Restricted Boltzmann Machines
    Note we assume that the standard deviation is a constant (not training parameter)
    You better normalize you data with range of [0, 1.0].
    """
    def __init__(self, inpt=None, n_visiable=784, n_hidden=500, sigma=1.0, W=None,
                 hbias=None, vbias=None, sample_visible=True):
        """
        :param inpt: Tensor, the input tensor [None, n_visiable]
        :param n_visiable: int, number of visiable units
        :param n_hidden: int, number of hidden units
        :param sigma: float, the standard deviation (note we use the same σ for all visible units)
        :param W, hbias, vbias: Tensor, the parameters of RBM (tf.Variable)
        :param sample_visble: bool, if True, do gaussian sampling.
        """
        super(GBRBM, self).__init__(inpt, n_visiable, n_hidden, W, hbias, vbias)
        self.sigma = sigma
        self.sample_visible = sample_visible
    
    @staticmethod
    def sample_gaussian(x, sigma):
        return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma)

    def propdown(self, h):
        """Compute the mean for visible units given hidden units"""
        return tf.matmul(h, tf.transpose(self.W)) + self.vbias
    
    def sample_v_given_h(self, h0_sample):
        """Sampling the visiable units given hidden sample"""
        v1_mean = self.propdown(h0_sample)
        v1_sample = v1_mean
        if self.sample_visible:
            v1_sample = GBRBM.sample_gaussian(v1_mean, self.sigma)
        return (v1_mean, v1_sample)
    
    def propdown(self, h):
        """Compute the sigmoid activation for visible units given hidden units"""
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) / self.sigma**2 + self.vbias)
    
    def free_energy(self, v_sample):
        """Compute the free energy"""
        wx_b = tf.matmul(v_sample, self.W) / self.sigma**2 + self.hbias
        vbias_term = tf.reduce_sum(0.5 * tf.square(v_sample - self.vbias) / self.sigma**2, axis=1)
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(wx_b)), axis=1)
        return -hidden_term + vbias_term
    

if __name__ == "__main__":
    data = np.random.randn(1000, 6)
    x = tf.placeholder(tf.float32, shape=[None, 6])

    gbrbm = GBRBM(x, n_visiable=6, n_hidden=5)

    learning_rate = 0.1
    k = 1
    batch_size = 20
    n_epochs = 10

    cost = gbrbm.get_reconstruction_cost()
    # Create the persistent variable
    #persistent_chain = tf.Variable(tf.zeros([batch_size, n_hidden]), dtype=tf.float32)
    persistent_chain = None
    train_ops = gbrbm.get_train_ops(learning_rate=learning_rate, k=1, persistent=persistent_chain)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for epoch in range(n_epochs):
        avg_cost = 0.0
        for i in range(len(data)//batch_size):
            sess.run(train_ops, feed_dict={x: data[i*batch_size:(i+1)*batch_size]})
            avg_cost += sess.run(cost, feed_dict={x: data[i*batch_size:(i+1)*batch_size]})/batch_size
        print(avg_cost)
    
    # test
    v = np.random.randn(10, 6)
    print(v)

    preds = sess.run(gbrbm.reconstruct(x), feed_dict={x: v})
    print(preds)   

    