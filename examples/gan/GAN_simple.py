"""
A simple generative adversarial networks (GAN)
source: https://github.com/AYLIEN/gan-intro/blob/master/gan.py
2017/01/12
"""
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed to reproduce
seed = 24
np.random.seed(seed)
tf.set_random_seed(seed)

class NormDistribution(object):
    """ 1-D Guassian Distribution"""
    def __init__(self, mu=-1, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, n):
        """
        Sample form the norm distribution
        :param n: int, the number of samples
        """
        samples = np.random.normal(loc=self.mu, scale=self.sigma, size=[n,])
        samples.sort()   # stratified sampling by sorting the samples
        return samples

class NoiseInput(object):
    """
    The nosie input `z` for the generator. 
    """
    def __init__(self, scope):
        """
        :param scope: int, `z` are generated in the range of [-scope, scope]
        """
        self.scope = scope
    
    def sample(self, n):
        """
        Sample form the noise input, the samples are sorted with some noise.
        :param n: int, the number of samples
        """
        return np.linspace(-self.scope, self.scope, n) + np.random.random(n)*0.01

# linear layer
def linear(input, output_dim, stddev=1.0, scope="linear"):
    input_dim = input.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", shape=[input_dim, output_dim], initializer=
                                tf.random_normal_initializer(mean=0.0, stddev=stddev))
        b = tf.get_variable("b", shape=[output_dim,], initializer=tf.constant_initializer(0.0))
        return tf.nn.xw_plus_b(input, w, b)

# Minibatch for discriminator
def minibatch(input, num_kernels=5, kernel_dim=3):
    """
    The minibatch method for the discriminator
    """
    x = linear(input, num_kernels*kernel_dim, stddev=0.02, scope="minibatch")
    activation = tf.reshape(x, shape=[-1, num_kernels, kernel_dim])
    # Compute the L1 distance over rows
    diffs = tf.expand_dims(activation, -1) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)
    return tf.concat(1, [x, minibatch_features])

class Generator(object):
    """A class of generator"""
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def __call__(self, input):
        """We only use two layers"""
        h0 = tf.nn.softplus(linear(input, self.hidden_size, scope="g0"))
        h1 = tf.tanh(linear(h0, 1, scope="g1"))
        return h1

class Discriminator(object):
    """A class of discriminator"""
    def __init__(self, hidden_size, minibatch_layer=True):
        self.hidden_size = hidden_size
        self.minibatch_layer = minibatch_layer
    
    def __call__(self, input):
        """We use more hidden layers"""
        h0 = tf.tanh(linear(input, self.hidden_size*2, scope="d0"))
        h1 = tf.tanh(linear(h0, self.hidden_size*2, scope="d1"))
        # We add a layer if you don not use minibatch method
        if self.minibatch_layer:
            h2 = minibatch(h1)
        else:
            h2 = tf.tanh(linear(h1, self.hidden_size*2, scope="d2"))
        h3 = tf.sigmoid(linear(h2, 1, scope="d3"))
        return h3

def optimizer(loss, var_list, init_lr):
    decay = 0.95
    num_decay_steps = 150
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(init_lr, global_step, num_decay_steps, decay, 
                                    staircase=True)
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step,
                                                                var_list=var_list)
    return train_op

class GAN(object):
    """A simple generative adversarial network to train 1-D norm distribution"""
    def __init__(self, data, z_data, hidden_size=4, is_minibatch=True):
        """
        :param data: a object to generate the true data distribution
        :param z_data: a object to generate nosie input for Generator
        :param hidden_size: int, the number of units in mlp
        :param is_minibatch: bool, if use minibatch method in discriminator
        """
        self.data = data
        self.z_data = z_data
        self.hidden_size = hidden_size
        self.is_minibatch = is_minibatch
        if is_minibatch:
            self.lr = 0.005
        else:
            self.lr = 0.03
        self._bulid_model()
    
    def _bulid_model(self):
        """The inner function to build the model"""
        # Pretrain the discriminator is helpful to GAN
        with tf.variable_scope("D_pre"):
            self.pre_x = tf.placeholder(tf.float32, shape=[None, 1])
            self.pre_y = tf.placeholder(tf.float32, shape=[None, 1])
            D_pre = Discriminator(self.hidden_size, self.is_minibatch)
            y = D_pre(self.pre_x)
            # Use mse loss
            self.pre_loss = tf.reduce_mean(tf.square(y - self.pre_y))

        # Generator model
        with tf.variable_scope("G"):
            self.z = tf.placeholder(tf.float32, shape=[None, 1])
            G = Generator(self.hidden_size)(self.z)
            self.G = tf.mul(G, self.z_data.scope)
            #self.G = tf.clip_by_value(self.G, 0.01, 0.999)
        
        # Discriminator model
        with tf.variable_scope("D") as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, 1])
            self.D1 = Discriminator(self.hidden_size, self.is_minibatch)(self.x)
            #self.D1 = tf.clip_by_value(self.D1, 0.01, 0.99)
            # Reuse the model
            scope.reuse_variables()
            self.D2 = Discriminator(self.hidden_size, self.is_minibatch)(self.G)
            #self.D2 = tf.clip_by_value(self.D2, 0.01, 0.999)

        # Compute the loss
        self.d_loss = -tf.reduce_mean(tf.log(self.D1) + tf.log(1.0 - self.D2))
        self.g_loss = -tf.reduce_mean(tf.log(self.D2))

        # Get the trainable vars for each model
        vars = tf.trainable_variables()
        self.d_pre_vars = sorted([v for v in vars if v.name.startswith("D_pre/")], key=lambda v: v.name)
        self.d_vars = sorted([v for v in vars if v.name.startswith("D/")], key=lambda v: v.name)
        self.g_vars = [v for v in vars if v.name.startswith("G/")]

        # Train_ops
        self.d_pre_train_op = optimizer(self.pre_loss, self.d_pre_vars, self.lr)
        self.d_train_op = optimizer(self.d_loss, self.d_vars, self.lr)
        self.g_train_op = optimizer(self.g_loss, self.g_vars, self.lr)

    def pretrain_discriminator(self, sess, batch_size=20, n_epochs=1000, display_every=50):
        """Pretrain the discriminator"""
        losses = []
        for epoch in range(n_epochs):
            x = (np.random.random(batch_size) - 0.5)*10.0
            y = norm.pdf(x, loc=self.data.mu, scale=self.data.sigma)
            loss, _ = sess.run([self.pre_loss, self.d_pre_train_op], feed_dict={self.pre_x: np.reshape(x, [-1, 1]), 
                                                                     self.pre_y: np.reshape(y, [-1, 1])})
            losses.append(loss)
            if epoch % display_every == 0:
                print("Epoch {0}, pretrain loss: {1}".format(epoch, loss))

        pretrain_vars = sess.run(self.d_pre_vars)
        for pre_v, v in zip(pretrain_vars, self.d_vars):
            sess.run(tf.assign(v, pre_v))

        # Plot the losses
        f, ax = plt.subplots(1)
        ax.plot(np.arange(n_epochs), np.array(losses))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Pretrain losses")
        plt.show()

    def train(self, sess, batch_size=20, n_epochs=100, d_k=1, display_every=10):
        """Train GAN"""

        for epoch in range(n_epochs):
            # train D
            d_losses = 0.0
            for i in range(d_k):
                x = np.reshape(self.data.sample(batch_size), [-1, 1])
                z = np.reshape(self.z_data.sample(batch_size), [-1, 1])
                d_loss, _ = sess.run([self.d_loss, self.d_train_op], feed_dict={self.x: x,
                                        self.z: z})
                d_losses += d_loss/d_k
            # train G
            z = np.reshape(self.z_data.sample(batch_size), [-1, 1])
            g_loss, _ = sess.run([self.g_loss, self.g_train_op], feed_dict={self.z: z})
            if epoch % display_every == 0:
                print("Epoch {0}, d_loss {1}, g_loss {2}".format(epoch, d_losses, g_loss)) 

    def _sample(self, sess, batch_size =20, num_points=10000, num_bins=100):
        """Sampler"""
        # Decision boundary given by Discriminator
        xs = np.linspace(-self.z_data.scope, self.z_data.scope, num_points)
        dbs = np.zeros((num_points,))
        for i in range(num_points // batch_size):
            x = np.reshape(xs[i*batch_size:(i+1)*batch_size], [-1, 1])
            db = sess.run(self.D1, feed_dict={self.x: x})
            dbs[i*batch_size:(i+1)*batch_size] = np.reshape(db, [-1])
        
        # True data distribution
        bins = np.linspace(-self.z_data.scope, self.z_data.scope, num_bins)
        d = self.data.sample(num_points)
        pds, _ = np.histogram(d, bins=bins, density=True)
        
        # The generated distribution
        zs = np.linspace(-self.z_data.scope, self.z_data.scope, num_points)
        gds = np.zeros((num_points))
        for i in range(num_points // batch_size):
            z = np.reshape(zs[i*batch_size:(i+1)*batch_size], [-1, 1])
            gd = sess.run(self.G, feed_dict={self.z: z})
            gds[i*batch_size:(i+1)*batch_size] = np.reshape(gd, [-1])
        
        gds, _ = np.histogram(gds, bins=bins, density=True)
    
        return (dbs, pds, gds)

    def ploter(self, sess, num_points=10000, num_bins=100):
        """Plot decision boundary, true data distribution, 
        generated distribution"""
        dbs, pds, gds = self._sample(sess, batch_size =20, num_points=num_points,
                                     num_bins=num_bins)
        f, ax = plt.subplots(1)
        x1 = np.linspace(-self.z_data.scope, self.z_data.scope, len(dbs))
        x2 = np.linspace(-self.z_data.scope, self.z_data.scope, len(pds))
        ax.plot(x1, dbs, label="Decision boundary")
        ax.plot(x2, pds, label="Data")
        ax.plot(x2, gds, label="G_data")
        ax.set_ylim(0, 1.2)
        plt.title("1-D Norm Distribution")
        plt.xlabel("Random variable")
        plt.ylabel("Probability density")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    sess = tf.Session()
    
    gan = GAN(NormDistribution(-1, 1), NoiseInput(5), hidden_size=4, is_minibatch=False)
    sess.run(tf.global_variables_initializer())
    gan.pretrain_discriminator(sess, batch_size=12, n_epochs=1000)
    gan.ploter(sess)
    gan.train(sess, batch_size=10, n_epochs=1000, d_k=1, display_every=10)
    gan.ploter(sess)



    
    