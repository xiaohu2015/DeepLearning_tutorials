"""
Variational Autoencoder for MNIST data
reference: https://jmetzen.github.io/2015-11-27/vae.html
2017/01/17
"""
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from input_data import read_data_sets

# Random seeds for reproduce
np.random.seed(2017)
tf.set_random_seed(2017)

class VAE(object):
    """A simple class of variational autoencoder"""
    def __init__(self, input_dim=784, z_dim=50, batch_size=100, encoder_hidden_size=[500, 500], 
                    decoder_hidden_size=[500, 500], act_fn=tf.nn.softplus):
        """
        :param input_dim: int, the dimension of input
        :param z_dim: int, the dimension of latent space
        :param batch_size: int, batch size
        :param encoder_hidden_size: list or tuple, the number of hidden units in encoder
        :param decoder_hidden_size: list or tuple, the number of hidden units in decoder
        :param act_fn: the activation function
        """
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.act_fn = act_fn
        
        self._bulid_model()

    def _bulid_model(self):
        """The inner function to build the model"""
        # Input placeholder
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim])
        # The encoder: determine the mean and (log) variance of Gaussian distribution
        self.z_mean, self.z_log_sigma_sq = self._encoder(self.x)
        # Sampling from Gaussian distribution
        eps = tf.random_normal([self.batch_size, self.z_dim], mean=0.0, stddev=1.0)
        # z = mean + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Decoder: determine the mean of Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._decoder(self.z)
        
        # Compute the loss
        with tf.name_scope("loss"):
            # The reconstruction loss: cross entropy
            reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + \
                            (1.0 - self.x) * tf.log(1e-10 + 1.0 - self.x_reconstr_mean), axis=1)
            # The latent loss: KL divergence
            latent_loss = -0.5 * tf.reduce_sum(1.0 + self.z_log_sigma_sq - tf.square(self.z_mean) - \
                                    tf.exp(self.z_log_sigma_sq), axis=1)
            # Average over the batch
            self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        
        # The optimizer
        self.lr = tf.Variable(0.001, trainable=False)
        vars = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost, var_list=vars)
        
    def _encoder(self, x, name="encoder"):
        """Encoder"""
        with tf.variable_scope(name):
            n_in = self.input_dim
            for i, s in enumerate(self.encoder_hidden_size):
                w, b = self._get_vars(n_in, s, name="h{0}".format(i))
                if i == 0:
                    h = self.act_fn(tf.nn.xw_plus_b(x, w, b))
                else:
                    h = self.act_fn(tf.nn.xw_plus_b(h, w, b))
                n_in = s
            w, b = self._get_vars(n_in, self.z_dim, name="out_mean")
            z_mean = tf.nn.xw_plus_b(h, w, b)
            w, b = self._get_vars(n_in, self.z_dim, name="out_log_sigma")
            z_log_sigma_sq = tf.nn.xw_plus_b(h, w, b)
            return z_mean, z_log_sigma_sq
        
    def _decoder(self, z, name="decoder"):
        """Decoder"""
        with tf.variable_scope(name):
            n_in = self.z_dim
            for i, s in enumerate(self.decoder_hidden_size):
                w, b = self._get_vars(n_in, s, name="h{0}".format(i))
                if i == 0:
                    h = self.act_fn(tf.nn.xw_plus_b(z, w, b))
                else:
                    h = self.act_fn(tf.nn.xw_plus_b(h, w, b))
                n_in = s
            # Use sigmoid for Bernoulli distribution
            w, b = self._get_vars(n_in, self.input_dim, name="out_mean")
            x_reconstr_mean = tf.nn.sigmoid(tf.nn.xw_plus_b(h, w, b))
            return x_reconstr_mean

    def _get_vars(self, n_in, n_out, name=""):
        """
        Create weight and bias variables 
        """
        with tf.variable_scope(name):
            w = tf.get_variable("w", [n_in, n_out], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", [n_out,], initializer=tf.constant_initializer(0.1))
            return w, b

if __name__ == "__main__":
    n_epochs = 30
    lr = 0.001
    batch_size = 100
    display_every = 1

    path = sys.path[0]
    mnist = read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as sess:
        vae = VAE(input_dim=784, z_dim=2, batch_size=batch_size, encoder_hidden_size=[500, 500],
                    decoder_hidden_size=[500, 500], act_fn=tf.nn.softplus)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #saver.restore(sess, save_path=path+"/model/model.ckpt")
        # Start training
        print("Start training...")
        total_batch = int(mnist.train.num_examples/batch_size)
        for epoch in range(n_epochs):
            avg_cost = 0.0
            # For each batch 
            for i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(batch_size)
                c, _ = sess.run([vae.cost, vae.train_op], feed_dict={vae.x: batch_xs})
                avg_cost += c/total_batch
            if epoch % display_every == 0:
                save_path = saver.save(sess, path+"/model/model.ckpt")
                #print("\tModel saved in file: {0}".format(save_path))
                print("\tEpoch {0}, cost {1}".format(epoch, avg_cost))
        
        # Sampling
        x_sample, _ = mnist.test.next_batch(batch_size)
        x_reconstr = sess.run(vae.x_reconstr_mean, feed_dict={vae.x: x_sample})
        plt.figure(figsize=(8, 12))
        for i in range(5):
            plt.subplot(5, 2, 2*i + 1)
            plt.imshow(np.reshape(x_sample[i],(28, 28)), vmin=0, vmax=1, cmap="gray")
            plt.title("Test input")
            plt.colorbar()
            plt.subplot(5, 2, 2*i + 2)
            plt.imshow(np.reshape(x_reconstr[i], [28, 28]), vmin=0, vmax=1, cmap="gray")
            plt.title("Reconstruction")
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(path+"/results/img_epoch{0}.jpg".format(n_epochs))
        plt.show()

        # Random sampling
        nx, ny = 20, 20
        xs = np.linspace(-3, 3, nx)
        ys = np.linspace(-3, 3, ny)
        xs, ys = np.meshgrid(xs, ys)
        xs = np.reshape(xs, [-1, 1])
        ys = np.reshape(ys, [-1, 1])
        zs = np.concatenate((xs, ys), axis=1)

        canvas = np.zeros((28*ny, 28*nx))
        xs_recon = np.zeros((batch_size*4, 28*28))
        for i in range(4):
            z_mu = zs[batch_size*i:batch_size*(i+1), :]
            x_mean = sess.run(vae.x_reconstr_mean, feed_dict={vae.z: z_mu})
            xs_recon[i*batch_size:(i+1)*batch_size] = x_mean
        
        n = 0
        for i in range(nx):
            for j in range(ny):
                canvas[(ny-i-1)*28:(ny-i)*28, j*28:(j+1)*28] = xs_recon[n].reshape(28, 28)
                n = n + 1
        
        plt.figure(figsize=(8, 10))
        plt.imshow(canvas, origin="upper", vmin=0, vmax=1, interpolation='none', cmap='gray')
        plt.tight_layout()
        plt.savefig(path+"/results/rand_img_epoch{0}.jpg".format(n_epochs))
        plt.show()
