"""
2017/01/09
"""
import sys
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from PIL import Image

# Batch normalization
def batch_norm(inpt, epsilon=1e-05, decay=0.9, is_training=True, name="batch_norm"):
    """
    Implements the bacth normalization
    The input is 4-D tensor
    """
    bn = tf.contrib.layers.batch_norm(inpt, decay=decay, updates_collections=None,
                                    epsilon=epsilon, scale=True, is_training=is_training, scope=name)
    return bn

# Convolution 2-D 
def conv2d(inpt, nb_filter, filter_size=5, strides=2, bias=True, stddev=0.02, padding="SAME", 
            name="conv2d"):
    in_channels = inpt.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape=[filter_size, filter_size, in_channels, nb_filter],
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev))
        conv = tf.nn.conv2d(inpt, w, strides=[1, strides, strides, 1], padding=padding)
        if bias:
            b = tf.get_variable("b", shape=[nb_filter,], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)
        return conv

# Convolution 2D Transpose
def deconv2d(inpt, output_shape, filter_size=5, strides=2, bias=True, stddev=0.02,
              padding="SAME", name="deconv2d"):
    in_channels = inpt.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        # Note: filter with shape [height, width, output_channels, in_channels]
        w = tf.get_variable("w", shape=[filter_size, filter_size, output_shape[-1], in_channels],
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev))
        deconv = tf.nn.conv2d_transpose(inpt, w, output_shape=output_shape, strides=[1, strides, strides, 1],
                                        padding=padding)
        if bias:
            b = tf.get_variable("b", shape=[output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)
        return deconv

# Leaky ReLU
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, x*leak)

# Linear 
def linear(x, output_dim, stddev=0.02, name="linear"):
    input_dim = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape=[input_dim, output_dim], initializer=\
                        tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", shape=[output_dim,], initializer=tf.constant_initializer(0.0))
        return tf.nn.xw_plus_b(x, w, b)

class DCGAN(object):
    """A class of DCGAN model"""
    def __init__(self, z_dim=100, output_dim=28, batch_size=100, c_dim=1, df_dim=64, gf_dim=64, dfc_dim=1024,
                  n_conv=3, n_deconv=2):
        """
        :param z_dim: int, the dimension of z (the noise input of generator)
        :param output_dim: int, the resolution in pixels of the images (height, width)
        :param batch_size: int, the size of the mini-batch
        :param c_dim: int, the dimension of image color, for minist, it is 1 (grayscale)
        :param df_dim: int, the number of filters in the first convolution layer of discriminator
        :param gf_dim: int, the number of filters in the penultimate deconvolution layer of generator (last is 1)
        :param dfc_dim: int, the number of units in the penultimate fully-connected layer of discriminator (last is 1)
        :param n_conv: int, number of convolution layer in discriminator (the number of filters is double increased)
        :param n_deconv: int, number of deconvolution layer in generator (the number of filters is double reduced)
        """
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.c_dim = c_dim
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.dfc_dim = dfc_dim
        self.n_conv = n_conv
        self.n_deconv = n_deconv
        self.batch_size = batch_size

        self._build_model()
    
    def _build_model(self):
        # input 
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dim, 
                                                    self.output_dim, self.c_dim])
        
        # G
        self.G = self._generator(self.z)
        # D
        self.D1, d1_logits = self._discriminator(self.x, reuse=False)
        self.D2, d2_logits = self._discriminator(self.G, reuse=True)

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d2_logits, tf.ones_like(self.D2)))
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(d1_logits, tf.ones_like(self.D1))
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(d2_logits, tf.zeros_like(self.D2))
        self.d_loss = tf.reduce_mean(real_loss + fake_loss)

        vars = tf.trainable_variables()
        self.d_vars = [v for v in vars if "D" in v.name]
        self.g_vars = [v for v in vars if "G" in v.name]

    def _discriminator(self, input, reuse=False):
        with tf.variable_scope("D", reuse=reuse):
            h = lrelu(conv2d(input, nb_filter=self.df_dim, name="d_conv0"))
            for i in range(1, self.n_conv):
                conv = conv2d(h, nb_filter=self.df_dim*(2**i), name="d_conv{0}".format(i))
                h = lrelu(batch_norm(conv, name="d_bn{0}".format(i)))
            h = linear(tf.reshape(h, shape=[self.batch_size, -1]), self.dfc_dim, name="d_lin0")
            h = linear(tf.nn.tanh(h), 1, name="d_lin1")
            return tf.nn.sigmoid(h), h

    def _generator(self, input):
        with tf.variable_scope("G"):
            nb_fliters = [self.gf_dim]
            f_size = [self.output_dim//2]
            for i in range(1, self.n_deconv):
                nb_fliters.append(nb_fliters[-1]*2)
                f_size.append(f_size[-1]//2)
    
            h = linear(input, nb_fliters[-1]*f_size[-1]*f_size[-1], name="g_lin0")
            h = tf.nn.relu(batch_norm(tf.reshape(h, shape=[-1, f_size[-1], f_size[-1], nb_fliters[-1]]),
                            name="g_bn0"))
            for i in range(1, self.n_deconv):
                h = deconv2d(h, [self.batch_size, f_size[-i-1], f_size[-i-1], nb_fliters[-i-1]], 
                                    name="g_deconv{0}".format(i-1))
                h = tf.nn.relu(batch_norm(h, name="g_bn{0}".format(i)))
            
            h = deconv2d(h, [self.batch_size, self.output_dim, self.output_dim, self.c_dim], 
                            name="g_deconv{0}".format(self.n_deconv-1))
            return tf.nn.tanh(h)

def combine_images(images):
    """Combine the bacth images"""
    num = images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(num/width))
    h, w = images.shape[1:-1]
    img = np.zeros((height*h, width*w), dtype=images.dtype)
    for index, m in enumerate(images):
        i = int(index/width)
        j = index % width
        img[i*h:(i+1)*h, j*w:(j+1)*w] = m[:, :, 0]
    return img

if __name__ == "__main__":
    # Load minist data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (np.asarray(X_train, dtype=np.float32) - 127.5)/127.5
    X_train = np.reshape(X_train, [-1, 28, 28, 1])

    z_dim = 100
    batch_size = 128
    lr = 0.0002
    n_epochs = 10

    sess = tf.Session()
    dcgan = DCGAN(z_dim=z_dim, output_dim=28, batch_size=128, c_dim=1)
    # The optimizers
    d_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(dcgan.d_loss, 
                                                var_list=dcgan.d_vars)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(dcgan.g_loss,
                                                var_list=dcgan.g_vars)
    sess.run(tf.global_variables_initializer())

    num_batches = int(len(X_train)/batch_size)
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        d_losses = 0
        g_losses = 0
        for idx in range(num_batches):
            # Train D
            z = np.random.uniform(-1, 1, size=[batch_size, z_dim])
            x = X_train[idx*batch_size:(idx+1)*batch_size]
            _, d_loss = sess.run([d_train_op, dcgan.d_loss], feed_dict={dcgan.z: z,
                                                        dcgan.x: x})
            d_losses += d_loss/num_batches
            # Train G
            z = np.random.uniform(-1, 1, size=[batch_size, z_dim])
            _, g_loss = sess.run([g_train_op, dcgan.g_loss], feed_dict={dcgan.z: z})
            g_losses += g_loss/num_batches
        
        print("\td_loss {0}, g_loss {1}".format(d_losses, g_losses))
        # Generate images
        z = np.random.uniform(-1, 1, size=[batch_size, z_dim])
        images = sess.run(dcgan.G, feed_dict={dcgan.z: z})
        img = combine_images(images)
        img = img*127.5 + 127.5
        Image.fromarray(img.astype(np.uint8)).save("epoch{0}_g_images.png".format(epoch))



        