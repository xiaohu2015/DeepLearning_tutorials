"""
Deep Residual Learning
source: 'https://github.com/wenxinxu/resnet_in_tensorflow'
2016/12/27
"""
import sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.files import load_cifar10_dataset

def tensor_summary(tensor):
    """Add histogram and scalar summary of the tensor"""
    tensor_name = tensor.op.name
    tf.summary.histogram(tensor_name+"/activations", tensor)
    tf.summary.scalar(tensor_name+"/sparsity", tf.nn.zero_fraction(tensor))

class ResnetConfig(object):
    """
    The default hyper parameters config
    """
    # The batch normalization variance epsilon
    bn_var_epsilon = 0.001
    # weight decay for regularization in fully connected layer
    fc_weight_decay = 0.0002
    # weight decay for regularization in convolutional layer
    conv_weight_decay = 0.0002
    # Default initializer
    initializer = tf.contrib.layers.xavier_initializer

class Resnet(object):
    """
    A deep residual learning network class
    """
    def __init__(self, input_tensor, n, is_training=True, config=ResnetConfig()):
        """
        :param input_tensor: 4-D input tensor
        :param n: int, the number of residual blocks
        :param is_training: bool, create new variables if True, otherwise, reuse the variables
        :param config: The hyper parameters config class
        """
        self.input = input_tensor
        self.n = n
        self.is_training = is_training
        self.config = ResnetConfig()
        self.__build__model__()
    
    def __build__model__(self):
        """
        This function will bulid the resnet model.
        """
        if self.is_training:
            reuse = False
        else:
            reuse = True
        # Keep track of all layers
        layers = []
        # The first layer
        with tf.variable_scope("conv0", reuse=reuse):
            conv0 = self._conv_bn_relu_layer(self.input, 16, 3, strides=1)
            tensor_summary(conv0)
            layers.append(conv0)
        
        # The first residual blocks
        for i in range(self.n):
            with tf.variable_scope("conv1_%d" % i, reuse=reuse):
                if i == 0:
                    conv1 = self._residual_block(layers[-1], 16, is_first_block=True)
                else:
                    conv1 = self._residual_block(layers[-1], 16)   #[None, 32, 32, 16]
                tensor_summary(conv1)
                layers.append(conv1)
        
        # The second residual blocks
        for i in range(self.n):
            with tf.variable_scope("conv2_%d" % i, reuse=reuse):
                conv2 = self._residual_block(layers[-1], 32) #[None, 16, 16, 32]
                tensor_summary(conv2)
                layers.append(conv2)
        
        # The 3th residual blocks
        for i in range(self.n):
            with tf.variable_scope("conv3_%d" % i, reuse=reuse):
                conv3 = self._residual_block(layers[-1], 64)  #[None, 8, 8, 64]
                tensor_summary(conv3)
                layers.append(conv3)
        
        # The fully connected layer
        with tf.variable_scope("fc", reuse=reuse):
            in_channels = layers[-1].get_shape().as_list()[-1]
            bn = self._batch_normalization_layer(layers[-1], in_channels)
            relu = tf.nn.relu(bn)
            global_pool = tf.reduce_mean(relu, axis=[1, 2])
            output = self._fc_layer(global_pool, 10)
            layers.append(output)
        
        self._output = output
        self._prediction = tf.cast(tf.argmax(tf.nn.softmax(output), axis=1), tf.int32)


    def _get_variable(self, name, shape, initializer=None, is_fc_layer=False):
        """
        Create the variable of layers
        :param name: string, variable name
        :param shape: list or tuple, the shape of variable
        :param initializer: default initializer
        :param is_fc_layer: use different regularization for different layers
        """
        if is_fc_layer:
            scale = self.config.fc_weight_decay
        else:
            scale = self.config.conv_weight_decay

        if initializer is None:
            initializer = self.config.initializer()

        var = tf.get_variable(name, shape, initializer=initializer, 
                                regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
        return var

    def _batch_normalization_layer(self, input_tensor, depth_dim=None):
        """
        The batch normalization layer
        :param input_tensor: 4-D tensor
        :param depth_dim: the last dimension of the input_tensor
        :return: the normalized tensor with the same shape of input tensor
        """
        if depth_dim is None:
            depth_dim = input_tensor.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2], keep_dims=False)
        # Define variables of batch normalization
        beta = tf.get_variable("beta", [depth_dim,], dtype=tf.float32, 
                            initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable("gamma", [depth_dim,], dtype=tf.float32,
                            initializer=tf.constant_initializer(1.0))
        output_tensor = tf.nn.batch_normalization(input_tensor, mean, variance, 
                                                beta, gamma, self.config.bn_var_epsilon)
        return output_tensor
    
    def _fc_layer(self, input_tensor, n_out, n_in=None, activation=tf.identity):
        """
        The fully connected layer
        :param input_tensor: 2-D tensor 
        :param n_in: int, the number of input units
        :param n_out: int, the number of output units
        :param activation: activation function, default you use identity activation
        """
        if n_in is None:
            n_in = input_tensor.get_shape().as_list()[-1]
        weights = self._get_variable("fc_weight", [n_in, n_out], initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                                    is_fc_layer=True)
        biases = self._get_variable("fc_bias", [n_out,], initializer=tf.zeros_initializer, is_fc_layer=True)
        wx_b = tf.matmul(input_tensor, weights) + biases
        return activation(wx_b)
    
    def _conv_bn_relu_layer(self, input_tensor, nb_filter, filter_size, strides=1):
        """
        This function implements a sequence layes with convolution, batch normalize, and relu
        :param input_tensor: 4-D tensor
        :param nb_filter: int, the number of filters
        :param filter_size: int,  the size of filters
        :param strides: int, the strides of convolution operation
        """
        in_channels = input_tensor.get_shape().as_list()[-1]
        filter = self._get_variable("conv", shape=[filter_size, filter_size, in_channels, nb_filter])
        conv = tf.nn.conv2d(input_tensor, filter, strides=[1, strides, strides, 1], padding="SAME")
        bn = self._batch_normalization_layer(conv, nb_filter)
        return tf.nn.relu(bn)
    
    def _bn_relu_conv_layer(self, input_tensor, nb_filter, filter_size, strides=1):
        """
        This function implements a sequence layers with batch normalize, relu and convolution
        :param input_tensor: 4-D tensor
        :param nb_filter: int, the number of filters
        :param filter_size: int,  the size of filters
        :param strides: int, the strides of convolution operation
        """
        in_channels = input_tensor.get_shape().as_list()[-1]
        bn = self._batch_normalization_layer(input_tensor, in_channels)
        relu = tf.nn.relu(bn)
        filter = self._get_variable("conv", shape=[filter_size, filter_size, in_channels, nb_filter])
        conv = tf.nn.conv2d(relu, filter, strides=[1, strides, strides, 1], padding="SAME")
        return conv

    def _residual_block(self, input_tensor, out_channels, is_first_block=False):
        """
        A residual block of resnet
        :param input_tensor: 4-D tensor
        :param out_channels: int, the number of output channels
        :param is_first_block: bool, if it is first residual block of Resnet
        """
        in_channels = input_tensor.get_shape().as_list()[-1]
        # If the map feature size reduces, you should use strides =2, also
        # you must average pool over the input, and pad the input at last dimension
        if in_channels*2 == out_channels:
            strides = 2
        elif in_channels == out_channels:
            strides = 1
        else:
            raise ValueError("There is mismatch betwwen input and output channels")
        
        # The first conv layer in the first residual block only implments the conv operation
        with tf.variable_scope("block_conv1"):
            if is_first_block:
                filter = self._get_variable("conv", shape=[3, 3, in_channels, out_channels])
                conv1 = tf.nn.conv2d(input_tensor, filter, strides=[1, 1, 1, 1], padding="SAME")
            else:
                conv1 = self._bn_relu_conv_layer(input_tensor, out_channels, 3, strides=strides)

        # The second conv layer
        with tf.variable_scope("block_conv2"):
            conv2 = self._bn_relu_conv_layer(conv1, out_channels, 3, strides=1)
        
        # Deal with input
        if strides > 1:
            pooled_input = tf.nn.avg_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding="VALID")
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [in_channels//2, in_channels//2]])
        
        else:
            padded_input = input_tensor
        
        return conv2 + padded_input
        
    @property
    def prediction(self):
        return self._prediction
    
    def get_cost(self, y):
        """
        Get the cost for training
        :param y: the target tensor (1-D, [None])
        """
        assert y.get_shape().as_list()[0] == self.input.get_shape().as_list()[0]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self._output, y)
        return tf.reduce_mean(cross_entropy)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    train_dir = sys.path[0] + "/train_dir"
    input_tensor = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.int32, shape=[None,])
    resent = Resnet(input_tensor, 2, is_training=True)
    cost = resent.get_cost(y)
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(resent.prediction, y), tf.float32))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print("Start training...")
    n_epochs = 10
    for epoch in range(n_epochs):
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size=128, shuffle=True):
            sess.run(train_op, feed_dict={resent.input: X_train_a, y: y_train_a})
        n_batchs = 0
        acc = 0
        for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, 128, shuffle=True):
            acc += sess.run(accuracy, feed_dict={resent.input: X_test_a, y: y_test_a})
            n_batchs += 1
        print("Epoch {0}, test accuracy {1}".format(epoch, acc/n_batchs))

