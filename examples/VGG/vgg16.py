"""
The VGG16 Model for cifra10 dataset
author: Ye Hu
2016/12/26
"""
import sys
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, toimage
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
from imagenet_classes import class_names

VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16(object):
    """
    The VGG16 model for image classification
    """
    def __init__(self, vgg16_npy_path=None, trainable=True):
        """
        :param vgg16_npy_path: string, vgg16_npz path
        :param trainable: bool, construct a trainable model if True
        """
        # The pretained data
        if vgg16_npy_path is None:
            self._data_dict = None
        else:
            self._data_dict = np.load(vgg16_npy_path, encoding="latin1").item()
        self.trainable = trainable
        # Keep all trainable parameters
        self._var_dict = {}
        self.__bulid__()

    def __bulid__(self):
        """
        The inner method to build VGG16 model
        """
        # input and output
        self._x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        self._y = tf.placeholder(tf.int64, shape=[None,])
        # Data preprocessiing
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3])
        x = self._x - mean
        self._train_mode = tf.placeholder(tf.bool)   # use training model is True, otherwise test model
        # construct model
        conv1_1 = self._conv_layer(x, 3, 64, "conv1_1")
        conv1_2 = self._conv_layer(conv1_1, 64, 64, "conv1_2")
        pool1 = self._max_pool(conv1_2, "pool1")

        conv2_1 = self._conv_layer(pool1, 64, 128, "conv2_1")
        conv2_2 = self._conv_layer(conv2_1, 128, 128, "conv2_2")
        pool2 = self._max_pool(conv2_2, "pool2")

        conv3_1 = self._conv_layer(pool2, 128, 256, "conv3_1")
        conv3_2 = self._conv_layer(conv3_1, 256, 256, "conv3_2")
        conv3_3 = self._conv_layer(conv3_2, 256, 256, "conv3_3")
        pool3 = self._max_pool(conv3_3, "pool3")

        conv4_1 = self._conv_layer(pool3, 256, 512, "conv4_1")
        conv4_2 = self._conv_layer(conv4_1, 512, 512, "conv4_2")
        conv4_3 = self._conv_layer(conv4_2, 512, 512, "conv4_3")
        pool4 = self._max_pool(conv4_3, "pool4")

        conv5_1 = self._conv_layer(pool4, 512, 512, "conv5_1")
        conv5_2 = self._conv_layer(conv5_1, 512, 512, "conv5_2")
        conv5_3 = self._conv_layer(conv5_2, 512, 512, "conv5_3")
        pool5 = self._max_pool(conv5_3, "pool5")

        # n_in = ((224 / (2**5)) ** 2) * 512
        fc6 = self._fc_layer(pool5, 25088, 4096, "fc6", act=tf.nn.relu, reshaped=False)
        # Use train_mode to control
        fc6 = tf.cond(self._train_mode, lambda : tf.nn.dropout(fc6, 0.5), lambda: fc6)
        fc7 = self._fc_layer(fc6, 4096, 4096, "fc7", act=tf.nn.relu)
        fc7 = tf.cond(self._train_mode, lambda : tf.nn.dropout(fc7, 0.5), lambda: fc7)
        fc8 = self._fc_layer(fc7, 4096, 1000, "fc8", act=tf.identity)

        self._prob = tf.nn.softmax(fc8, name="prob")

        if self.trainable:
            self._cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(fc8, self._y))
            correct_pred = tf.equal(self._y, tf.argmax(self._prob, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        else:
            self._cost = None
            self._accuracy = None

    def _conv_layer(self, inpt, in_channels, out_channels, name):
        """
        Create conv layer
        """
        with tf.variable_scope(name):
            filters, biases = self._get_conv_var(3, in_channels, out_channels, name)
            conv_output = tf.nn.conv2d(inpt, filters, strides=[1, 1, 1, 1], padding="SAME")
            conv_output = tf.nn.bias_add(conv_output, biases)
            conv_output = tf.nn.relu(conv_output)
        return conv_output
    
    def _fc_layer(self, inpt, n_in, n_out, name, act=tf.nn.relu, reshaped=True):
        """Create fully connected layer"""
        if not reshaped:
            inpt = tf.reshape(inpt, shape=[-1, n_in])
        with tf.variable_scope(name):
            weights, biases = self._get_fc_var(n_in, n_out, name)
            output = tf.matmul(inpt, weights) + biases
        return act(output)

    
    def _avg_pool(self, inpt, name):
        return tf.nn.avg_pool(inpt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                name=name)
    
    def _max_pool(self, inpt, name):
        return tf.nn.max_pool(inpt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                name=name)
    
    def _get_fc_var(self, n_in, n_out, name):
        """Get the weights and biases of fully connected layer"""
        if self.trainable:
            init_weights = tf.truncated_normal([n_in, n_out], 0.0, 0.001)
            init_biases = tf.truncated_normal([n_out,], 0.0, 0.001)
        else:
            init_weights = None
            init_biases = None
        weights = self._get_var(init_weights, name, 0, name+"_weights")
        biases = self._get_var(init_biases, name, 1, name+"_biases")
        return weights, biases

    def _get_conv_var(self, filter_size, in_channels, out_channels, name):
        """
        Get the filter and bias of conv layer 
        """
        if self.trainable:
            initial_value_filter = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
            initial_value_bias = tf.truncated_normal([out_channels,], 0.0, 0.001)
        else:
            initial_value_filter = None
            initial_value_bias = None
        filters = self._get_var(initial_value_filter, name, 0, name+"_filters")
        biases = self._get_var(initial_value_bias, name, 1, name+"_biases")
        return filters, biases

    
    def _get_var(self, initial_value, name, idx, var_name):
        """
        Use this method to construct variable parameters
        """
        if self._data_dict is not None:
            value = self._data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, dtype=tf.float32, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name="var_name")
        # Save
        self._var_dict[(name, idx)] = var
        return var
    
    def get_train_op(self, lr=0.01):
        if not self.trainable:
            return
        return tf.train.GradientDescentOptimizer(lr).minimize(self.cost, 
                                                    var_list=list(self._var_dict.values()))
    
    def save_npy(self, sess, npy_path="./vgg16_save.npy"):
        data_dict = {}
        for (name, idx), var in self._var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        
        np.save(npy_path, data_dict)
        print("File saved", npy_path)

    @property
    def input(self):
        return self._x
    
    @property
    def target(self):
        return self._y
    
    @property
    def train_mode(self):
        return self._train_mode
    
    @property
    def accuracy(self):
        return self._accuracy

    @property
    def cost(self):
        return self._cost
    
    @property
    def prob(self):
        return self._prob

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    # assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img



def test_not_trainable_vgg16():
    path = sys.path[0]
    img1 = load_image(path+"/tiger.jpeg")*255.0
    img2 = load_image(path+"/puzzle.jpeg")*255.0
    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))

    batch = np.concatenate((batch1, batch2), 0)

    with tf.Graph().as_default(), tf.Session() as sess:
        vgg = VGG16(path+"/vgg16.npy", trainable=False)
        probs = sess.run(vgg.prob, feed_dict={vgg.input: batch, vgg.train_mode: False})
        for i, prob in enumerate([probs[0], probs[1]]):
            preds = (np.argsort(prob)[::-1])[0:5]
            print("The" +str(i) + " image:")
            for p in preds:
                print("\t", p, class_names[p], prob[p])

if __name__ == "__main__":
    path = sys.path[0]
    img1 = load_image(path+"/tiger.jpeg")*255.0
    img2 = load_image(path+"/puzzle.jpeg")*255.0
    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))
    x = np.concatenate((batch1, batch2), 0)
    y = np.array([292, 611], dtype=np.int64)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            vgg = VGG16(path+"/vgg16.npy", trainable=True)
            sess.run(tf.global_variables_initializer())

            train_op = vgg.get_train_op(lr=0.0001)
            _, cost = sess.run([train_op, vgg.cost], feed_dict={vgg.input: x,
                                                            vgg.target: y, vgg.train_mode:True})
            accuracy = sess.run(vgg.accuracy, feed_dict={vgg.input: x,
                                                        vgg.target: y, vgg.train_mode:False})
            print(cost, accuracy)



