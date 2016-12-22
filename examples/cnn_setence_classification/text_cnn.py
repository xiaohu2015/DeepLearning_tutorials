"""
A CNN model for sentence classification
source: 'https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py'
2016/12/21
"""
import numpy as np
import tensorflow as tf

class TextCNN(object):
    """
    A CNN class for sentence classification
    The model includes an embedding layer, a convolutional layer, a max-pooling layer and
    a softmax layer as the output.
    """
    def __init__(self, seq_len, vocab_size, embedding_size, filter_sizes, num_filters,
                    num_classes=2, l2_reg_lambda=0.0):
        """
        :param seq_len: int, the sequence length (i.e. the length of the sentences, 
                        keep all length same by zero-padding)
        :param vocab_size: int, the size of vocabulary to define the embedding layer
        :param embedding_size: int, the dimensionality of the embeddings (word vector). 
        :param filter_sizes: list or tuple, The number of words we want our convolutional filters to cover. 
                            For example, [3, 4, 5] means that we will have filters that slide over 3, 4 
                            and 5 words respectively
        :param num_filters: int, the number of each filter with different filter_size, hence, we have a total of
                            len(filter_sizes) * num_filters filters
        :param num_classes: the number of classes we want to predict in the output layer, default 2
        :param l2_reg_lambda: float, the ratio of L2 loss
        """
        # keep track of all parameters
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_szie = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        # Define the input and output
        self.x = tf.placeholder(tf.int32, shape=[None, seq_len], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")
        # The dropout probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Compute the L2 regularization loss
        L2_loss = tf.constant(0.0)   # initial value 0.0

        # The Embedding layer
        with tf.device("/cpu:0"):   # embedding implementation not support GPU
            with tf.name_scope("embedding"):
                # The embedding matrix
                self.W_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                            dtype=tf.float32, name="W_embedding")
                # The embedding results   
                self.embedded_chars = tf.nn.embedding_lookup(self.W_embedding, self.x)   #[None, seq_len, embedding_size]
                # Expand it to use conv2D operation
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, axis=-1) # [None, seq_len, embedding_size, 1]
        
        # The convolution and maxpool layer
        pooled_outputs = []
        self.Ws_conv = []
        self.bs_conv = []
        # For each filter
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv_maxpool_{0}".format(filter_size)):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # Conv params
                W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                            dtype=tf.float32, name="W_conv")
                self.Ws_conv.append(W_conv)
                b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters,]), dtype=tf.float32,
                                                name="b_conv")
                self.bs_conv.append(b_conv)
                # conv result
                conv_output = tf.nn.conv2d(self.embedded_chars_expanded, W_conv, strides=[1, 1, 1, 1],
                                            padding="VALID", name="conv")   # [None, seq_len-filter_size+1, 1, num_filters]
                # use relu as activation
                conv_h = tf.nn.relu(tf.nn.bias_add(conv_output, b_conv), name="relu")
                # Use max-pooling
                pool_output = tf.nn.max_pool(conv_h, ksize=[1, seq_len-filter_size+1, 1, 1],
                                            strides=[1, 1, 1, 1], padding="VALID", name="max_pooling")
                pooled_outputs.append(pool_output)   # [None, 1, 1, num_filters]
        # Combine all pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)  # [None, 1, 1, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total]) # [None, num_filters_total]

        # The dropout layer
        with tf.name_scope("dropout"):
            self.h_dropout = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob, name="dropout")
        
        # The output layer (softmax)
        with tf.name_scope("output"):
            self.W_fullyconn = tf.get_variable("W_fullyconn", shape=[num_filters_total, num_classes],
                                                initializer=tf.contrib.layers.xavier_initializer())
            self.b_fullyconn = tf.Variable(tf.constant(0.1, shape=[num_classes,]), dtype=tf.float32, name="b_fullyconn")
            # L2_loss
            L2_loss += tf.nn.l2_loss(self.W_fullyconn)
            self.scores = tf.nn.xw_plus_b(self.h_dropout, self.W_fullyconn, self.b_fullyconn, name="scores")
            self.preds = tf.argmax(self.scores, axis=1, name="preds")
        
        # The loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y)
            self.loss = tf.reduce_mean(losses) + L2_loss * l2_reg_lambda
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_preds = tf.equal(self.preds, tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    
    def save_weights(self, sess, filename, name="TextRNN"):
        """"""
        save_dicts = {name+"_W_embedding": self.W_embedding}
        for i in range(len(self.Ws_conv)):
            save_dicts.update({name+"_W_conv_"+str(i): self.Ws_conv[i],
                                name+"_b_conv_"+str(i): self.bs_conv[i]})
        save_dicts.update({name+"_W_fullyconn": self.W_fullyconn,
                            name+"_b_fullyconn": self.b_fullyconn})
        saver = tf.train.Saver(save_dicts)
        return saver.save(sess, filename)
    
    def load_weights(self, sess, filename, name="TextRNN"):
        """"""
        save_dicts = {name+"_W_embedding": self.W_embedding}
        for i in range(len(self.Ws_conv)):
            save_dicts.update({name+"_W_conv_"+str(i): self.Ws_conv[i],
                                name+"_b_conv_"+str(i): self.bs_conv[i]})
        save_dicts.update({name+"_W_fullyconn": self.W_fullyconn,
                            name+"_b_fullyconn": self.b_fullyconn})
        saver = tf.train.Saver(save_dicts)
        saver.restore(sess)