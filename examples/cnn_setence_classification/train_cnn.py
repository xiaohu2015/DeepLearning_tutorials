"""
Test the TextRNN class 
2016/12/22
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

from data_helpers import load_data_and_labels, batch_iter
from text_cnn import TextCNN


# Load original data
path = sys.path[0]
pos_filename = path + "/data/rt-polarity.pos"
neg_filename = path + "/data/rt-polarity.neg"

X_data, y_data = load_data_and_labels(pos_filename, neg_filename)
max_document_length = max([len(sen.split(" ")) for sen in X_data])
print("Max_document_length:,", max_document_length)
# Create the vacabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# The idx data
x = np.array(list(vocab_processor.fit_transform(X_data)), dtype=np.float32)
y = np.array(y_data, dtype=np.int32)
vocabulary_size = len(vocab_processor.vocabulary_)
print("The size of vocabulary:", vocabulary_size)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1111)
print("X_train shape {0}, y_train shape {1}".format(X_train.shape, y_train.shape))
print("X_test shape {0}, y_test shape {1}".format(X_test.shape, y_test.shape))

# The parameters of RNN
seq_len = X_train.shape[1]
vocab_size = vocabulary_size
embedding_size = 128
filter_sizes = [2, 3, 4]
num_filters = 128
num_classes = y_train.shape[1]
l2_reg_lambda = 0.0

# Construct RNN model
text_rnn_model = TextCNN(seq_len=seq_len, vocab_size=vocab_size, embedding_size=embedding_size, filter_sizes=
                        filter_sizes, num_filters=num_filters, num_classes=num_classes)
loss = text_rnn_model.loss
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
accuracy = text_rnn_model.accuracy
# The parameters for training
batch_size = 64
training_epochs = 10
dispaly_every = 1
dropout_keep_prob = 0.5

batch_num = int(X_train.shape[0]/batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Starting training...")
for epoch in range(training_epochs):
    avg_cost = 0
    for batch in range(batch_num):
        _, cost = sess.run([train_op, loss], feed_dict={text_rnn_model.x: X_train[batch*batch_size:(batch+1)*batch_size],
                                    text_rnn_model.y: y_train[batch*batch_size:(batch+1)*batch_size],
                                    text_rnn_model.dropout_keep_prob:dropout_keep_prob})
        avg_cost += cost
    if epoch % dispaly_every == 0:
        cost, acc = sess.run([loss, accuracy], feed_dict={text_rnn_model.x: X_test,
                                    text_rnn_model.y: y_test,
                                    text_rnn_model.dropout_keep_prob: 1.0})
        print("\nEpoch {0} : loss {1}, accuracy {2}".format(epoch, cost, acc))

