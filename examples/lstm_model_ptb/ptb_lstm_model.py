"""
A lstm model for PTB data
source: "https://github.com/xiaohu2015/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py"
2016/12/25
"""
import sys
import time
import numpy as np
import tensorflow as tf
from reader import ptb_raw_data, ptb_iterator

class LSTM_Model(object):
    """
    A LSTM class for language model on PTB data
    """
    def __init__(self, num_steps=20, vocab_size=10000, batch_size=20, hidden_size=1500, num_lstm_layers=2,
                keep_prob=0.5, max_grad_norm=5, is_training=True):
        """
        :param num_steps: int, the number of time steps (also sequence length)
        :param vocab_size: int, vocabulary size
        :param batch_size: int, batch size, you can also not give the batch_size
        :param hidden_size: int, the number of hidden units in lstm
        :param num_lstm_layers: int, the number of lstm layers
        :param keep_prob: float, the keep probability of dropout layer
        :param max_grad_norm: int, regularize gradients by norm
        :param is_training: bool, set True for training model, but False for test model
                            Note we construct three models with shared weight variables
        """
        # Keep all parameters
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.is_training = is_training

        # The input and output 
        self._x = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
        self._y = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
        if batch_size is None:
            batch_size = tf.shape(self._x)[0]
        
        # Construct lstm cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
        if is_training and keep_prob < 1.0:  # use dropout
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_lstm_layers)
        # The initial state
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # The embedding layer
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", shape=[vocab_size, hidden_size])
            inputs = tf.nn.embedding_lookup(embedding, self._x)  # [batch_size, num_steps, hidden_size]
        # Dropout
        if is_training and keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        
        # The lstm layer
        # Note: we compute the outputs by unrolling the lstm
        # you can also use tf.nn.rnn to simplify the following codes
        """
        inputs = tf.unstack(inputs, num_steps, axis=1) # list of [batch_size, hidden_size] from time step 0 to the end
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        """
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                # Note: we use shared variables (The cell creates the variables when it firstly starts running)
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        # Reshape
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])  # [num_steps*batch_size, hidden_size]
        # The softmax layer
        softmax_W = tf.get_variable("softmax_W", shape=[hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", shape=[vocab_size, ], dtype=tf.float32)
        logits = tf.matmul(output, softmax_W) + softmax_b
        # The loss
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._y, [-1,])], 
                                                        weights=[tf.ones(batch_size*num_steps)])
        self._cost = tf.reduce_sum(loss) / tf.to_float(batch_size)
        self._final_state = state

        if not is_training:
            return
        
        # The training operations
        # learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()  # The variables for training
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), max_grad_norm)
        self._train_op = tf.train.GradientDescentOptimizer(self._lr).apply_gradients(zip(grads, tvars))
        
    def assign_lr(self, sess, lr_value):
        if self.is_training:
            sess.run(tf.assign(self._lr, lr_value))
    
    
    @property
    def input(self):
        return self._x

    @property
    def target(self):
        return self._y
    
    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def final_state(self):
        return self._final_state
    
    @property
    def lr(self):
        return self._lr

    @property
    def cost(self):
        return self._cost
    
    @property
    def train_op(self):
        return self._train_op
    

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def model_run_epoch(sess, model, data, eval_op, verbose=True):
    """Runs the model for one epoch on the given data"""
    epoch_size = ((len(data)// model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.num_steps)):
        feed_dict = {model.input: x, model.target: y, model.initial_state: state}
        cost, state, _ = sess.run([model.cost, model.final_state, eval_op],
                                    feed_dict=feed_dict)
        costs += cost
        iters += model.num_steps
        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters), iters * m.batch_size / (time.time() - start_time)))
    return np.exp(costs/iters)

if __name__ == "__main__":
    # Load the PTB data
    data_path = sys.path[0] + "/data/"
    train_data, valid_data, test_data, vocab= ptb_raw_data(data_path=data_path)
    print(len(train_data), len(valid_data), len(test_data), vocab)
    # Configs
    config = LargeConfig()
    eval_config = LargeConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = LSTM_Model(num_steps=config.num_steps, vocab_size=config.vocab_size, batch_size=
                                config.batch_size, hidden_size=config.hidden_size, num_lstm_layers=config.num_layers,
                                keep_prob=config.keep_prob, max_grad_norm=config.max_grad_norm, is_training=True)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            val_model = LSTM_Model(num_steps=config.num_steps, vocab_size=config.vocab_size, batch_size=
                                config.batch_size, hidden_size=config.hidden_size, num_lstm_layers=config.num_layers,
                                keep_prob=config.keep_prob, max_grad_norm=config.max_grad_norm, is_training=False)
            test_model = LSTM_Model(num_steps=eval_config.num_steps, vocab_size=eval_config.vocab_size, batch_size=
                                eval_config.batch_size, hidden_size=eval_config.hidden_size, num_lstm_layers=eval_config.num_layers,
                                keep_prob=eval_config.keep_prob, max_grad_norm=eval_config.max_grad_norm, is_training=False)
        
        sess.run(tf.global_variables_initializer())
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            model.assign_lr(sess, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(model.lr)))
            train_perplexity = model_run_epoch(sess, model, train_data, model.train_op,
                                   verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = model_run_epoch(sess, val_model, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = model_run_epoch(session, test_model, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)



