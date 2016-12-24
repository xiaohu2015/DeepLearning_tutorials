"""
RNN model implemented by Numpy library
author: Ye Hu
2016/12/17
from " https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb "
"""
import sys
import operator
from datetime import datetime
import timeit
import numpy as np

from input_data_rnn import get_data

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNN_np(object):
    """A simple rnn class with numpy"""
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=-1):
        """
        """
        # keep
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the params
        bound = np.sqrt(1.0 / self.word_dim)
        self.U = np.random.uniform(-bound, bound, size=[self.word_dim, self.hidden_dim]) # input
        bound = np.sqrt(1.0 / self.hidden_dim)
        self.V = np.random.uniform(-bound, bound, size=[self.hidden_dim, self.word_dim])  # output
        self.W = np.random.uniform(-bound, bound, size=[self.hidden_dim, self.hidden_dim]) # old memeory

    def forward_propagation(self, x):
        """
        Forward propagation
        """
        sequence_dim = len(x)  # time steps, also sequence dim
        # keep the hidden states
        s = np.zeros((sequence_dim+1, self.hidden_dim))
        # the initial hidden of time step 0 (last)
        s[-1] = np.zeros((self.hidden_dim))
        # the output of each time step
        o = np.zeros((sequence_dim, self.word_dim))
        # for each time step
        for t in range(sequence_dim):
            # indeing with one-hot vector
            s[t] = np.tanh(self.U[x[t], :] + np.dot(s[t-1], self.W))
            o[t] = softmax(np.dot(s[t], self.V))

        return (o, s)

    def predict(self, x):
        """Give word with the highest probability """
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)  # for each time step

    def calculate_total_loss(self, xs, ys):
        """Cross entropy loss"""
        loss = 0
        # for each sequence
        for i in range(len(ys)):
            o, _ = self.forward_propagation(xs[i])
            correct_predictions = o[np.arange(len(ys[i])), ys[i]]
            loss += -1.0*np.sum(np.log(correct_predictions))
        return loss

    def calculate_loss(self, xs, ys):
        """"""
        # the training examples
        N = np.sum((len(e) for e in ys))
        return self.calculate_total_loss(xs, ys)/float(N)

    def bptt(self, x, y):
        """Compute the gradients by BPTT"""
        N = len(x) # time steps, also sequence dim
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # the initial gradients
        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdV = np.zeros(self.V.shape)
        # dL/do
        delta_o = o
        delta_o[np.arange(N), y] += -1.0
        # for each time step (also each output)
        for t in np.arange(N)[::-1]:
            # dL/dV
            dLdV += np.outer(s[t], delta_o[t])
            # dL/ds
            delta_t = np.dot(self.V, delta_o[t])*(1 - (s[t]**2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                #print("Backpropagation step t=%d bptt step=%d " % (t, bptt_step))
                dLdW += np.outer(s[bptt_step-1], delta_t)
                dLdU[x[bptt_step], :] += delta_t
                # Update delta for next time step
                delta_t = np.dot(self.W, delta_t)*(1 - (s[bptt_step-1]**2))
        return (dLdU, dLdV, dLdW)

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = model.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with shape %s." % (pname, str(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = model.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = model.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))

    def sgd(self, x, y, learning_rate):
        """Train the model with SGD"""
        # Compute the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Update the parameters
        self.U += -learning_rate * dLdU
        self.W += -learning_rate * dLdW
        self.V += -learning_rate * dLdV


def train_rnn_with_sgd(model, X_train, y_train, learning_rate=0.005, n_epochs=100,
                       evaluate_loss_after=5):
    """"""
    N = len(X_train)  # number of training examples
    losses = []
    num_examples_seen = 0
    for epoch in range(n_epochs):
        # if evaluate the loss
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" %
                  (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate *= 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # Training
        for i in range(N):
            model.sgd(X_train[i], y_train[i], learning_rate=learning_rate)
            num_examples_seen += 1


if __name__ == "__main__":

    np.random.seed(10)
    vocabulary_size = 8000
    X_train, y_train = get_data(vocabulary_size=vocabulary_size)

    model = RNN_np(word_dim=8000, bptt_truncate=-1)
    start_time = timeit.default_timer()
    train_rnn_with_sgd(model, X_train[:1000], y_train[:1000], n_epochs=10, evaluate_loss_after=1)
    end_time = timeit.default_timer()
    print("Time elapsed {0} seconds".format((end_time-start_time)))



