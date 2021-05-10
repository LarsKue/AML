
import numpy as np
from sklearn import datasets

import utils

####################################


class ReLULayer(object):
    def __init__(self):
        self.input = None

    def forward(self, inp):
        # remember the input for later backpropagation
        self.input = inp
        # return the ReLU of the input
        relu = utils.relu(self.input)
        N = self.input.shape[0]
        m_l = self.input.shape[1]

        assert relu.shape == (N, m_l)
        return relu

    def backward(self, upstream_gradient):
        # compute the derivative of ReLU from upstream_gradient and the stored input
        grad = (self.input >= 0).astype(float)
        downstream_gradient = upstream_gradient * grad

        return downstream_gradient

    def update(self, learning_rate):
        pass  # ReLU is parameter-free


####################################


class OutputLayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.input = None

    def forward(self, inp):
        # remember the input for later backpropagation
        self.input = inp
        # return the softmax of the input
        activations = utils.softmax(inp, axis=-1)

        N = self.input.shape[0]
        m_l = self.n_classes
        assert activations.shape == (N, m_l)
        assert not np.any(np.isnan(activations))
        return activations

    def backward(self, predicted_posteriors, true_labels):
        # return the loss derivative with respect to the stored inputs
        # (use cross-entropy loss and the chain rule for softmax,
        # as derived in the lecture)

        one_hot = np.eye(self.n_classes)[true_labels]
        downstream_gradient = predicted_posteriors - one_hot
        return downstream_gradient

    def update(self, learning_rate):
        pass  # softmax is parameter-free


####################################


class LinearLayer(object):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # randomly initialize weights and intercepts
        self.B = np.random.normal(size=(n_inputs, n_outputs))
        self.b = np.random.normal(size=(1, n_outputs))

        self.input = None
        self.grad_b = None
        self.grad_B = None

    def forward(self, inp):
        assert not np.any(np.isnan(inp))
        # remember the input for later backpropagation
        self.input = inp
        # compute the scalar product of input and weights
        # (these are the preactivations for the subsequent non-linear layer)
        preactivations = np.dot(self.input, self.B) + self.b

        N = self.input.shape[0]
        m_l = self.n_outputs
        assert preactivations.shape == (N, m_l)
        return preactivations

    def backward(self, upstream_gradient):
        # compute the derivative of the weights from
        # upstream_gradient and the stored input
        self.grad_b = np.mean(upstream_gradient, axis=0)
        self.grad_B = np.dot(self.input.T, upstream_gradient)
        # compute the downstream gradient to be passed to the preceding layer
        downstream_gradient = np.dot(upstream_gradient, self.B.T)

        return downstream_gradient

    def update(self, learning_rate):
        # update the weights by batch gradient descent
        self.B = self.B - learning_rate * self.grad_B
        self.b = self.b - learning_rate * self.grad_b


####################################


class MLP(object):
    def __init__(self, n_features, layer_sizes):
        # construct a multi-layer perceptron
        # with ReLU activation in the hidden layers and softmax output
        # (i.e. it predicts the posterior probability of a classification problem)
        #
        # n_features: number of inputs
        # len(layer_size): number of layers
        # layer_size[k]: number of neeurons in layer k
        # (specifically: layer_sizes[-1] is the number of classes)
        self.n_layers = len(layer_sizes)
        self.layers = []

        # create interior layers (linear + ReLU)
        n_in = n_features
        for n_out in layer_sizes[:-1]:
            self.layers.append(LinearLayer(n_in, n_out))
            self.layers.append(ReLULayer())
            n_in = n_out

        # create last linear layer + output layer
        n_out = layer_sizes[-1]
        self.layers.append(LinearLayer(n_in, n_out))
        self.layers.append(OutputLayer(n_out))

    def forward(self, X):
        # X is a mini-batch of instances
        batch_size = X.shape[0]
        # flatten the other dimensions of X (in case instances are images)
        X = X.reshape(batch_size, -1)

        # compute the forward pass
        # (implicitly stores internal activations for later backpropagation)
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, predicted_posteriors, true_classes):
        # perform backpropagation w.r.t. the prediction for the latest mini-batch X
        upstream_gradient = self.layers[-1].backward(predicted_posteriors, true_classes)

        for layer in reversed(self.layers[:-1]):
            upstream_gradient = layer.backward(upstream_gradient)

        return upstream_gradient

    def update(self, X, Y, learning_rate):
        posteriors = self.forward(X)
        self.backward(posteriors, Y)
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x, y, n_epochs, batch_size, learning_rate):
        N = len(x)
        n_batches = N // batch_size

        for i in range(n_epochs):
            if i % 100 == 0:
                print("Epoch", i)
            # reorder data for every epoch
            # (i.e. sample mini-batches without replacement)
            permutation = np.random.permutation(N)

            for batch in range(n_batches):
                # create mini-batch
                start = batch * batch_size
                x_batch = x[permutation[start:start + batch_size]]
                y_batch = y[permutation[start:start + batch_size]]

                # perform one forward and backward pass and update network parameters
                self.update(x_batch, y_batch, learning_rate)


####################################


if __name__ == "__main__":
    # set training/test set size
    N = 2000

    # create training and test data
    X_train, Y_train = datasets.make_moons(N, noise=0.05)

    import matplotlib.pyplot as plt

    plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, )
    plt.show()

    X_test, Y_test = datasets.make_moons(N, noise=0.05)
    n_features = 2
    n_classes = 2

    # standardize features to be in [-1, 1]
    offset = X_train.min(axis=0)
    scaling = X_train.max(axis=0) - offset
    X_train = ((X_train - offset) / scaling - 0.5) * 2.0
    X_test = ((X_test - offset) / scaling - 0.5) * 2.0

    # set hyperparameters (play with these!)
    layer_sizes = [32, 16, n_classes]
    n_epochs = 10000
    batch_size = 200
    learning_rate = 0.003

    # create network
    network = MLP(n_features, layer_sizes)

    # train
    network.train(X_train, Y_train, n_epochs, batch_size, learning_rate)

    # test
    predicted_posteriors = network.forward(X_test)
    # determine class predictions from posteriors by winner-takes-all rule
    predicted_classes = np.argmax(predicted_posteriors, axis=-1)
    # compute and output the error rate of predicted_classes

    incorrect = predicted_classes != Y_test

    error_rate = incorrect.sum() / len(incorrect)
    print("error rate:", error_rate)

    W = 100
    H = 100
    left = np.min(X_train[:, 0])
    right = np.max(X_train[:, 0])
    bottom = np.min(X_train[:, 1])
    top = np.max(X_train[:, 1])
    x = np.linspace(left, right, W)
    y = np.linspace(bottom, top, H)

    xx, yy = np.meshgrid(x, y)

    X = np.stack([xx.flat, yy.flat]).T

    predictions = network.forward(X)
    predictions = np.asarray(predictions)
    classes = np.argmax(predictions, axis=-1)

    classes = classes.reshape((W, H))
    predictions = predictions.reshape((W, H, 2))
    predictions = np.max(predictions, axis=-1)

    plt.imshow(classes, extent=(left, right, bottom, top), alpha=predictions, origin="lower")
    plt.show()


