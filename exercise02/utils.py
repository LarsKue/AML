
import numpy as np
import tensorflow as tf

# softmax = tf.nn.softmax


def relu(x):
    return np.maximum(x, 0)


def softmax(x, axis=None):
    ex = np.exp(x)
    sums = np.sum(ex, axis=axis)

    if axis is not None:
        shape = list(ex.shape)
        shape[axis] = -1

        return ex / sums.reshape(shape)

    # sums is scalar
    return ex / sums
