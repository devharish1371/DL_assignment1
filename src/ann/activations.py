import numpy as np


class Activation:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        """
        Numerically stable softmax
        """
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)


def get_activation(name):

    if name == "sigmoid":
        return Activation.sigmoid, Activation.sigmoid_derivative

    if name == "tanh":
        return Activation.tanh, Activation.tanh_derivative

    if name == "relu":
        return Activation.relu, Activation.relu_derivative
    if name == "linear":
        return Activation.linear, Activation.linear_derivative

    raise ValueError(f"Unsupported activation: {name}")