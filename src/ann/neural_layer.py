import numpy as np
from .activations import get_activation


class NeuralLayer:
    def __init__(self, input_size, output_size, activation, weight_init="random"):

        self.input_size = input_size
        self.output_size = output_size

        # activation functions
        self.activation, self.activation_derivative = get_activation(activation)

        # weight initialization
        if weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            raise ValueError("Unsupported weight initialization")

        self.b = np.zeros((1, output_size))

        # gradient placeholders (required by assignment)
        self.grad_W = None
        self.grad_b = None

        # cache for backprop
        self.input = None
        self.z = None

    def forward(self, x):
        """
        Forward propagation
        """
        self.input = x

        self.z = np.dot(x, self.W) + self.b
        a = self.activation(self.z)

        return a

    def backward(self, grad_output):
        """
        Backward propagation

        grad_output = dL/dA
        """

        # dA/dZ
        activation_grad = self.activation_derivative(self.z)

        delta = grad_output * activation_grad

        # gradients
        self.grad_W = np.dot(self.input.T, delta)
        self.grad_b = np.sum(delta, axis=0, keepdims=True)

        # gradient to previous layer
        grad_input = np.dot(delta, self.W.T)

        return grad_input