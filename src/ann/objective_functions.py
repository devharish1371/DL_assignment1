import numpy as np
from .activations import Activation


class LossFunctions:

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        MSE Loss
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        """
        Gradient of MSE wrt predictions
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def cross_entropy(y_true, logits):
        """
        Cross entropy with internal softmax
        """
        probs = Activation.softmax(logits)
        eps = 1e-12
        log_probs = np.log(probs + eps)
        loss = -np.sum(y_true * log_probs) / y_true.shape[0]
        return loss

    @staticmethod
    def cross_entropy_derivative(y_true, logits):
        """
        Gradient of CE wrt logits
        """
        probs = Activation.softmax(logits)
        return (probs - y_true) / y_true.shape[0]


def get_loss(name):

    if name == "mean_squared_error":
        return (
            LossFunctions.mean_squared_error,
            LossFunctions.mse_derivative
        )

    if name == "cross_entropy":
        return (
            LossFunctions.cross_entropy,
            LossFunctions.cross_entropy_derivative
        )

    raise ValueError(f"Unsupported loss: {name}")