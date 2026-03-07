import numpy as np

from .neural_layer import NeuralLayer
from .objective_functions import get_loss
from .optimizers import get_optimizer
from .activations import Activation


class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation,
        loss,
        optimizer,
        learning_rate,
        weight_decay,
        weight_init
    ):

        self.layers = []

        prev = input_size

        # hidden layers
        for h in hidden_sizes:
            self.layers.append(
                NeuralLayer(prev, h, activation, weight_init)
            )
            prev = h

        # output layer (no activation applied here)
        self.layers.append(
            NeuralLayer(prev, output_size, activation="linear", weight_init=weight_init)
        )

        # loss function
        self.loss_fn, self.loss_grad = get_loss(loss)

        # optimizer
        self.optimizer = get_optimizer(
            optimizer,
            self.layers,
            learning_rate,
            weight_decay
        )

        self.loss_name = loss

    # ------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------

    def forward(self, X):

        out = X

        for i, layer in enumerate(self.layers):

            if i == len(self.layers) - 1:
                # last layer → return logits
                layer.input = out
                out = np.dot(out, layer.W) + layer.b
            else:
                out = layer.forward(out)

        return out

    # ------------------------------------------------
    # BACKWARD PASS
    # ------------------------------------------------

    def backward(self, y_true, logits):

        grad = self.loss_grad(y_true, logits)

        for i in reversed(range(len(self.layers))):

            layer = self.layers[i]

            if i == len(self.layers) - 1:

                layer.grad_W = np.dot(layer.input.T, grad)
                layer.grad_b = np.sum(grad, axis=0, keepdims=True)

                grad = np.dot(grad, layer.W.T)

            else:

                grad = layer.backward(grad)

    # ------------------------------------------------
    # TRAIN
    # ------------------------------------------------

    def train(self, X, y, epochs, batch_size):

        n = X.shape[0]

        for epoch in range(epochs):

            perm = np.random.permutation(n)

            X = X[perm]
            y = y[perm]

            epoch_loss = 0

            for i in range(0, n, batch_size):

                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                logits = self.forward(X_batch)

                loss = self.loss_fn(y_batch, logits)

                self.backward(y_batch, logits)

                self.optimizer.step()

                epoch_loss += loss

            print(f"Epoch {epoch+1}/{epochs} Loss {epoch_loss:.4f}")

    # ------------------------------------------------
    # PREDICT
    # ------------------------------------------------

    def predict(self, X):

        logits = self.forward(X)

        probs = Activation.softmax(logits)

        return np.argmax(probs, axis=1)

    # ------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------

    def save(self, path):

        weights = []

        for layer in self.layers:
            weights.append(layer.W)
            weights.append(layer.b)

        weights = np.array(weights, dtype=object)

        np.save(path, weights, allow_pickle=True)

    # ------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------

    def load(self, path):

        weights = np.load(path, allow_pickle=True).tolist()

        idx = 0

        for layer in self.layers:

            layer.W = weights[idx]
            layer.b = weights[idx + 1]

            idx += 2