import wandb
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def accuracy(y_true, y_pred):

    y_true = np.argmax(y_true, axis=1)

    return np.mean(y_true == y_pred)


def train():

    wandb.init()

    config = wandb.config

    # ----------------------
    # Load MNIST
    # ----------------------

    X_train, y_train, X_val, y_val, X_test, y_test = load_data("mnist")

    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=config.hidden_sizes,
        output_size=10,
        activation=config.activation,
        loss=config.loss,
        optimizer=config.optimizer,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        weight_init=config.weight_init
    )

    epochs = config.epochs
    batch_size = config.batch_size

    n = X_train.shape[0]

    for epoch in range(epochs):

        perm = np.random.permutation(n)

        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0

        for i in range(0, n, batch_size):

            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            logits = model.forward(X_batch)

            loss = model.loss_fn(y_batch, logits)

            model.backward(y_batch, logits)

            model.optimizer.step()

            epoch_loss += loss

        preds = model.predict(X_val)

        val_acc = accuracy(y_val, preds)

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_accuracy": val_acc
        })

    preds = model.predict(X_test)

    test_acc = accuracy(y_test, preds)

    wandb.log({"test_accuracy": test_acc})

    model.save("best_model.npy")