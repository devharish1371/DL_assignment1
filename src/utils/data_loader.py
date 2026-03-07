import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def one_hot(y, num_classes=10):

    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1

    return y_onehot


def preprocess(X):

    # normalize
    X = X.astype("float32") / 255.0

    # flatten images (28x28 -> 784)
    X = X.reshape(X.shape[0], -1)

    return X


def load_data(dataset="mnist", val_split=0.1):

    if dataset == "mnist":

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset == "fashion_mnist":

        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:

        raise ValueError("Dataset must be mnist or fashion_mnist")

    # preprocess
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    # train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_split,
        random_state=42,
        stratify=y_train
    )

    # one-hot labels
    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
    y_test = one_hot(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
