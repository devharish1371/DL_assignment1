import argparse
import json
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def accuracy(y_true, y_pred):

    y_true = np.argmax(y_true, axis=1)

    return np.mean(y_true == y_pred)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)

    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")

    parser.add_argument("-o", "--optimizer",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        default="adam")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument("-nhl", "--num_layers", type=int, default=2)

    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128, 128])

    parser.add_argument("-a", "--activation", choices=["sigmoid", "tanh", "relu"], default="relu")

    parser.add_argument("-wi", "--weight_init", choices=["random", "xavier"], default="xavier")

    return parser.parse_args()


def main():

    args = parse_args()

    # -------------------------
    # Load data
    # -------------------------

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    hidden_sizes = args.hidden_size

    # -------------------------
    # Build Model
    # -------------------------

    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=args.activation,
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        weight_init=args.weight_init
    )

    # -------------------------
    # Train
    # -------------------------

    model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # -------------------------
    # Validation
    # -------------------------

    preds = model.predict(X_val)

    val_acc = accuracy(y_val, preds)

    print("Validation Accuracy:", val_acc)

    # -------------------------
    # Save best model
    # -------------------------

    model.save("models/best_model.npy")

    config = vars(args)

    config["validation_accuracy"] = float(val_acc)

    with open("models/best_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("Model and config saved.")


if __name__ == "__main__":
    main()