import argparse
import json
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="models/best_model.npy")
    parser.add_argument("--config_path", default="models/best_config.json")

    return parser.parse_args()


def main():

    args = parse_args()

    # -------------------------
    # Load config
    # -------------------------

    with open(args.config_path, "r") as f:
        config = json.load(f)

    dataset = config["dataset"]

    # -------------------------
    # Load dataset
    # -------------------------

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset)

    input_size = X_test.shape[1]
    output_size = y_test.shape[1]

    hidden_sizes = config["hidden_size"]

    # -------------------------
    # Rebuild model
    # -------------------------

    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=config["activation"],
        loss=config["loss"],
        optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        weight_init=config["weight_init"]
    )

    # -------------------------
    # Load weights
    # -------------------------

    model.load(args.model_path)

    # -------------------------
    # Predict
    # -------------------------

    preds = model.predict(X_test)

    y_true = np.argmax(y_test, axis=1)

    # -------------------------
    # Metrics
    # -------------------------

    acc = accuracy_score(y_true, preds)

    precision = precision_score(y_true, preds, average="macro")
    recall = recall_score(y_true, preds, average="macro")
    f1 = f1_score(y_true, preds, average="macro")

    print("Test Accuracy :", acc)
    print("Precision     :", precision)
    print("Recall        :", recall)
    print("F1 Score      :", f1)


if __name__ == "__main__":
    main()