"""
wandb_experiments.py

Script to run all Section 2.xx experiments for DA6401 Assignment 1 and
log everything to Weights & Biases.

You can:
  - Run all key experiments (except the long 100-run sweep) with:

        python -m src.wandb_experiments --run-all

  - Run a specific section only, e.g.:

        python -m src.wandb_experiments --section 2.3

The notebook version can simply import and call these helpers to
reproduce experiments cell-by-cell.
"""

import argparse
import os
import types
import numpy as np
import wandb

from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_data
from sklearn.metrics import confusion_matrix


WANDB_PROJECT = os.environ.get("DA6401_WANDB_PROJECT", "da6401-a1")
WANDB_ENTITY = None  # set to your username/team if needed


def make_cfg(
    dataset="mnist",
    epochs=10,
    batch_size=64,
    loss="cross_entropy",
    optimizer="adam",
    learning_rate=0.001,
    weight_decay=0.0,
    num_layers=3,
    hidden_size=(128, 128, 128),
    activation="relu",
    weight_init="xavier",
):
    """Create a simple namespace config compatible with NeuralNetwork."""
    if isinstance(hidden_size, int):
        hidden_size = [hidden_size] * num_layers

    cfg = types.SimpleNamespace(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        loss=loss,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation,
        weight_init=weight_init,
        input_size=784,
        output_size=10,
    )
    return cfg


def build_model_and_data(cfg):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg.dataset)
    cfg.input_size = X_train.shape[1]
    cfg.output_size = y_train.shape[1]
    model = NeuralNetwork(cfg)
    return model, (X_train, y_train, X_val, y_val, X_test, y_test)


def run_one_training(cfg, run_name=None, group=None, tags=None, log_gradients=False):
    """Generic training loop with wandb logging."""
    model, (X_train, y_train, X_val, y_val, X_test, y_test) = build_model_and_data(cfg)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=cfg.__dict__,
        name=run_name,
        group=group,
        tags=tags,
        reinit=True,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    n = X_train.shape[0]

    for epoch in range(1, cfg.epochs + 1):
        perm = np.random.permutation(n)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        epoch_correct = 0
        num_batches = int(np.ceil(n / cfg.batch_size))

        for b in range(num_batches):
            start = b * cfg.batch_size
            end = min(start + cfg.batch_size, n)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            logits = model.forward(X_batch)
            y_pred = model.activations[-1].forward(logits)

            loss = model.loss_fn.forward(y_batch, y_pred)
            if model.optimizer.weight_decay > 0:
                l2_reg = sum(np.sum(layer.W ** 2) for layer in model.layers)
                loss += 0.5 * model.optimizer.weight_decay * l2_reg

            epoch_loss += loss * (end - start)
            epoch_correct += np.sum(
                np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1)
            )

            model.backward(y_batch, logits)

            if log_gradients:
                # gradient norm for first hidden layer weights
                grad_W_first = model.grad_W[-1]
                gnorm = np.linalg.norm(grad_W_first)
                wandb.log({"grad_norm_first_hidden": gnorm}, commit=False)

            model.update_weights()

        train_loss = epoch_loss / n
        train_acc = epoch_correct / n

        val_loss, val_acc = model.evaluate(X_val, y_val)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    run.finish()
    return history, (test_loss, test_acc)


# --------------------------
# 2.1 Data exploration
# --------------------------

def run_2_1_data_exploration():
    from keras.datasets import mnist

    (X_train, y_train), _ = mnist.load_data()
    num_classes = 10
    samples_per_class = 5
    table = wandb.Table(columns=["image", "label"])

    for cls in range(num_classes):
        idxs = np.where(y_train == cls)[0]
        chosen = np.random.choice(idxs, size=samples_per_class, replace=False)
        for idx in chosen:
            img = X_train[idx]
            table.add_data(wandb.Image(img), int(cls))

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="2.1_data_exploration_mnist",
        group="2.1_data_exploration",
        tags=["2.1"],
        reinit=True,
    )
    run.log({"samples_table": table})
    run.finish()


# --------------------------
# 2.2 Hyperparameter sweep
# --------------------------

def run_2_2_sweep(count=20):
    """
    Launch a W&B sweep. For the full assignment requirement set count>=100.
    """
    sweep_config = {
        "method": "random",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "min": 1e-4,
                "max": 1e-1,
                "distribution": "log_uniform_values",
            },
            "batch_size": {"values": [32, 64, 128]},
            "num_layers": {"values": [2, 3, 4]},
            "hidden_size": {"values": [64, 128]},
            "optimizer": {"values": ["sgd", "momentum", "adam", "rmsprop"]},
            "activation": {"values": ["relu", "tanh"]},
            "weight_decay": {"values": [0.0, 1e-4, 5e-4]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT)

    def sweep_train():
        """
        One sweep trial. We MUST call wandb.init() first so wandb.config is populated.
        We then train inside this same run (no nested runs).
        """
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group="2.2_sweep",
            tags=["2.2", "sweep"],
            reinit=True,
        )
        cfg_dict = run.config
        cfg = make_cfg(
            dataset="mnist",
            epochs=10,
            batch_size=cfg_dict.batch_size,
            loss="cross_entropy",
            optimizer=cfg_dict.optimizer,
            learning_rate=cfg_dict.learning_rate,
            weight_decay=cfg_dict.weight_decay,
            num_layers=cfg_dict.num_layers,
            hidden_size=cfg_dict.hidden_size,
            activation=cfg_dict.activation,
            weight_init="xavier",
        )

        # Inline training loop (similar to run_one_training, but reusing this run)
        model, (X_train, y_train, X_val, y_val, X_test, y_test) = build_model_and_data(
            cfg
        )
        n = X_train.shape[0]
        for epoch in range(1, cfg.epochs + 1):
            perm = np.random.permutation(n)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0.0
            epoch_correct = 0
            num_batches = int(np.ceil(n / cfg.batch_size))

            for b in range(num_batches):
                start = b * cfg.batch_size
                end = min(start + cfg.batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = model.forward(X_batch)
                y_pred = model.activations[-1].forward(logits)

                loss = model.loss_fn.forward(y_batch, y_pred)
                if model.optimizer.weight_decay > 0:
                    l2_reg = sum(np.sum(layer.W ** 2) for layer in model.layers)
                    loss += 0.5 * model.optimizer.weight_decay * l2_reg

                epoch_loss += loss * (end - start)
                epoch_correct += np.sum(
                    np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1)
                )

                model.backward(y_batch, logits)
                model.update_weights()

            train_loss = epoch_loss / n
            train_acc = epoch_correct / n
            val_loss, val_acc = model.evaluate(X_val, y_val)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

        test_loss, test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        run.finish()

    wandb.agent(sweep_id, function=sweep_train, count=count)


# --------------------------
# 2.3 Optimizer showdown
# --------------------------

def run_2_3_optimizers():
    optimizers = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
    for opt in optimizers:
        cfg = make_cfg(
            dataset="mnist",
            epochs=10,
            batch_size=64,
            loss="cross_entropy",
            optimizer=opt,
            learning_rate=0.001,
            weight_decay=0.0,
            num_layers=3,
            hidden_size=[128, 128, 128],
            activation="relu",
            weight_init="xavier",
        )
        run_one_training(
            cfg,
            run_name=f"2.3_optimizer_{opt}",
            group="2.3_optimizer_showdown",
            tags=["2.3", f"opt_{opt}"],
        )


# --------------------------
# 2.4 Vanishing gradient
# --------------------------

def run_2_4_vanishing():
    for act in ["sigmoid", "relu"]:
        cfg = make_cfg(
            dataset="mnist",
            epochs=10,
            batch_size=64,
            loss="cross_entropy",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            num_layers=3,
            hidden_size=[128, 128, 128],
            activation=act,
            weight_init="xavier",
        )
        run_one_training(
            cfg,
            run_name=f"2.4_vanishing_grads_{act}",
            group="2.4_vanishing_gradient",
            tags=["2.4", f"act_{act}"],
            log_gradients=True,
        )


# --------------------------
# 2.5 Dead neurons
# --------------------------

def run_2_5_dead_neurons():
    def log_dead_neurons(cfg, run_name):
        model, (X_train, y_train, X_val, y_val, X_test, y_test) = build_model_and_data(
            cfg
        )
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=cfg.__dict__,
            name=run_name,
            group="2.5_dead_neurons",
            tags=["2.5"],
            reinit=True,
        )

        n = X_train.shape[0]
        for epoch in range(1, cfg.epochs + 1):
            perm = np.random.permutation(n)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            num_batches = int(np.ceil(n / cfg.batch_size))

            for b in range(num_batches):
                start = b * cfg.batch_size
                end = min(start + cfg.batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = model.forward(X_batch)
                y_pred = model.activations[-1].forward(logits)
                model.backward(y_batch, logits)
                model.update_weights()

                first_layer_out = model.activations[0].output
                dead_frac = np.mean(first_layer_out == 0, axis=0)
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": b,
                        "dead_neuron_fraction_mean": float(np.mean(dead_frac)),
                    }
                )

        run.finish()

    cfg_relu = make_cfg(
        dataset="mnist",
        epochs=10,
        batch_size=64,
        loss="cross_entropy",
        optimizer="sgd",
        learning_rate=0.1,
        weight_decay=0.0,
        num_layers=3,
        hidden_size=[128, 128, 128],
        activation="relu",
        weight_init="xavier",
    )
    cfg_tanh = make_cfg(
        dataset="mnist",
        epochs=10,
        batch_size=64,
        loss="cross_entropy",
        optimizer="sgd",
        learning_rate=0.1,
        weight_decay=0.0,
        num_layers=3,
        hidden_size=[128, 128, 128],
        activation="tanh",
        weight_init="xavier",
    )

    log_dead_neurons(cfg_relu, "2.5_dead_neurons_relu")
    log_dead_neurons(cfg_tanh, "2.5_dead_neurons_tanh")


# --------------------------
# 2.6 Loss comparison
# --------------------------

def run_2_6_loss_comparison():
    for loss_name in ["mean_squared_error", "cross_entropy"]:
        cfg = make_cfg(
            dataset="mnist",
            epochs=15,
            batch_size=64,
            loss=loss_name,
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            num_layers=3,
            hidden_size=[128, 128, 128],
            activation="relu",
            weight_init="xavier",
        )
        run_one_training(
            cfg,
            run_name=f"2.6_loss_{loss_name}",
            group="2.6_loss_comparison",
            tags=["2.6", f"loss_{loss_name}"],
        )


# --------------------------
# 2.7 Global performance
# --------------------------

def run_2_7_global_performance():
    cfg = make_cfg(
        dataset="mnist",
        epochs=20,
        batch_size=64,
        loss="cross_entropy",
        optimizer="adam",
        learning_rate=0.0005,
        weight_decay=0.0,
        num_layers=3,
        hidden_size=[128, 128, 128],
        activation="relu",
        weight_init="xavier",
    )
    run_one_training(
        cfg,
        run_name="2.7_global_perf_overfit_candidate",
        group="2.7_global_performance",
        tags=["2.7", "global_perf"],
    )


# --------------------------
# 2.8 Error analysis
# --------------------------

def run_2_8_error_analysis():
    cfg = make_cfg(
        dataset="mnist",
        epochs=1,
        batch_size=64,
        loss="cross_entropy",
        optimizer="adam",
        learning_rate=0.001,
        weight_decay=0.0,
        num_layers=3,
        hidden_size=[128, 128, 128],
        activation="relu",
        weight_init="xavier",
    )
    model, (_, _, _, _, X_test, y_test) = build_model_and_data(cfg)
    # Load your best weights if desired, else this uses a freshly initialized model
    # model.load("models/best_model.npy")

    logits = model.forward(X_test)
    probs = model.activations[-1].forward(logits)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(probs, axis=1)

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=cfg.__dict__,
        name="2.8_error_analysis",
        group="2.8_error_analysis",
        tags=["2.8"],
        reinit=True,
    )

    # Log confusion matrix with W&B built-in plot (no external plotting libs)
    cm_plot = wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=y_pred,
        class_names=[str(i) for i in range(10)],
    )
    run.log({"confusion_matrix": cm_plot})

    incorrect_idx = np.where(y_true != y_pred)[0]
    sample_idx = np.random.choice(
        incorrect_idx, size=min(25, len(incorrect_idx)), replace=False
    )
    error_table = wandb.Table(columns=["image", "true_label", "pred_label"])
    X_test_img = X_test.reshape(-1, 28, 28)
    for idx in sample_idx:
        img = X_test_img[idx]
        error_table.add_data(wandb.Image(img), int(y_true[idx]), int(y_pred[idx]))

    run.log({"error_examples": error_table})
    run.finish()


# --------------------------
# 2.9 Init & symmetry
# --------------------------

def run_2_9_init_symmetry():
    def log_gradients_across_neurons(weight_init, run_name):
        cfg = make_cfg(
            dataset="mnist",
            epochs=1,
            batch_size=64,
            loss="cross_entropy",
            optimizer="sgd",
            learning_rate=0.01,
            weight_decay=0.0,
            num_layers=2,
            hidden_size=[64, 64],
            activation="relu",
            weight_init=weight_init,
        )

        model, (X_train, y_train, X_val, y_val, X_test, y_test) = build_model_and_data(
            cfg
        )
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=cfg.__dict__,
            name=run_name,
            group="2.9_init_symmetry",
            tags=["2.9", f"init_{weight_init}"],
            reinit=True,
        )

        n = X_train.shape[0]
        steps = 0
        max_steps = 50
        num_neurons_to_track = 5

        while steps < max_steps:
            perm = np.random.permutation(n)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            num_batches = int(np.ceil(n / cfg.batch_size))

            for b in range(num_batches):
                if steps >= max_steps:
                    break
                start = b * cfg.batch_size
                end = min(start + cfg.batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = model.forward(X_batch)
                y_pred = model.activations[-1].forward(logits)
                model.backward(y_batch, logits)

                grad_W_first = model.grad_W[-1]
                grads_neurons = grad_W_first[:, :num_neurons_to_track]
                mean_abs = np.mean(np.abs(grads_neurons), axis=0)
                log_dict = {
                    f"neuron_{i}_mean_abs_grad": float(m)
                    for i, m in enumerate(mean_abs)
                }
                log_dict["step"] = steps
                wandb.log(log_dict)

                model.update_weights()
                steps += 1

        run.finish()

    log_gradients_across_neurons("zeros", "2.9_init_zeros")
    log_gradients_across_neurons("xavier", "2.9_init_xavier")


# --------------------------
# 2.10 Fashion-MNIST transfer
# --------------------------

def run_2_10_fashion_transfer():
    configs = [
        make_cfg(
            dataset="fashion_mnist",
            epochs=15,
            batch_size=64,
            loss="cross_entropy",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            num_layers=3,
            hidden_size=[128, 128, 128],
            activation="relu",
            weight_init="xavier",
        ),
        make_cfg(
            dataset="fashion_mnist",
            epochs=20,
            batch_size=64,
            loss="cross_entropy",
            optimizer="adam",
            learning_rate=0.0005,
            weight_decay=0.0,
            num_layers=4,
            hidden_size=[128, 128, 128, 128],
            activation="relu",
            weight_init="xavier",
        ),
        make_cfg(
            dataset="fashion_mnist",
            epochs=20,
            batch_size=64,
            loss="cross_entropy",
            optimizer="rmsprop",
            learning_rate=0.0007,
            weight_decay=0.0,
            num_layers=3,
            hidden_size=[128, 128, 128],
            activation="tanh",
            weight_init="xavier",
        ),
    ]

    for i, cfg in enumerate(configs, start=1):
        run_one_training(
            cfg,
            run_name=f"2.10_fashionmnist_cfg{i}",
            group="2.10_fashion_mnist_transfer",
            tags=["2.10", f"cfg_{i}"],
        )


SECTION_FUNCS = {
    "2.1": run_2_1_data_exploration,
    "2.2": run_2_2_sweep,
    "2.3": run_2_3_optimizers,
    "2.4": run_2_4_vanishing,
    "2.5": run_2_5_dead_neurons,
    "2.6": run_2_6_loss_comparison,
    "2.7": run_2_7_global_performance,
    "2.8": run_2_8_error_analysis,
    "2.9": run_2_9_init_symmetry,
    "2.10": run_2_10_fashion_transfer,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        type=str,
        help="Which 2.xx section to run, e.g. 2.3. If omitted and --run-all is set, runs all (except long sweep).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all key experiments except the long 2.2 sweep.",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=20,
        help="Number of runs for 2.2 sweep (use >=100 for full assignment).",
    )
    args = parser.parse_args()

    if args.section:
        if args.section == "2.2":
            run_2_2_sweep(count=args.sweep_count)
        else:
            func = SECTION_FUNCS.get(args.section)
            if func is None:
                raise ValueError(f"Unknown section {args.section}")
            func()
        return

    if args.run_all:
        # Run everything except full sweep by default
        run_2_1_data_exploration()
        run_2_3_optimizers()
        run_2_4_vanishing()
        run_2_5_dead_neurons()
        run_2_6_loss_comparison()
        run_2_7_global_performance()
        run_2_8_error_analysis()
        run_2_9_init_symmetry()
        run_2_10_fashion_transfer()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
