# DA6401 – Assignment 1

## Multi-Layer Perceptron for Image Classification (NumPy Implementation)

This project implements a **configurable Multi-Layer Perceptron (MLP)** from scratch using **NumPy** for image classification on **MNIST** and **Fashion-MNIST**.

The implementation supports:

* Forward propagation
* Backpropagation
* Multiple optimizers
* Configurable architectures
* Weight initialization strategies
* Experiment tracking using **Weights & Biases (W&B)**

No deep learning frameworks such as **PyTorch**, **TensorFlow**, or **JAX** are used.

---

## Submission Links

- **GitHub repository**: [`DL_assignment1`](https://github.com/devharish1371/DL_assignment1)
- **W&B report**: [`DA6401 Assignment 1 – MLP Experiments Report`](https://wandb.ai/devharishabsm-indian-institute-of-technology-madras/da6401-a1/reports/DA6401-Assignment-1-MLP-Experiments-Report--VmlldzoxNjEzNTAzNw?accessToken=lnvatragvy2ibvv51zxmr2m7guwxzuweanyqlfa43t0t0nswerz0s8xy5o985358)

---

# Project Structure

```
.
├── README.md
├── requirements.txt
├── sweep.yaml
│
├── models
│
└── src
    │
    ├── train.py
    ├── inference.py
    ├── wandb_experiments.py
    │
    ├── ann
    │   ├── activations.py
    │   ├── neural_layer.py
    │   ├── neural_network.py
    │   ├── objective_functions.py
    │   ├── optimizers.py
    │   └── __init__.py
    │
    └── utils
        ├── data_loader.py
        └── __init__.py
```

---

# Installation

Clone the repository:

```
git clone <repo_link>
cd assignment1
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Supported Libraries

Allowed libraries used in this implementation:

* numpy
* scikit-learn
* matplotlib
* keras.datasets (for loading MNIST datasets)
* wandb (experiment tracking)

Deep learning frameworks are **not used**.

---

# Dataset

The project supports:

* **MNIST**
* **Fashion-MNIST**

Images are:

* normalized to `[0,1]`
* flattened from **28×28 → 784 features**
* labels are **one-hot encoded**

---

# Training

Training is performed using the CLI interface required by the assignment.

Example:

```
python src/train.py \
-d mnist \
-e 10 \
-b 64 \
-l cross_entropy \
-o adam \
-lr 0.001 \
-wd 0.0005 \
-nhl 3 \
-sz 128 128 128 \
-a relu \
-wi xavier
```

---

# CLI Arguments

| Argument          | Description                              |
| ----------------- | ---------------------------------------- |
| `--dataset`       | mnist or fashion_mnist                   |
| `--epochs`        | number of training epochs                |
| `--batch_size`    | mini-batch size                          |
| `--loss`          | mean_squared_error or cross_entropy      |
| `--optimizer`     | sgd, momentum, nag, rmsprop, adam, nadam |
| `--learning_rate` | learning rate                            |
| `--weight_decay`  | L2 regularization                        |
| `--num_layers`    | number of hidden layers                  |
| `--hidden_size`   | neurons in hidden layers                 |
| `--activation`    | sigmoid, tanh, relu                      |
| `--weight_init`   | random or xavier                         |

---

# Model Saving

After training the following files are generated:

```
best_model.npy
best_config.json
```

These store:

* model weights
* training configuration

---

# Inference

To evaluate the trained model:

```
python src/inference.py \
--model_path best_model.npy \
--config_path best_config.json
```

Output metrics:

* Accuracy
* Precision
* Recall
* F1-Score

---

# Optimizers Implemented

The following optimizers are implemented from scratch:

* SGD
* Momentum
* Nesterov Accelerated Gradient (NAG)
* RMSProp
* Adam
* Nadam

---

# Weight Initialization

Two initialization strategies are supported:

* Random initialization
* Xavier initialization

---

# Weights & Biases Experiments

Experiments are logged using **Weights & Biases**.

Login:

```
wandb login
```

Run experiment tracking:

```
python src/wandb_experiments.py
```

---

# Hyperparameter Sweep

To run the required **100 experiment sweep**:

```
wandb sweep sweep.yaml
```

Then start agents:

```
wandb agent <SWEEP_ID>
```

---

# Evaluation Tasks Covered

The project supports all assignment evaluation components:

* Data exploration
* Hyperparameter sweeps
* Optimizer comparison
* Vanishing gradient analysis
* Dead neuron investigation
* Loss function comparison
* Training vs testing performance analysis
* Confusion matrix visualization
* Weight initialization experiments
* Fashion-MNIST transfer evaluation
