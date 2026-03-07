import numpy as np


class Optimizer:
    def __init__(self, layers, lr=0.001, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay

    def apply_weight_decay(self, layer):
        if self.weight_decay > 0:
            layer.grad_W += self.weight_decay * layer.W


# ------------------------------------------------
# SGD
# ------------------------------------------------

class SGD(Optimizer):

    def step(self):
        for layer in self.layers:

            self.apply_weight_decay(layer)

            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


# ------------------------------------------------
# Momentum
# ------------------------------------------------

class Momentum(Optimizer):

    def __init__(self, layers, lr=0.001, weight_decay=0.0, beta=0.9):

        super().__init__(layers, lr, weight_decay)

        self.beta = beta

        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self):

        for i, layer in enumerate(self.layers):

            self.apply_weight_decay(layer)

            self.vW[i] = self.beta * self.vW[i] + self.lr * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + self.lr * layer.grad_b

            layer.W -= self.vW[i]
            layer.b -= self.vb[i]


# ------------------------------------------------
# NAG
# ------------------------------------------------

class NAG(Optimizer):

    def __init__(self, layers, lr=0.001, weight_decay=0.0, beta=0.9):

        super().__init__(layers, lr, weight_decay)

        self.beta = beta

        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self):

        for i, layer in enumerate(self.layers):

            self.apply_weight_decay(layer)

            prev_vW = self.vW[i]
            prev_vb = self.vb[i]

            self.vW[i] = self.beta * self.vW[i] + self.lr * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + self.lr * layer.grad_b

            layer.W -= (-self.beta * prev_vW + (1 + self.beta) * self.vW[i])
            layer.b -= (-self.beta * prev_vb + (1 + self.beta) * self.vb[i])


# ------------------------------------------------
# RMSProp
# ------------------------------------------------

class RMSProp(Optimizer):

    def __init__(self, layers, lr=0.001, weight_decay=0.0, beta=0.9, eps=1e-8):

        super().__init__(layers, lr, weight_decay)

        self.beta = beta
        self.eps = eps

        self.sW = [np.zeros_like(l.W) for l in layers]
        self.sb = [np.zeros_like(l.b) for l in layers]

    def step(self):

        for i, layer in enumerate(self.layers):

            self.apply_weight_decay(layer)

            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)


# ------------------------------------------------
# Adam
# ------------------------------------------------

class Adam(Optimizer):

    def __init__(self, layers, lr=0.001, weight_decay=0.0,
                 beta1=0.9, beta2=0.999, eps=1e-8):

        super().__init__(layers, lr, weight_decay)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mW = [np.zeros_like(l.W) for l in layers]
        self.vW = [np.zeros_like(l.W) for l in layers]

        self.mb = [np.zeros_like(l.b) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

        self.t = 0

    def step(self):

        self.t += 1

        for i, layer in enumerate(self.layers):

            self.apply_weight_decay(layer)

            # update biased moments
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (layer.grad_W ** 2)

            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            # bias correction
            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)

            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


# ------------------------------------------------
# Nadam
# ------------------------------------------------

class Nadam(Adam):

    def step(self):

        self.t += 1

        for i, layer in enumerate(self.layers):

            self.apply_weight_decay(layer)

            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (layer.grad_W ** 2)

            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)

            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

            mW_nadam = self.beta1 * mW_hat + (1 - self.beta1) * layer.grad_W
            mb_nadam = self.beta1 * mb_hat + (1 - self.beta1) * layer.grad_b

            layer.W -= self.lr * mW_nadam / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_nadam / (np.sqrt(vb_hat) + self.eps)


# ------------------------------------------------
# OPTIMIZER FACTORY
# ------------------------------------------------

def get_optimizer(name, layers, lr, weight_decay):

    if name == "sgd":
        return SGD(layers, lr, weight_decay)

    if name == "momentum":
        return Momentum(layers, lr, weight_decay)

    if name == "nag":
        return NAG(layers, lr, weight_decay)

    if name == "rmsprop":
        return RMSProp(layers, lr, weight_decay)

    if name == "adam":
        return Adam(layers, lr, weight_decay)

    if name == "nadam":
        return Nadam(layers, lr, weight_decay)

    raise ValueError(f"Unsupported optimizer {name}")