import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)

        for k in params.keys():
            self.h[k] += grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + 1e-7)


class RMSprop:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-7):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)

        for k in params.keys():
            self.h[k] = self.rho * self.h[k] + (1 - self.rho) * grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + self.eps)


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 1

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            for k, v in params.items():
                self.m[k] = np.zeros_like(v)

        if self.v is None:
            self.v = {}
            for k, v in params.items():
                self.v[k] = np.zeros_like(v)

        for k in params.keys():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]            
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] * grads[k]

            m = self.m[k] / (1 - self.b1**self.t)
            v = self.v[k] / (1 - self.b2**self.t)

            params[k] -= self.lr * m / (np.sqrt(v) + self.eps)

        self.t += 1