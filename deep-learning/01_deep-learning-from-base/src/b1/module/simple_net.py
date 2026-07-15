import numpy as np

from b1.func import softmax, cee

class SimpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)
        self.b = np.random.rand(3)

    def predict(self, x):
        return np.dot(x, self.W) + self.b

    def loss(self, x, t):
        z = self.predict(x) # logits
        y = softmax(z)
        return cee(y, t)