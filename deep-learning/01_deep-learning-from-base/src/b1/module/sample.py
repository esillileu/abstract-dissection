import numpy as np
import b1.func as F

def init_network(): # 784, 50, 100, 10
    network = {}
    network["W1"] = np.random.rand(784, 50)
    network["b1"] = np.random.rand(50)
    network["W2"] = np.random.rand(50, 100)
    network["b2"] = np.random.rand(100)
    network["W3"] = np.random.rand(100, 10)
    network["b3"] = np.random.rand(10)

    return network

def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = F.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = F.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = F.softmax(a3)

    return y
