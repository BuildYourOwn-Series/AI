#!/usr/bin/env python3
# andor.py
# A perceptron for binary classification of "and" and "or" functions
# Python >=3.9 recommended.

import numpy as np

class Perceptron:
    def __init__(self, in_dim, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-1, 1, in_dim)
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)

    def train(self, X, Y, epochs=10):
        for _ in range(epochs):
            for x, y in zip(X, Y):
                yhat = self.predict(x)
                if yhat != y:
                    self.w += self.lr * (y - yhat) * x
                    self.b += self.lr * (y - yhat)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
X = 2*X - 1  # map 0→-1, 1→+1

Y_and = np.array([-1,-1,-1,+1], dtype=float)
Y_or  = np.array([-1,+1,+1,+1], dtype=float)

p_and = Perceptron(2, lr=0.1, seed=42)
p_and.train(X, Y_and, epochs=10)

p_or = Perceptron(2, lr=0.1, seed=42)
p_or.train(X, Y_or, epochs=10)

print("AND weights:", p_and.w, "bias:", p_and.b)
print("OR  weights:", p_or.w,  "bias:", p_or.b)

