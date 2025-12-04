#!/usr/bin/env python3
# nots.py
# A binary classification of "nand", "nor" and "xnor" using our "mynn" library
# Python >=3.9 recommended.

from mynn import Tensor, nn, optim, no_grad
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], np.float32)

class GateNet(nn.Module):
    """
    A minimal multilayer perceptron for learning Boolean gates.
    This is the same architecture used for the XOR experiment: a
    two-unit hidden layer with a nonlinearity and a single output.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.a1 = nn.Tanh()
        self.l2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.l2(self.a1(self.l1(x)))

gates = {
    "nand": np.array([[1],[1],[1],[0]], np.float32),
    "nor":  np.array([[1],[0],[0],[0]], np.float32),
    "xnor": np.array([[1],[0],[0],[1]], np.float32),
}

for name, target in gates.items():
    model = GateNet()
    opt   = optim.Adam(model.parameters(), lr=0.1)
    lossf = nn.MSELoss()

    for epoch in range(1500):
        pred = model(Tensor(X))
        loss = lossf(pred, Tensor(target))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with no_grad():
        out = model(Tensor(X)).data
    print(name, "->", out.round().flatten())
