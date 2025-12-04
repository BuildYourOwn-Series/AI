#!/usr/bin/env python3
# xor.py
# A binary classification of "xor" using our "mynn" library
# Python >=3.9 recommended.

from mynn import Tensor, nn, optim, no_grad
import numpy as np

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], np.float32)
Y = np.array([[0],[1],[1],[0]],        np.float32)

# Model: 2 → 2 → 1
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.a1 = nn.Tanh()
        self.l2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.l2(self.a1(self.l1(x)))

model = XORNet()
opt   = optim.Adam(model.parameters(), lr=0.1)
lossf = nn.MSELoss()

#print(list(model.named_parameters()))

for epoch in range(2000):
    x = Tensor(X)
    y = Tensor(Y)

    pred = model(x)
    loss = lossf(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

#    if epoch % 200 == 0:
#        print(epoch, loss.data)

with no_grad():
    out = model(Tensor(X)).data
print("Predictions:", out.round())
