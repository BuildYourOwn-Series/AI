#!/usr/bin/env python3
# and_plot.py
# Plotting the perceptron decision boundary for "and"
# Python >=3.9 recommended

import numpy as np
import matplotlib.pyplot as plt

# Inputs and outputs for the AND function
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y = np.array([-1,-1,-1,+1], dtype=float)

# Convert to from {0,1} to {-1,+1} domain for plotting consistency
X = 2*X - 1

# Learned parameters (example)
w = np.array([0.1479121, 0.27775688])
b = -0.4

# Line x2 = -(w1/w2)*x1 - (b/w2)
x1 = np.linspace(-1.5, 1.5, 100)
x2 = -(w[0]/w[1]) * x1 - b / w[1]

# Plot points and line
plt.figure(figsize=(4,4))
plt.scatter(X[Y==-1,0], X[Y==-1,1], color='gray', label='-1')
plt.scatter(X[Y==+1,0], X[Y==+1,1], color='blue', label='+1')
plt.plot(x1, x2, 'r-', label=r'$w\cdot x + b = 0$')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.grid(True, linestyle=':')
plt.gca().set_aspect('equal')
plt.show()

