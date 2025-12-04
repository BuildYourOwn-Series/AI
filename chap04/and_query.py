#!/usr/bin/env python3
# and_query.py
# Query the perceptron for "and"
# Python >=3.9 recommended

import numpy as np

def perceptron_side(x, w, b, eps=1e-9):
    """
    x: shape (2,) here, but works for (n,)
    w: weights, shape (n,)
    b: scalar bias
    Returns: (label, signed_distance, raw_score)
      label in {-1, 0, +1} where 0 = 'on the line' (within eps)
      signed_distance = (w·x + b)/||w||
      raw_score = w·x + b
    """
    s = float(np.dot(w, x) + b)
    if abs(s) < eps:
        return 0, 0.0, s
    label = 1.0 if s > 0 else -1.0
    return label, s / np.linalg.norm(w), s

w = np.array([0.1479121, 0.27775688])
b = -0.4

# Truth-table inputs mapped from {0,1} -> {-1,+1}
X01 = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
X   = 2*X01 - 1

for x01, x in zip(X01, X):
    label, dist, score = perceptron_side(x, w, b)
    side = "+1 half-space" if label==1 else ("-1 half-space" if label==-1 else "on the boundary")
    print(f"x={x01} -> score={score:+.3f}, dist={dist:+.3f}, side={side}")
