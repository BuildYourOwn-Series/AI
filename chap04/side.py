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
