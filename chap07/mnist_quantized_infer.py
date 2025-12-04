#!/usr/bin/env python3
"""
mnist_quantized_infer.py

Load the MNIST MLP trained in Chapter 05 (saved as float32
in mnist_mlp.npz), quantize its weights to int8, and run
inference using integer matrix multiplies with float32
dequantization.
"""

import numpy as np
from pathlib import Path

MODEL_PATH = Path("mnist_mlp.npz")
TEST_PATH  = Path("mnist_test.npz")


# ---------------------------------------------------------------------
# Utility: load data and model
# ---------------------------------------------------------------------

def load_mnist_test():
    data = np.load(TEST_PATH)
    x = data["x_test"].astype(np.float32) # (N, 28, 28)
    y = data["y_test"]

    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)  # just in case

    # If labels are one-hot (N, 10), convert to class indices (N,)
    if y.ndim == 2 and y.shape[1] == 10:
        y = np.argmax(y, axis=1)

    y = y.astype(np.int64)
    x = x.reshape(x.shape[0], -1)  # (N, 784)

    return x, y


def load_model():
    data = np.load(MODEL_PATH)
    # These names and shapes must match Chapter 05:
    # W1: (128, 784), b1: (128,), W2: (10, 128), b2: (10,)
    W1 = data["l1.W"].astype(np.float32)
    b1 = data["l1.b"].astype(np.float32)
    W2 = data["l2.W"].astype(np.float32)
    b2 = data["l2.b"].astype(np.float32)
    return (W1, b1, W2, b2)


# ---------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------

def quantize_tensor_sym(x, num_bits=8):
    """
    Symmetric per-tensor quantization to int8.

    Returns:
        q : int8 array
        s : positive float scale
    """
    qmin, qmax = -128, 127
    max_abs = float(np.max(np.abs(x)))
    if max_abs == 0.0:
        # Degenerate case: all zeros.
        s = 1.0
        q = np.zeros_like(x, dtype=np.int8)
    else:
        s = max_abs / qmax
        q = np.round(x / s).astype(np.int8)
    return q, s


def dynamic_quantize_activation(x):
    """
    Dynamically quantize activations per batch to int8
    (symmetric). Returns q, scale.
    """
    qmin, qmax = -128, 127
    max_abs = float(np.max(np.abs(x)))
    if max_abs == 0.0:
        s = 1.0
        q = np.zeros_like(x, dtype=np.int8)
    else:
        s = max_abs / qmax
        q = np.round(x / s).astype(np.int8)
    return q, s


# ---------------------------------------------------------------------
# Float32 reference forward pass
# ---------------------------------------------------------------------

def relu(x):
    return np.maximum(x, 0.0)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # numerical stability
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def forward_float32(x, W1, b1, W2, b2):
    # W1: (784, 128), W2: (128, 10)
    h = relu(x @ W1 + b1)     # (N, 128)
    logits = h @ W2 + b2      # (N, 10)
    probs = softmax(logits)
    return probs


# ---------------------------------------------------------------------
# Quantized forward pass
# ---------------------------------------------------------------------

def forward_int8(x, W1_q, s_W1, b1,
                    W2_q, s_W2, b2,
                    batch_size=256):
    """
    Run a quantized forward pass using:
      - int8 weights (W_q) with symmetric scales s_W
      - dynamic symmetric quantization for activations

    Accumulation happens in int32, then we dequantize
    back to float32 for ReLU / softmax.
    """
    N = x.shape[0]
    probs_all = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = x[start:end]  # float32, (B, 784)

        # Layer 1: quantize activations, int8 matmul, dequantize.
        x_q, s_x = dynamic_quantize_activation(xb)
        # (B, 784) x (128, 784)^T -> (B, 128)
        acc1 = x_q.astype(np.int32) @ W1_q.astype(np.int32)
        # Dequantize: real ≈ (s_x * s_W1) * acc1 + b1
        h = (s_x * s_W1) * acc1.astype(np.float32) + b1
        h = relu(h)

        # Layer 2: quantize hidden activations, int8 matmul, dequantize.
        h_q, s_h = dynamic_quantize_activation(h)
        acc2 = h_q.astype(np.int32) @ W2_q.astype(np.int32)
        logits = (s_h * s_W2) * acc2.astype(np.float32) + b2

        probs = softmax(logits)
        probs_all.append(probs)

    return np.vstack(probs_all)


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def evaluate(probs, y_true):
    preds = np.argmax(probs, axis=1)
    correct = np.sum(preds == y_true)
    return correct / len(y_true)


def model_size_bytes(*arrays):
    return sum(a.nbytes for a in arrays)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model file: {MODEL_PATH}")
    if not TEST_PATH.exists():
        raise SystemExit(f"Missing test file: {TEST_PATH}")

    print("Loading model and test data...")
    W1, b1, W2, b2 = load_model()
    x_test, y_test = load_mnist_test()

    # Float32 reference
    size_float = model_size_bytes(W1, b1, W2, b2) / 1024.0
    probs_float = forward_float32(x_test, W1, b1, W2, b2)
    acc_float = evaluate(probs_float, y_test)
    print(f"float32 model size: {size_float:.1f} KB")
    print(f"float32 accuracy:  {acc_float * 100:.2f} %")

    # Quantize weights (symmetric, per-tensor)
    W1_q, s_W1 = quantize_tensor_sym(W1)
    W2_q, s_W2 = quantize_tensor_sym(W2)
    size_int8 = model_size_bytes(W1_q, b1, W2_q, b2) / 1024.0
    print(f"int8 model size:   {size_int8:.1f} KB")

    # Quantized inference
    probs_int8 = forward_int8(x_test, W1_q, s_W1, b1,
                                       W2_q, s_W2, b2)
    acc_int8 = evaluate(probs_int8, y_test)
    print(f"int8 accuracy:     {acc_int8 * 100:.2f} %")


if __name__ == "__main__":
    main()
