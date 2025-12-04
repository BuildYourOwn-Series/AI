def save_mlp(net, path="mlp_mnist.npz"):
    """
    Save MLP weights and biases to a compressed .npz file.
    """
    arrays = {}
    for i, layer in enumerate(net.layers):
        arrays[f"W{i}"] = layer.W
        arrays[f"b{i}"] = layer.b
    np.savez(path, **arrays)
    print(f"Saved network to {path}")
