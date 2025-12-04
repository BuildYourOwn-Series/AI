def load_mlp(path, shapes, alpha=1.0):
    """
    Load a trained MLP from a .npz file created by save_mlp().
    """
    data = np.load(path)
    layers = []
    for i, (din, dout) in enumerate(zip(shapes[:-1], shapes[1:])):
        W = data[f"W{i}"]
        b = data[f"b{i}"]
        layers.append(Layer(W=W, b=b))
    net = MLP(layers, alpha=alpha)
    print(f"Loaded network from {path}")
    return net
