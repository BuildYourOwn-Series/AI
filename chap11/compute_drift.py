import torch

def compute_drift(hidden_trace):
    drifts = []
    for h_prev, h_curr in zip(hidden_trace[:-1], hidden_trace[1:]):
        d = torch.norm(h_curr - h_prev).item()
        drifts.append(d)
    return drifts

