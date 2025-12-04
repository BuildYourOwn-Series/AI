#!/usr/bin/env python3
# tiny_transformer_attn.py
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from tiny_transformer_model import TinyTransformer
from tiny_transformer_data import CharTokenizer, load_text


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tiny_transformer.pt")
    ap.add_argument("--data_path", default="data/source.txt")
    ap.add_argument("--prompt", default="if (tok == TOK_GT) {\n")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--outfile", default="attention.png")
    args = ap.parse_args()

    device = "cpu"
    text = load_text(args.data_path)
    tok = CharTokenizer(text)

    # load model + config
    state = torch.load(args.ckpt, map_location=device)
    cfg = state["config"]

    # rebuild the model
    model = TinyTransformer(**cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # encode prompt
    ids = torch.tensor([[tok.stoi[c] for c in args.prompt]], dtype=torch.long)

    # get attention heatmap (T × T)
    attn = model.get_attention(ids, layer=args.layer, head=args.head)

    # convert IDs back to character labels for axes
    tokens = [tok.itos[int(i)] for i in ids[0]]

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(attn, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    ax.set_title(f"Attention Head {args.head}, Layer {args.layer}")
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=200)
    print("Saved", args.outfile)


if __name__ == "__main__":
    main()
