#!/usr/bin/env python3
# tiny_transformer_train.py

import time
import argparse
import torch
from torch.optim import AdamW

from tiny_transformer_data import build_dataloaders
from tiny_transformer_model import TinyTransformer

import os, multiprocessing as mp
# Use OMP threads ~ number of physical cores
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(max(1, mp.cpu_count() // 2 or 1))
# Let PyTorch pick it up now
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

def train(
    data_path: str = "data/source.txt",
    block_size: int = 128,
    batch_size: int = 32,
    d_model: int = 192,          # slightly smaller default
    n_layers: int = 3,           # slightly smaller default
    n_heads: int = 3,
    d_ff: int = 4 * 192,
    lr: float = 3e-4,
    max_iters: int = 2000,
    eval_interval: int = 200,
    eval_batches: int = 8,       # NEW: cap validation cost
    device: str = "cpu",
):
    torch.set_grad_enabled(True)
    torch.set_num_threads(max(1, torch.get_num_threads()))  # no-op but explicit

    print(f"[info] loading data from {data_path} …", flush=True)
    t0 = time.time()
    train_loader, val_loader, tokenizer = build_dataloaders(
        path=data_path,
        block_size=block_size,
        batch_size=batch_size,
    )
    print(f"[info] data ready in {time.time()-t0:.2f}s | vocab={tokenizer.vocab_size} "
          f"| block={block_size} | batch={batch_size}", flush=True)

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        block_size=block_size,
    ).to(device)
    try:
        model = torch.compile(model)  # may speed up forward/backward on CPU
    except Exception as e:
        print(f"[info] torch.compile unavailable: {e}")
    print(f"[info] model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    def evaluate(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for b_idx, (xb, yb) in enumerate(loader):
                xb = xb.to(device); yb = yb.to(device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
                if b_idx + 1 >= eval_batches:   # NEW: only a few batches
                    break
        model.train()
        return sum(losses) / len(losses)

    iter_count = 0
    model.train()
    last_print = time.time()

    for epoch in range(10**9):  # effectively infinite; we stop by iters
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            iter_count += 1

            # frequent heartbeat
            if iter_count % 50 == 0 or (time.time() - last_print) > 5:
                last_print = time.time()
                print(f"[{iter_count:>6}] train_loss={loss.item():.4f}", flush=True)

            if iter_count % eval_interval == 0:
                t_eval = time.time()
                val_loss = evaluate(val_loader)
                print(f"[{iter_count:>6}] val_loss={val_loss:.4f} "
                      f"(eval {time.time()-t_eval:.2f}s)", flush=True)

            if iter_count >= max_iters:
                to_save = model
                if hasattr(model, "_orig_mod"):  # torch.compile wrapped
                    to_save = model._orig_mod

                torch.save(
                    {
                        "model": to_save.state_dict(),
                        "config": {
                            "vocab_size": tokenizer.vocab_size,
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "n_heads": n_heads,
                            "d_ff": d_ff,
                            "block_size": block_size,
                        },
                    },
                    "tiny_transformer.pt",
                )
                print("[done] saved tiny_transformer.pt", flush=True)
                return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/source.txt")
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=768)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(**vars(args))
