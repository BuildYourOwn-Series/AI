#!/usr/bin/env python3
# tiny_transformer_data.py

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


class CharDataset(Dataset):
    """
    A simple dataset of overlapping character sequences.

    Given a long tensor of token ids 'data' of length N and a block_size B,
    this dataset exposes all windows of length B+1:
        data[0 : B+1], data[1 : B+2], ..., data[N-B-1 : N]
    and returns (x, y) pairs where:
        x = window[0:B], y = window[1:B+1].
    """
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        assert data.ndim == 1, "data must be a 1D tensor of token ids"
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return self.data.size(0) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.data[idx : idx + self.block_size + 1]
        x = window[:-1]
        y = window[1:]
        return x, y


def build_dataloaders(
    path: str,
    block_size: int,
    batch_size: int,
    train_frac: float = 0.9,
) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    """
    Load text, build tokenizer, encode to ids, split into train/val,
    and wrap into PyTorch DataLoaders.
    """
    text = load_text(path)
    tokenizer = CharTokenizer(text)

    # encode entire corpus as a single 1D tensor of token ids
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # simple train/validation split
    n = ids.size(0)
    n_train = int(train_frac * n)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    train_ds = CharDataset(train_ids, block_size)
    val_ds = CharDataset(val_ids, block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=1, prefetch_factor=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=1, prefetch_factor=4, pin_memory=False)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # quick smoke test
    train_loader, val_loader, tokenizer = build_dataloaders("data/source.txt", block_size=128, batch_size=32)

    xb, yb = next(iter(train_loader))
    print("Batch shape:", xb.shape, "->", yb.shape)
    print("Vocab size:", tokenizer.vocab_size)
