# tiny_transformer_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

class CausalSelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Input shape:  (B, T, d_model)
    Output shape: (B, T, head_size)
    """
    def __init__(self, d_model: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        # lower-triangular mask for causal attention
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

        self.scale = 1.0 / math.sqrt(head_size)

    def forward(self, x: torch.Tensor, return_attn=False) -> torch.Tensor:
        B, T, _ = x.shape

        # project to queries, keys, and values
        K = self.key(x)   # (B, T, head_size)
        Q = self.query(x) # (B, T, head_size)
        V = self.value(x) # (B, T, head_size)

        # compute scaled dot-product attention scores
        # scores: (B, T, T)
        scores = Q @ K.transpose(-2, -1) * self.scale

        # apply causal mask: disallow attending to future positions
        # mask[ :T, :T ] is a (T, T) lower-triangular matrix
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # row-wise softmax over the last dimension
        weights = F.softmax(scores, dim=-1)

        # weighted sum of values
        out = weights @ V  # (B, T, head_size)

        if return_attn:
            return out, weights
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input shape:  (B, T, d_model)
    Output shape: (B, T, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_size = d_model // num_heads

        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(d_model, head_size, block_size)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads * head_size, d_model)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, return_attn=False, head_index=0) -> torch.Tensor:
        # apply all heads in parallel and concatenate along the last dimension
        head_outputs = [head(x) for head in self.heads]   # list of (B, T, head_size)
        concat = torch.cat(head_outputs, dim=-1)          # (B, T, num_heads * head_size)
        out = self.proj(concat)                           # (B, T, d_model)
        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network:
      FFN(x) = GELU(x W1 + b1) W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    Transformer decoder block (pre-norm):
      x -> LN -> MHA -> +x -> LN -> FFN -> + (previous)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, block_size=block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """
    Minimal decoder-only transformer for next-character prediction.

    Args:
        vocab_size: number of distinct tokens
        d_model: embedding/model dimension
        n_layers: number of decoder blocks
        n_heads: number of attention heads
        d_ff: hidden size of FFN
        block_size: maximum context length (sequence length)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 4 * 256,
        block_size: int = 128,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        # token + positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        # stack of decoder blocks
        self.blocks = nn.ModuleList(
            [Block(d_model=d_model, num_heads=n_heads, d_ff=d_ff, block_size=block_size)
             for _ in range(n_layers)]
        )

        # final layer norm and language-model head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        # init: small std helps stability for tiny models
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively sample next tokens.
        idx: (B, T) integer token ids
        Returns: (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop to context
            logits, _ = self.forward(idx_cond)    # (B, T, vocab)
            next_logits = logits[:, -1, :]        # last time step
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def forward(
        self,
        idx: torch.Tensor,                 # (B, T) ints
        targets: Optional[torch.Tensor] = None,  # (B, T) ints
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # embeddings and positions
        tok = self.tok_emb(idx)                               # (B, T, d_model)
        pos_ids = torch.arange(T, device=idx.device)          # (T,)
        pos = self.pos_emb(pos_ids)[None, :, :].expand(B, T, -1)  # (B, T, d_model)
        x = tok + pos

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # head
        x = self.ln_f(x)
        logits = self.head(x)                                  # (B, T, vocab)

        loss = None
        if targets is not None:
            # flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(B * T, self.vocab_size),
                targets.view(B * T),
            )
        return logits, loss

    @torch.no_grad()
    def get_attention(self, idx: torch.Tensor, layer: int = 0, head: int = 0):
        """
        Return attention weights for a given input sequence.

        Args:
            idx:   (B, T) token ids
            layer: which transformer block (0-based)
            head:  which attention head in that block (0-based)

        Returns:
            attn: (T, T) tensor with attention weights for that head.
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        device = idx.device

        # Same embedding logic as in forward()
        tok = self.tok_emb(idx)                               # (B, T, d_model)
        pos_ids = torch.arange(T, device=device)              # (T,)
        pos = self.pos_emb(pos_ids)[None, :, :].expand(B, T, -1)
        x = tok + pos                                         # (B, T, d_model)

        # Walk through blocks until the target layer
        for i, blk in enumerate(self.blocks):
            if i == layer:
                # pre-norm: ln1 before attention
                x_norm = blk.ln1(x)

                # talk directly to the chosen head in this block
                head_module = blk.attn.heads[head]
                _, attn = head_module(x_norm, return_attn=True)  # (B, T, T)

                return attn[0].detach().cpu()  # first batch element

            # normal block pass for earlier layers
            x = blk(x)

        raise ValueError(f"Layer index {layer} out of range (have {len(self.blocks)} blocks)")
