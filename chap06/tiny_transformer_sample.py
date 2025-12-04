#!/usr/bin/env python3
# tiny_transformer_sample.py
import argparse
import torch
import torch.nn.functional as F
from tiny_transformer_model import TinyTransformer
from tiny_transformer_data import CharTokenizer, load_text

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tiny_transformer.pt")
    ap.add_argument("--data_path", default="data/source.txt")
    ap.add_argument("--prompt", default="int main() {")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)      # 0 = disabled
    ap.add_argument("--top_p", type=float, default=0.0)  # 0 = disabled
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # tokenizer from the same text used in training
    text = load_text(args.data_path)
    tok = CharTokenizer(text)

    # load checkpoint + config
    state = torch.load(args.ckpt, map_location=device)
    cfg = state.get("config", {
        "vocab_size": state.get("vocab_size", tok.vocab_size),
        "d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024, "block_size": 128
    })

    model = TinyTransformer(**cfg).to(device)

    # handle torch.compile-wrapped checkpoints
    sd = state["model"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    model.eval()

    # encode prompt
    ids = torch.tensor([[tok.stoi.get(ch, 0) for ch in args.prompt]], dtype=torch.long, device=device)

    # autoregressive generation
    for _ in range(args.steps):
        idx_cond = ids[:, -cfg["block_size"]:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(args.temperature, 1e-6)

        # Optional: nudge formatting tokens to get tidier C
        for ch, bonus in {'\n': 1.2, ';': 0.3, ' ': 0.1}.items():
            tid = tok.stoi.get(ch)
            if tid is not None:
                logits[0, tid] += bonus

        # top-k / top-p filtering (optional)
        if args.top_k and args.top_k > 0:
            v, _ = torch.topk(logits, k=min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        if args.top_p and args.top_p > 0.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            cutoff = cumprobs > args.top_p
            # keep at least one token
            cutoff[..., 0] = False
            sorted_logits[cutoff] = -float("inf")
            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
        ids = torch.cat([ids, next_id], dim=1)

    out = "".join(tok.itos[int(i)] for i in ids[0].tolist())
    print(out)

if __name__ == "__main__":
    main()
