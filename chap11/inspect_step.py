def inspect_step(logits, tokenizer, k=10):
    values, indices = torch.topk(logits, k)
    for v, idx in zip(values[0], indices[0]):
        token = tokenizer.decode([idx])
        print(f"{token:>12}  |  {v.item():.3f}")

