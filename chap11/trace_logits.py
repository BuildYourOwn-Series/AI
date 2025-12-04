import torch

def trace_logits(model, tokenizer, prompt, max_tokens=40):
    model.eval()
    device = next(model.parameters()).device

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long)[None].to(device)

    logits_trace = []

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[:, -1, :]  # Last-token logits

            # Store a snapshot for inspection
            logits_trace.append(logits.cpu())

            # Greedy decoding for clarity
            next_token = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids,
                                   next_token[:, None]], dim=1)

    return logits_trace

