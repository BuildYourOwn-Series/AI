import torch

def trace_embedding_drift(model, tokenizer, prompt, max_tokens=40):
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long)[None].to(device)

    hidden_trace = []

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids, return_hidden=True)
            hidden = outputs.hidden_states[-1]  # last layer
            hidden_trace.append(hidden[0, -1].cpu())  # last token

            # Next token (greedy for clarity)
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids,
                                   next_token[:, None]], dim=1)

    return hidden_trace

