def trace_reasoning(model, tokenizer, prompt, max_tokens=30):
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens)[None].to(device)

    logits_trace      = []
    attention_trace   = []
    attribution_trace = []
    drift_trace       = []

    # Set up attention hooks
    attn_scores, handles = capture_attention(model)

    prev_hidden = None

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids, return_hidden=True)

        # Logits
        logits = outputs.logits[:, -1, :]
        logits_trace.append(logits.cpu())

        # Attention
        attention_trace.append([a.cpu() for a in attn_scores])
        attn_scores.clear()

        # Attribution (single-step)
        attr, ids = compute_attribution(model, tokenizer, 
                                        tokenizer.decode(input_ids[0]))
        attribution_trace.append(attr)

        # Drift
        hidden = outputs.hidden_states[-1][0, -1].cpu()
        if prev_hidden is not None:
            drift_trace.append(torch.norm(hidden - prev_hidden).item())
        prev_hidden = hidden

        # Next token (greedy)
        next_token = logits.argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)

    # Clean up hooks
    for h in handles:
        h.remove()

    return {
        "logits": logits_trace,
        "attention": attention_trace,
        "attribution": attribution_trace,
        "drift": drift_trace,
    }

