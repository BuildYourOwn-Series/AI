def trace_attention(model, tokenizer, prompt):
    model.eval()
    device = next(model.parameters()).device

    # Encode input
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens)[None].to(device)

    # Install attention hooks
    attention_scores, handles = capture_attention(model)

    # Run a forward pass
    with torch.no_grad():
        _ = model(input_ids)

    # Clean up hooks
    for h in handles:
        h.remove()

    return attention_scores, input_ids

