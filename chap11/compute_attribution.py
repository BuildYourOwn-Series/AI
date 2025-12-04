import torch

def compute_attribution(model, tokenizer, prompt):
    device = next(model.parameters()).device
    model.eval()

    # Encode and enable gradients on the embedding indices
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens)[None].to(device)
    input_ids = input_ids.clone().detach().requires_grad_(False)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs[:, -1, :]  # next-token logits

    # Choose the predicted token's logit
    target = logits.argmax(dim=-1)
    selected = logits[0, target]

    # Backpropagate to the embedding layer
    model.embedding.weight.grad = None
    selected.backward()

    # Extract gradient vectors for the input tokens
    grads = model.embedding.weight.grad
    token_grads = grads[input_ids[0]]

    # Attribution = gradient norm
    attribution = token_grads.norm(dim=1)
    return attribution.cpu(), input_ids[0].cpu()

