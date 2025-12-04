def capture_attention(model):
    attention_scores = []

    def hook(module, _, output):
        # output = (attn_output, attn_weights)
        attn = output[1].detach().cpu()
        attention_scores.append(attn)

    handles = []
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            h = module.register_forward_hook(hook)
            handles.append(h)

    return attention_scores, handles

