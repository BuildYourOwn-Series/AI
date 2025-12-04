import torch

class DriftDirector:
    def __init__(self, strength=0.15):
        self.prev = None
        self.strength = strength

    def stabilize(self, logits, h_curr, h_prev):
        drift = torch.norm(h_curr - h_prev)
        if drift < 0.0:   # sanity, shouldn't happen
            return logits

        # Compute correction weight (soft clamp)
        w = torch.tanh(self.strength * drift)

        # Apply correction: bias logits slightly toward earlier distribution
        corrected = (1 - w) * logits + w * logits.mean(dim=-1, keepdim=True)
        return corrected

def generate_with_director(model, tokenizer, prompt, director, max_steps=40):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens)[None].cuda()
    director.prev = None

    for _ in range(max_steps):
        out = model(input_ids, return_hidden=True)
        h = out.hidden_states[-1][0, -1]

        if director.prev is not None:
            logits = out.logits[:, -1, :]
            logits = director.stabilize(logits, h, director.prev)
            next_tok = logits.argmax(dim=-1)
        else:
            next_tok = out.logits[:, -1, :].argmax(dim=-1)

        director.prev = h.detach()
        input_ids = torch.cat([input_ids, next_tok[:, None]], dim=1)

    return tokenizer.decode(input_ids[0])

