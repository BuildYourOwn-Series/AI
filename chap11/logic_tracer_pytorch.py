import torch

# Load the model (TinyTransformer, or any decoder-only model)
model = TinyTransformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4
)

# Load trained weights
state = torch.load("tiny_transformer.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "Explain why neural networks hallucinate."

# Trace the logits step-by-step
logits_trace = trace_logits(model, tokenizer, prompt, max_tokens=20)

# Examine the first few steps
print("=== STEP 0 ===")
inspect_step(logits_trace[0], tokenizer)

print("\n=== STEP 1 ===")
inspect_step(logits_trace[1], tokenizer)

print("\n=== STEP 2 ===")
inspect_step(logits_trace[2], tokenizer)

