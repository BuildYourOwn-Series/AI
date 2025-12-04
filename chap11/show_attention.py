import matplotlib.pyplot as plt

def show_attention(attn, input_ids, tokenizer, layer=0, head=0):
    A = attn[layer][head]           # (seq_len, seq_len)
    tokens = [tokenizer.decode([t]) for t in input_ids[0]]

    plt.figure(figsize=(6, 6))
    plt.imshow(A, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.title(f"Layer {layer}, Head {head}")
    plt.tight_layout()
    plt.show()

