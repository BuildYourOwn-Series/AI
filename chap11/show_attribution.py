def show_attribution(scores, input_ids, tokenizer):
    tokens = [tokenizer.decode([t]) for t in input_ids]
    scores = scores / scores.max()  # normalize for display

    for tok, sc in zip(tokens, scores):
        bar = "█" * int(sc.item() * 20)
        print(f"{tok:>12} | {bar}")

