def show_drift(drifts):
    for i, d in enumerate(drifts):
        bar = "█" * int(d * 20)  # crude scaling
        print(f"step {i:02d} | {bar}")

