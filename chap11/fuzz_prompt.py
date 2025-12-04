import random

def fuzz_prompt(prompt):
    variants = set()
    tokens = prompt.split()

    # 1. Random swaps
    if len(tokens) > 3:
        i, j = random.sample(range(len(tokens)), 2)
        swapped = tokens[:]
        swapped[i], swapped[j] = swapped[j], swapped[i]
        variants.add(" ".join(swapped))

    # 2. Random deletion
    if len(tokens) > 4:
        k = random.randrange(len(tokens))
        deleted = tokens[:k] + tokens[k+1:]
        variants.add(" ".join(deleted))

    # 3. Punctuation perturbation
    variants.add(prompt + "?")
    variants.add(prompt + "...")
    variants.add(prompt.replace(",", ""))

    # 4. Synonym injection (toy example)
    synonyms = {"explain": "describe", "why": "how", "network": "model"}
    toks = [synonyms.get(t.lower(), t) for t in tokens]
    variants.add(" ".join(toks))

    return list(variants)

