def run_fuzzer(model, tokenizer, prompt):
    variants = fuzz_prompt(prompt)
    results = {}

    for v in variants:
        trace = trace_reasoning(model, tokenizer, v, max_tokens=20)
        results[v] = trace

    return results

