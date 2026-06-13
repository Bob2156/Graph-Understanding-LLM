import json, sys

with open("results/benchmark_20260612_165328.json", encoding="utf-8") as f:
    d = json.load(f)

# Show the first result's full prompt and response
r = d["results"][0]
print(f"=== Task: {r['task']} | Size: {r['n_nodes']} nodes | Format: {r['format']} ===\n")
print("PROMPT SENT TO MODEL:")
print(r["prompt"])
print("\n---\nMODEL RESPONSE:")
print(r["response"])
print(f"\n---\nGROUND TRUTH: {r['ground_truth']}")
print(f"CORRECT: {r['evaluation']['correct']}")
