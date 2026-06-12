import json, sys

with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)

for r in d["results"]:
    task = r["task"]
    ev = r["evaluation"]
    if ev and not ev["correct"]:
        print(f"=== {task} ===")
        print(f"Ground truth: {r['ground_truth']}")
        print(f"Model parsed: {ev.get('model_answer', 'N/A')}")
        print(f"Full response:\n{r['response']}")
        print()
