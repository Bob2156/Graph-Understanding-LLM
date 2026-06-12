import json, sys

with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)

print(f"Total results: {d['metadata']['total_results']}\n")
for r in d["results"]:
    task = r["task"]
    ev = r["evaluation"]
    status = r["status"]
    if ev:
        correct = ev["correct"]
        gt = r["ground_truth"]
        resp = r["response"][:120].replace("\n", " ") if r["response"] else ""
        mark = "PASS" if correct else "FAIL"
        print(f"[{mark}] {task:20s} | ground_truth={gt} | response={resp}")
    else:
        print(f"[ERR ] {task:20s} | status={status}")
