"""
Smoke Test
==========

Quick verification that the entire experiment infrastructure works:
1. Generate a small graph (10 nodes)
2. Serialize it in all formats
3. Solve all tasks and print results
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.graphs.graph_generator import generate_erdos_renyi, GRAPH_TYPES
from experiments.graphs.graph_serializer import (
    serialize_graph,
    get_text_stats,
    compute_ground_truth,
    SERIALIZATION_FORMATS,
)
from experiments.benchmarks.tasks import TASKS, generate_task_prompt, evaluate_response


def run_smoke_test() -> str:
    """Run a comprehensive smoke test and return the output as a string."""
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("SMOKE TEST — Graph Experiment Infrastructure")
    log("=" * 70)

    # 1. Generate graph
    log("\n1. GRAPH GENERATION")
    log("-" * 40)
    G = generate_erdos_renyi(n_nodes=10, p=0.3, seed=42)
    log(f"   Generated Erdős-Rényi graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    log(f"   Available graph types: {list(GRAPH_TYPES.keys())}")

    # Generate one of each type at n=10
    log("\n   All graph types at n=10:")
    for gtype, gen_fn in GRAPH_TYPES.items():
        import inspect
        sig = inspect.signature(gen_fn)
        kwargs = {"seed": 42} if "seed" in sig.parameters else {}
        g = gen_fn(n_nodes=10, **kwargs)
        log(f"     {gtype:20s}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # 2. Serialization
    log("\n2. SERIALIZATION FORMATS")
    log("-" * 40)
    stats = get_text_stats(G)
    for fmt_name, info in stats.items():
        log(f"\n   === {fmt_name} ({info['token_count']} tokens, {info['char_count']} chars) ===")
        text_preview = info["text"][:200]
        if len(info["text"]) > 200:
            text_preview += "\n   ..."
        for line in text_preview.split("\n"):
            log(f"   {line}")

    # 3. Ground truth
    log("\n3. GROUND TRUTH STATISTICS")
    log("-" * 40)
    gt = compute_ground_truth(G)
    for key, value in gt.items():
        log(f"   {key:20s}: {value}")

    # 4. Tasks
    log("\n4. TASK EXECUTION")
    log("-" * 40)
    graph_text = serialize_graph(G, "adjacency_list")
    for task_name in TASKS:
        prompt_data = generate_task_prompt(G, graph_text, task_name, seed=42)
        log(f"\n   Task: {task_name}")
        log(f"   Params: {prompt_data['params']}")
        log(f"   Ground truth: {prompt_data['ground_truth']}")

        # Simulate a correct LLM response
        gt_val = prompt_data["ground_truth"]
        if isinstance(gt_val, bool):
            fake_response = "Yes" if gt_val else "No"
        elif isinstance(gt_val, list):
            fake_response = ", ".join(str(x) for x in gt_val)
        else:
            fake_response = str(gt_val)

        evaluation = evaluate_response(task_name, fake_response, gt_val)
        log(f"   Simulated response: '{fake_response}'")
        log(f"   Evaluation: correct={evaluation['correct']}, parsed={evaluation['parsed']}")

    log("\n" + "=" * 70)
    log("SMOKE TEST PASSED ✓")
    log("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    output = run_smoke_test()

    # Save output
    results_dir = Path(_PROJECT_ROOT) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "smoke_test_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nOutput saved to: {output_path}")
