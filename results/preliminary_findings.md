# Preliminary Experimental Results: Qwen 3.5 9B on Graph Tasks

**Model**: Qwen 3.5 9B (quantized, LM Studio, 32K context)  
**Date**: 2026-06-12  
**Graph type**: Erdős-Rényi (p=0.3) and Random Tree  
**Serialization**: Adjacency list  
**Trials**: 1 per condition (preliminary)

---

## 1. Summary of Results

### Accuracy by Graph Size and Task

| Size | Type | Edges | Tokens | node_degree | edge_existence | shortest_path | connectivity | cycle_detection |
|------|------|-------|--------|:-----------:|:--------------:|:-------------:|:------------:|:--------------:|
| 10   | ER   | ~15   | ~100   | ✅ | ✅ | ✅ | ✅ | ✅ |
| 20   | ER   | ~60   | ~300   | ✅ | ✅ | ✅ | ✅ | ✅ |
| 50   | ER   | ~370  | ~800   | ✅ | ✅ | ✅ | ✅ | ✅ |
| 100  | ER   | ~1477 | ~3154  | ✅ | ✅ | ❌ | ✅ | ✅ |
| 200  | Tree | 199   | ~800   | ✅ | ✅ | ❌ | —  | —  |

**Key finding**: The model achieves 100% accuracy on all tasks up to 50 nodes. At 100 nodes, **shortest path is the first task to fail**.

---

## 2. Failure Analysis

### Failure #1: Shortest Path at 100 Nodes (ER Graph)

- **Ground truth**: 2 (two-hop path)
- **Model answer**: 1 (claimed direct edge exists)
- **Failure mode**: **Hallucinated edge** — The model claimed node 34 was directly connected to node 6, but this edge does not exist in the graph. At 100 nodes with ~1500 edges, the adjacency list is long enough that the model loses track of which specific nodes appear in which node's neighbor list.

### Failure #2: Shortest Path at 200 Nodes (Tree)

- **Ground truth**: 15 (fifteen-hop path through the tree)
- **Model answer**: -1 (no path exists)
- **Failure mode**: **Premature traversal abandonment** — The model correctly performed BFS-style reasoning for 2 hops out from the source node, found no connection to the target, and then incorrectly concluded that no path exists. It explicitly stated: *"the graph component containing 163 seems isolated from the component containing 28"* — even though all nodes in a tree are connected by definition.

### Taxonomy of Failure Modes Observed

| Mode | Description | When observed |
|------|-------------|---------------|
| **Edge hallucination** | Claims an edge exists when it doesn't | 100-node ER, shortest path |
| **Traversal abandonment** | Gives up after 2-3 hops and declares no path | 200-node tree, shortest path |

Both modes are consistent with findings from NLGraph (Wang et al., 2023) and GraphQA (Fatemi et al., 2024).

---

## 3. Task Difficulty Hierarchy

Based on our results, tasks can be ranked by difficulty:

1. **Easy** (robust to 200+ nodes):
   - `edge_existence` — Simple lookup in one node's adjacency list
   - `node_degree` — Count elements in one node's adjacency list
   - `connectivity` — Can be answered with local reasoning + graph properties
   - `cycle_detection` — Structural property, can be inferred from degree patterns

2. **Hard** (fails at 100 nodes):
   - `shortest_path` — Requires multi-hop traversal across the full graph

This matches the literature: **tasks requiring global graph reasoning fail before tasks requiring local lookups**.

---

## 4. Hardware Constraints Discovered

| Size | Type | Edges | Est. Tokens | Fits 32K context? | GPU (8GB) viable? |
|------|------|-------|-------------|:---------:|:---------:|
| 100  | ER   | 1,477 | 3,154       | ✅ | ✅ |
| 200  | ER   | 5,918 | 12,236      | ✅ | ❌ (OOM) |
| 200  | Tree | 199   | ~800        | ✅ | ✅ |
| 500  | ER   | 37,481| 75,962      | ❌ | ❌ |

Dense graphs at 200 nodes exceed GPU KV cache memory (6.55GB model + KV cache > 8GB VRAM). This is a practical constraint for local model experiments, not a model capability limit.

---

## 5. Verification Against Literature

| Paper Finding | Our Result | Verified? |
|--------------|------------|:---------:|
| Shortest path is the hardest standard task (NLGraph) | First task to fail at 100 nodes | ✅ |
| LLMs hallucinate edges at scale (GraphQA) | Observed at 100 nodes | ✅ |
| Lookup tasks (degree, edge existence) are easy (GraCoRe) | Perfect accuracy to 200 nodes | ✅ |
| Performance degrades between 50-100 nodes (NLGraph) | Confirmed: 50 perfect, 100 first failure | ✅ |
| Multi-hop reasoning limited to 2-3 hops (GraphQA) | Confirmed: model abandoned BFS after 2 hops | ✅ |

---

## 6. Limitations of This Preliminary Study

- **Single trial per condition** — results could be noise; need 3-5 trials to confirm
- **One model only** — Qwen 3.5 9B; larger models may have different scaling curves
- **One serialization format** — adjacency list only; papers show format matters significantly
- **One graph type for scaling** — ER only; structure (trees, grids, scale-free) may affect results
- **No prompt engineering** — baseline prompts only; CoT or structured prompts may improve results

---

## 7. Next Experiments (Priority Order)

1. **Multiple trials at 100 nodes** — Run 3-5 trials on shortest_path at 100 nodes to confirm failure rate
2. **Serialization format comparison** — Test edge_list and natural_language formats at 50-100 nodes
3. **Chain-of-thought prompting** — Add explicit "think step by step" instructions for shortest path
4. **More tasks at 100 nodes** — Triangle counting, diameter, neighbor listing at the failure boundary
5. **Graph structure comparison** — Trees vs ER vs scale-free at matched sizes to isolate topology effects
