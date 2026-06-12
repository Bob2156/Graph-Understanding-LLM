# Experiment Design: LLM Graph Understanding

> Blueprint for a systematic experimental campaign evaluating how LLMs process and reason about graph-structured data represented as text.
>
> **Created**: June 2026
> **Status**: Draft — ready for team review

---

## Table of Contents

1. [What Existing Papers Found](#1-what-existing-papers-found)
2. [Our Experiment Design](#2-our-experiment-design)
3. [Predictions](#3-predictions)
4. [Novel Angles](#4-novel-angles)

---

## 1. What Existing Papers Found

### 1.1 Summary of Key Benchmarks

| Benchmark | Year | # Problems | # Tasks | Graph Sizes Tested | Models Tested |
|---|---|---|---|---|---|
| **NLGraph** | 2023 | 29,370 | 8 | 5–100+ nodes (easy/med/hard) | GPT-3.5, GPT-4 |
| **GraphQA** | 2024 | ~10,000 | 7+ | ~10–50 nodes | PaLM 2 (XXS–L), GPT-3.5/4 |
| **GraCoRe** | 2025 | 5,140 graphs | 19 | Varying (small–medium) | GPT-4, o1, open-source LLMs |
| **GraphInstruct** | 2024 | 72,000 | 9 | Small–medium | GraphWiz, GPT-4 |
| **Erdős (G1)** | 2025 | 105,000 | 50 | Real-world graphs | G1-3B, Qwen2.5-72B |
| **GraphAgent-Reasoner** | 2024 | GraphInstruct | Polynomial tasks | Up to 1,000+ nodes | Multi-agent LLMs |

### 1.2 Accuracy by Task and Model

The following table synthesizes reported results across multiple papers. Numbers represent approximate accuracy percentages. Where ranges are given, they reflect variation across graph sizes and prompting strategies.

| Task | GPT-3.5 | GPT-4 | GraphWiz-DPO | G1-3B (RL) | Difficulty |
|---|---|---|---|---|---|
| **Connectivity** | 60–80% | >90% (small) | ~85% | High | Easy |
| **Degree Counting** | 50–75% | 70–85% | ~80% | High | Easy |
| **Shortest Path** | 40–65% | 60–80% | ~70% | Moderate | Medium |
| **Cycle Detection** | 35–55% | 60–70% | ~65% | Moderate | Medium-Hard |
| **Triangle Counting** | 20–40% | 35–55% | ~50% | Moderate | Hard |
| **Bipartite Matching** | 15–30% | 25–40% | ~45% | — | Hard |
| **Maximum Flow** | 10–20% | <20% | ~40% | — | Very Hard |
| **Hamiltonian Path** | 10–20% | 15–30% | ~35% | — | Very Hard |
| **Graph Isomorphism** | 20–35% | 30–50% | — | — | Hard |
| **Average (9 tasks)** | ~30% | ~43.8% | ~65% | >72B-level | — |

> [!NOTE]
> GraphWiz-DPO (65% avg) significantly outperforms GPT-4 (43.8% avg) through instruction tuning with explicit reasoning paths and DPO alignment. G1's 3B RL-trained model outperforms Qwen2.5-72B-Instruct — a 24× larger model — demonstrating the power of RL over SFT.

### 1.3 Accuracy vs. Graph Size (The Scaling Wall)

This is the central finding across all papers: **performance degrades approximately linearly with graph size for most tasks**.

#### Approximate accuracy by node count (GPT-4, connectivity task):

| Nodes | Accuracy | Notes |
|---|---|---|
| 5 | ~95–98% | Near-ceiling for simple tasks |
| 10 | ~90–95% | Still strong |
| 20 | ~80–90% | Noticeable degradation begins |
| 30 | ~70–80% | Clear decline for medium tasks |
| 50 | ~55–70% | Significant errors emerge |
| 100 | ~40–55% | Below reliable threshold |
| 200 | ~25–40% | Near-random for complex tasks |
| 500+ | ~15–25% | Effectively broken without decomposition |

#### Approximate accuracy by node count (GPT-4, shortest path task):

| Nodes | Accuracy |
|---|---|
| 5 | ~90% |
| 10 | ~80% |
| 20 | ~60–70% |
| 50 | ~35–50% |
| 100 | ~20–30% |

> [!IMPORTANT]
> **The scaling wall** is the defining challenge. For polynomial-time tasks, GPT-4 performance drops below 50% accuracy at roughly 50–100 nodes for most tasks. For NP-hard tasks (Hamiltonian path, max flow), the wall appears even earlier (20–30 nodes). GraCoRe confirmed that simply increasing context window length does NOT solve this — longer context ≠ better graph comprehension.

#### Key scaling insights:

1. **Performance decreases approximately linearly** with graph size for most tasks (all papers agree)
2. **Even reasoning models (o1, DeepSeek-R1)** face scaling limits beyond certain complexity thresholds
3. **Context rot**: quality degrades before hitting context window limits — models accumulate noise
4. **Structural hallucinations increase with graph size** — models invent edges, forget nodes
5. **Multi-agent decomposition (GraphAgent-Reasoner)** achieves near-perfect accuracy on polynomial tasks up to 1,000+ nodes by distributing computation

### 1.4 Serialization Format Comparison

GraphQA (Fatemi et al., ICLR 2024) is the most thorough study of format effects. Key findings:

| Format | Relative Performance | Best For | Worst For | Token Efficiency |
|---|---|---|---|---|
| **Incident List** | **Highest overall** | General reasoning | — | Medium |
| **Adjacency List** | High | Neighbor lookup, local tasks | Global tasks | Medium |
| **Edge List** | Moderate | Edge-existence queries | Neighbor finding | High (most compact) |
| **Natural Language** | Moderate | LLM familiarity | Large graphs (verbose) | Low |
| **DOT Notation** | Moderate | Code-trained models | Models without code training | Medium-High |
| **Adjacency Matrix** | **Lowest typically** | Dense graph ops | Sparse/large graphs | Very Low |

#### Critical findings on serialization:

1. **No single encoding is universally best** — optimal choice depends on both task and graph structure
2. **Encoding choice can shift accuracy by 5–62%** (GraphQA's key finding)
3. **Adjacency list** tends to be the best general-purpose default
4. **Incident list** outperforms on tasks requiring explicit edge reasoning
5. **Adjacency matrix** is consistently poor due to token inefficiency and parsing difficulty
6. **Performance varies by 20–35%** when changing format on the same task (multiple studies)
7. **Graph type matters**: different formats perform differently on ER vs. BA vs. SBM graphs

### 1.5 Permutation Sensitivity

One of the most robust findings across all papers:

- **Changing node ordering** in the serialization changes LLM answers, even though the graph is identical
- **Performance can swing from 42% to 70%** solely from reordering nodes (reported in ordering studies)
- **BFS/DFS-aligned orderings** tend to outperform random orderings for path-related tasks
- **Larger models** are somewhat more robust but still sensitive
- **Fine-tuning can reduce sensitivity** to specific reorderings but may increase sensitivity to other variations
- **PEARL** (permutation-resilient LLMs) uses distributionally robust optimization to improve consistency

### 1.6 Surprising / Non-Obvious Findings

1. **CoT helps simple tasks but hurts hard ones** — Chain-of-thought can introduce reasoning errors on complex graph problems that zero-shot avoids (NLGraph)
2. **Code generation >> natural language reasoning** — Having LLMs write code to solve graph problems dramatically outperforms having them reason directly (CodeGraph)
3. **Semantic context improves performance** — Adding meaningful labels (city names vs. "Node 0") helps (GraCoRe), though it introduces world-knowledge bias
4. **Build-a-Graph prompting** (3–17% improvement) — Asking the LLM to reconstruct the graph before reasoning helps (NLGraph)
5. **RL training >> SFT** — G1's RL approach unlocks latent graph reasoning more effectively than supervised fine-tuning, and doesn't degrade general reasoning
6. **Small RL models > large base models** — G1-3B outperforms Qwen2.5-72B-Instruct on graph tasks

---

## 2. Our Experiment Design

### 2.1 Research Questions

1. **RQ1: How does accuracy scale with graph size across tasks, models, and serialization formats?** (Replication + extension of existing work with current models)
2. **RQ2: Which serialization format is optimal for each task category?** (Extending GraphQA with newer models and larger graphs)
3. **RQ3: How does graph structure (beyond size) affect performance?** (Novel: structure vs. size effects)
4. **RQ4: Can auxiliary structural information (degree sequences, hints) improve performance?** (Novel: information-theoretic approach)
5. **RQ5: How severe is permutation sensitivity, and what ordering strategies mitigate it?** (Extension of existing work)

### 2.2 Independent Variables

#### Variable 1: Graph Size (Nodes)

| Level | Nodes (n) | Expected Edges (ER, p=0.2) | Rationale |
|---|---|---|---|
| XS | 5 | ~2 | Baseline / trivial |
| S | 10 | ~9 | Easy range |
| M | 20 | ~38 | Degradation onset |
| L | 50 | ~245 | Significant decline |
| XL | 100 | ~990 | Near-failure range |
| XXL | 200 | ~3,980 | Failure range for most models |

#### Variable 2: Graph Density

| Level | Density | Edge Probability (ER) | Edges (n=50) | Description |
|---|---|---|---|---|
| Sparse | ~0.05 | p = 0.05 | ~61 | Tree-like |
| Low | ~0.10 | p = 0.10 | ~123 | Typical real-world |
| Medium | ~0.20 | p = 0.20 | ~245 | Moderate |
| High | ~0.40 | p = 0.40 | ~490 | Dense |
| Complete | 1.0 | p = 1.0 | 1,225 | Maximum density |

#### Variable 3: Graph Type

| Type | Generation Method | Properties | Rationale |
|---|---|---|---|
| **Erdős-Rényi (ER)** | G(n, p) | Random, Poisson degree dist. | Standard baseline |
| **Barabási-Albert (BA)** | Preferential attachment | Scale-free, power-law degree | Hub structure effects |
| **Stochastic Block Model (SBM)** | Community structure | Clear community boundaries | Community detection effects |
| **Regular (k-regular)** | Each node has degree k | Uniform degree | Isolate degree effects |
| **Tree** | Random spanning tree | Acyclic, connected | Simplest connected structure |
| **Star** | One hub, n-1 leaves | Extreme degree heterogeneity | Test hub-centric reasoning |

#### Variable 4: Serialization Format

| Format | Example | Token Cost (n=50, m=150) |
|---|---|---|
| **Adjacency List** | `Node 0: [1, 3, 4]` | ~450 |
| **Edge List** | `(0, 1), (0, 3), ...` | ~600 |
| **Natural Language** | `Node 0 is connected to nodes 1, 3, and 4.` | ~1,100 |
| **DOT Notation** | `graph { 0 -- 1; ... }` | ~500 |
| **Parenthetical/Dict** | `{0: {1, 3, 4}, ...}` | ~400 |

> [!NOTE]
> We deliberately exclude adjacency matrix format. Existing research consistently shows it performs worst, and it is prohibitively token-expensive for larger graphs (n² tokens). This exclusion lets us focus token budget on more informative comparisons.

#### Variable 5: Task Type

| Task | Category | Complexity | Output Type |
|---|---|---|---|
| **Edge Existence** | Local | O(1) lookup | Binary (yes/no) |
| **Degree Counting** | Local | O(deg(v)) | Integer |
| **Neighbor Listing** | Local | O(deg(v)) | Set |
| **Connectivity** | Global | O(n + m) | Binary |
| **Shortest Path** | Global | O(n + m) | Integer + path |
| **Cycle Detection** | Global | O(n + m) | Binary + cycle |
| **Triangle Counting** | Global | O(n³) | Integer |
| **Connected Components** | Global | O(n + m) | Integer |
| **Graph Diameter** | Global | O(n(n + m)) | Integer |
| **Bipartiteness** | Global | O(n + m) | Binary |

#### Variable 6: Model

| Model | Family | Parameters | Context Window | Type |
|---|---|---|---|---|
| **GPT-4o** | OpenAI | ~200B (est.) | 128K | Closed-source frontier |
| **Claude 3.5 Sonnet** | Anthropic | ~175B (est.) | 200K | Closed-source frontier |
| **Gemini 1.5 Pro** | Google | ~540B (est.) | 1M | Closed-source, largest context |
| **Llama 3.1 70B** | Meta | 70B | 128K | Open-source large |
| **Llama 3.1 8B** | Meta | 8B | 128K | Open-source small |
| **Qwen 2.5 72B** | Alibaba | 72B | 128K | Open-source large |
| **Qwen 2.5 7B** | Alibaba | 7B | 128K | Open-source small |

> [!TIP]
> The model selection spans three dimensions: (1) closed vs. open source, (2) model scale (7–540B parameters), and (3) context window size (128K–1M). This enables analysis of which factor most influences graph reasoning.

### 2.3 Dependent Variables

| Metric | Description | Applicable Tasks |
|---|---|---|
| **Exact Match Accuracy** | Binary: answer is exactly correct or not | All tasks |
| **Partial Credit (Jaccard)** | Overlap between predicted and true sets | Neighbor listing, path finding |
| **Numerical Error** | |predicted − true| for counting tasks | Degree counting, triangle counting, diameter |
| **Path Validity** | Is the returned path actually valid in the graph? | Shortest path, connectivity |
| **Response Consistency** | Same answer across permutations of the same graph | All tasks (for permutation experiments) |
| **Latency** | Time to generate response | All tasks |
| **Token Usage** | Input + output tokens consumed | All tasks |

### 2.4 Controls

| Control | Implementation |
|---|---|
| **Random seeds** | Fixed seeds for graph generation (seed = trial_number × 1000 + condition_id) |
| **System prompt** | Identical system prompt across all models: *"You are a graph analysis assistant. Answer the following question about the given graph. Provide only the answer, with no explanation."* |
| **Temperature** | 0.0 for all models (deterministic decoding) |
| **Max tokens** | 512 for output (sufficient for all tasks) |
| **Prompt template** | Standardized: `[Graph Description]\n\nQuestion: [Task-specific question]\n\nAnswer:` |
| **Multiple trials** | Each condition run with different random graphs of the same parameters |
| **Answer extraction** | Regex-based extraction of final answer from model output |
| **Ground truth** | Computed algorithmically using NetworkX for every generated graph |

### 2.5 Experimental Phases

#### Phase 1: Core Scaling Experiment (Priority: Highest)

**Goal**: Map the accuracy × graph-size curve for each task and model.

| Parameter | Values |
|---|---|
| Sizes | 5, 10, 20, 50, 100, 200 |
| Density | Fixed at p=0.2 (ER) |
| Format | Adjacency List (default) |
| Tasks | All 10 tasks |
| Models | All 7 models |
| Trials per condition | 30 |
| **Total API calls** | 6 × 10 × 7 × 30 = **12,600** |

#### Phase 2: Format Comparison (Priority: High)

**Goal**: Determine optimal serialization format for each task.

| Parameter | Values |
|---|---|
| Sizes | 10, 20, 50 (the "interesting" range) |
| Density | Fixed at p=0.2 (ER) |
| Formats | All 5 formats |
| Tasks | 6 representative tasks: Edge Existence, Degree, Connectivity, Shortest Path, Cycle Detection, Triangle Counting |
| Models | GPT-4o, Claude 3.5, Llama 3.1 70B (3 diverse models) |
| Trials per condition | 20 |
| **Total API calls** | 3 × 5 × 6 × 3 × 20 = **5,400** |

#### Phase 3: Graph Structure Effects (Priority: High)

**Goal**: Disentangle structure from size effects.

| Parameter | Values |
|---|---|
| Sizes | 20, 50 (fixed) |
| Graph Types | All 6 types |
| Density | Controlled to ~same avg degree across types |
| Format | Adjacency List |
| Tasks | Connectivity, Shortest Path, Cycle Detection, Triangle Counting |
| Models | GPT-4o, Claude 3.5, Llama 3.1 70B |
| Trials per condition | 20 |
| **Total API calls** | 2 × 6 × 4 × 3 × 20 = **2,880** |

#### Phase 4: Density Effects (Priority: Medium)

**Goal**: Understand how edge density affects performance independent of node count.

| Parameter | Values |
|---|---|
| Sizes | 20, 50 (fixed) |
| Densities | All 5 density levels |
| Format | Adjacency List |
| Tasks | Connectivity, Shortest Path, Triangle Counting, Degree Counting |
| Models | GPT-4o, Claude 3.5 |
| Trials per condition | 20 |
| **Total API calls** | 2 × 5 × 4 × 2 × 20 = **1,600** |

#### Phase 5: Permutation Sensitivity (Priority: Medium)

**Goal**: Quantify ordering effects and test mitigation strategies.

| Parameter | Values |
|---|---|
| Sizes | 10, 20, 50 |
| Orderings | Random, BFS-ordered, DFS-ordered, Degree-sorted (ascending), Degree-sorted (descending) — 5 orderings |
| Format | Adjacency List |
| Tasks | Connectivity, Shortest Path, Cycle Detection |
| Models | GPT-4o, Claude 3.5, Llama 3.1 70B |
| Trials per condition | 15 (same 15 graphs, different orderings) |
| **Total API calls** | 3 × 5 × 3 × 3 × 15 = **2,025** |

#### Phase 6: Novel Experiments (Priority: Medium-High)

See [Section 4: Novel Angles](#4-novel-angles) for details.

| Experiment | API calls (est.) |
|---|---|
| 6A: Auxiliary Information | ~2,400 |
| 6B: Graph Decomposition | ~1,800 |
| 6C: Structure Probing | ~1,200 |

### 2.6 Total Experiment Budget

| Phase | API Calls | Priority |
|---|---|---|
| Phase 1: Core Scaling | 12,600 | Highest |
| Phase 2: Format Comparison | 5,400 | High |
| Phase 3: Structure Effects | 2,880 | High |
| Phase 4: Density Effects | 1,600 | Medium |
| Phase 5: Permutation Sensitivity | 2,025 | Medium |
| Phase 6: Novel Experiments | ~5,400 | Medium-High |
| **Total** | **~29,905** | — |

### 2.7 Hypotheses

| ID | Hypothesis | Based On | Testable Via |
|---|---|---|---|
| H1 | Accuracy decreases approximately linearly with log(n) for all tasks | NLGraph, GraCoRe, scaling studies | Phase 1 |
| H2 | The accuracy-size slope is steeper for global tasks than local tasks | Task complexity hierarchy | Phase 1 |
| H3 | Adjacency list format yields highest average accuracy across tasks | GraphQA, literature consensus | Phase 2 |
| H4 | Format effect magnitude (best minus worst) exceeds 20% for most tasks | GraphQA (5–62% range) | Phase 2 |
| H5 | Barabási-Albert graphs are harder than ER graphs of the same size (due to hub nodes) | Novel — degree heterogeneity hypothesis | Phase 3 |
| H6 | Tree graphs are easiest (unique paths, no cycles) | Structural simplicity | Phase 3 |
| H7 | Higher density improves connectivity/cycle detection but hurts path finding | More edges = more information but harder optimization | Phase 4 |
| H8 | BFS-ordered serialization outperforms random ordering for path tasks | Ordering alignment hypothesis | Phase 5 |
| H9 | Permutation sensitivity decreases with model size | Larger models = more robust | Phase 5 |
| H10 | Providing degree sequence as auxiliary information improves triangle counting by >15% | Structural hint hypothesis | Phase 6A |
| H11 | Graph decomposition into communities improves accuracy on large (n>50) graphs | GraphAgent-Reasoner principle | Phase 6B |

### 2.8 Sample Size Justification

With 20–30 trials per condition:
- **Power analysis**: At α=0.05 and power=0.80, n=25 trials can detect a 15 percentage-point difference between conditions (assuming σ ≈ 0.25 for binary accuracy). This is sufficient given the literature reports effects of 20–60%.
- **Variance**: Binary accuracy has maximum variance at p=0.5 (σ²=0.25). With n=30 trials, the 95% CI width is approximately ±18%, narrowing to ±12% for non-boundary accuracies.
- **Cost-accuracy tradeoff**: 30 trials balances statistical reliability against API cost.

### 2.9 Statistical Analysis Plan

#### Primary Analysis

1. **Multi-factor ANOVA** for each task:
   - Factors: graph_size × format × model (Phase 1+2 combined)
   - Dependent variable: accuracy (binary, but with sufficient n, ANOVA is robust)
   - Post-hoc: Tukey's HSD for pairwise comparisons

2. **Logistic regression** for accuracy prediction:
   - `logit(accuracy) ~ log(n) + density + format + model + task + interactions`
   - This captures the scaling curve shape and interaction effects

3. **Cochran's Q test** for permutation sensitivity:
   - Test whether the same graph produces significantly different accuracy across orderings

#### Secondary Analysis

4. **Effect size estimation** (Cohen's d) for format comparisons
5. **Regression analysis** of accuracy vs. graph properties:
   - `accuracy ~ n + m + avg_degree + diameter + clustering_coefficient + model`
6. **Bootstrap confidence intervals** for all accuracy estimates
7. **Friedman test** for ranking formats across tasks (non-parametric)

#### Visualization Plan

- **Scaling curves**: Accuracy (y) vs. nodes (x), faceted by task, colored by model
- **Heatmaps**: Task × format accuracy matrix for each model
- **Interaction plots**: Size × format interaction effects
- **Permutation variance**: Box plots of accuracy across orderings per condition
- **Radar charts**: Model capability profiles across tasks

---

## 3. Predictions

### 3.1 Scaling Wall Predictions (Graph Size at 50% Accuracy)

Based on the literature, we predict the following approximate node counts at which each model drops to ~50% accuracy:

| Task | GPT-4o | Claude 3.5 | Gemini 1.5 Pro | Llama 70B | Llama 8B |
|---|---|---|---|---|---|
| Edge Existence | >200 | >200 | >200 | 150 | 50 |
| Degree Counting | 100 | 100 | 100 | 50 | 20 |
| Connectivity | 80 | 80 | 100 | 40 | 15 |
| Shortest Path | 40 | 40 | 50 | 25 | 10 |
| Cycle Detection | 35 | 35 | 40 | 20 | 10 |
| Triangle Counting | 25 | 25 | 30 | 15 | 8 |
| Connected Components | 50 | 50 | 60 | 30 | 15 |
| Graph Diameter | 30 | 30 | 40 | 20 | 10 |
| Bipartiteness | 40 | 40 | 50 | 25 | 10 |

> [!NOTE]
> These predictions are extrapolated from NLGraph and GraphQA trends and should be treated as hypotheses to be tested. Gemini 1.5 Pro may have an advantage on larger graphs due to its 1M context window, though GraCoRe suggests context length alone doesn't predict comprehension.

### 3.2 Serialization Format Predictions

| Rank | Format | Predicted Best For | Predicted Worst For |
|---|---|---|---|
| 1 | **Adjacency List** | Local tasks (degree, neighbors) | — |
| 2 | **Parenthetical/Dict** | Code-trained models | Non-technical tasks |
| 3 | **Edge List** | Edge-centric tasks, large sparse graphs | Neighbor queries |
| 4 | **Natural Language** | Smallest graphs, simple questions | Large graphs (token cost) |
| 5 | **DOT Notation** | Models with code training data | Models trained primarily on prose |

**Predicted format effect magnitude**: Average 25% accuracy difference between best and worst formats, with task-dependent variation from 10% (edge existence) to 45% (triangle counting).

### 3.3 Task Difficulty Predictions (Easiest to Hardest)

| Rank | Task | Predicted Avg Accuracy (n=50, GPT-4o) | Reasoning |
|---|---|---|---|
| 1 | Edge Existence | ~90% | Simple lookup |
| 2 | Degree Counting | ~75% | Local counting |
| 3 | Neighbor Listing | ~70% | Local retrieval |
| 4 | Connectivity | ~65% | BFS/DFS simulation |
| 5 | Bipartiteness | ~55% | Two-coloring |
| 6 | Connected Components | ~50% | Multiple BFS |
| 7 | Shortest Path | ~45% | Optimal path selection |
| 8 | Cycle Detection | ~40% | State tracking |
| 9 | Graph Diameter | ~30% | All-pairs shortest paths |
| 10 | Triangle Counting | ~25% | Combinatorial enumeration |

### 3.4 Model Ranking Predictions

For overall graph reasoning ability (averaged across tasks):

1. **GPT-4o** ≈ **Claude 3.5 Sonnet** (top tier, ~55–65% avg across tasks and sizes)
2. **Gemini 1.5 Pro** (slightly below, ~50–60%, with advantage on very large graphs)
3. **Llama 3.1 70B** ≈ **Qwen 2.5 72B** (~40–50%)
4. **Llama 3.1 8B** ≈ **Qwen 2.5 7B** (~25–35%)

**Predicted gap between closed and open source**: ~15–20 percentage points for frontier vs. 70B open-source, ~30–35 percentage points for frontier vs. 7B open-source.

### 3.5 Interaction Effect Predictions

- **Size × Task interaction**: Global tasks degrade faster than local tasks as size increases (H2)
- **Format × Task interaction**: Edge list will outperform adjacency list on edge-centric tasks but underperform on node-centric tasks
- **Size × Format interaction**: Natural language format will degrade fastest with size (due to token cost), while edge list will degrade slowest
- **Density × Task interaction**: Connectivity detection gets easier with density (more paths), but path-finding gets harder (more choices to evaluate)

---

## 4. Novel Angles

### 4.1 Experiment 6A: Auxiliary Structural Information

**Research Question**: Does providing pre-computed structural information in the prompt improve LLM graph reasoning accuracy?

**Motivation**: If LLMs struggle because they can't extract structural properties from raw serializations, providing those properties as "hints" should help. This tests whether the bottleneck is *information extraction* vs. *reasoning*.

#### Design

| Condition | Information Provided |
|---|---|
| **Baseline** | Graph serialization only |
| **+Degree sequence** | Graph + "The degree sequence is: [2, 3, 3, 4, ...]" |
| **+Summary statistics** | Graph + "The graph has 50 nodes, 120 edges, avg degree 4.8, diameter 5, 0 triangles." |
| **+Neighborhood summaries** | Graph + "Node 0 has 3 neighbors and is in a cluster with nodes {1, 2, 5}." |
| **+Partial BFS** | Graph + BFS tree from relevant source node |

| Parameter | Value |
|---|---|
| Sizes | 20, 50, 100 |
| Format | Adjacency List |
| Tasks | Shortest Path, Triangle Counting, Connectivity, Graph Diameter |
| Models | GPT-4o, Claude 3.5, Llama 70B |
| Trials | 20 per condition |
| **Total** | 3 × 5 × 4 × 3 × 20 = **3,600** API calls |

**Hypothesis (H10)**: Providing degree sequence will improve triangle counting accuracy by >15%. Providing partial BFS will improve connectivity and shortest path by >10%.

**Why this is novel**: Existing papers test raw serialization formats but don't systematically test the effect of *augmenting* the serialization with pre-computed structural features. This creates a bridge between pure text-based approaches and tool-augmented approaches, identifying exactly which structural features LLMs can't extract on their own.

### 4.2 Experiment 6B: Graph Decomposition

**Research Question**: Does decomposing a large graph into overlapping subgraphs improve accuracy compared to presenting the full graph?

**Motivation**: GraphAgent-Reasoner shows that distributed reasoning helps, but it uses a complex multi-agent setup. We test a simpler approach: partition the graph into communities, serialize each subgraph separately, and ask the LLM to reason about each part before combining answers.

#### Design

| Condition | Method |
|---|---|
| **Full graph** | Entire graph in one prompt |
| **Community decomposition** | Louvain communities → separate prompts per community → aggregation prompt |
| **BFS neighborhoods** | k-hop neighborhoods centered on query-relevant nodes |
| **Random partition** | Random node partitions (control for decomposition quality) |

| Parameter | Value |
|---|---|
| Sizes | 50, 100, 200 (large enough to benefit from decomposition) |
| Graph Types | ER, SBM (SBM has natural community structure) |
| Format | Adjacency List |
| Tasks | Connectivity, Shortest Path, Connected Components |
| Models | GPT-4o, Claude 3.5 |
| Trials | 15 per condition |
| **Total** | 3 × 2 × 4 × 3 × 2 × 15 = **2,160** API calls |

**Hypothesis (H11)**: Community decomposition improves accuracy by >20% on graphs with n≥100 nodes compared to full-graph presentation, with larger gains on SBM graphs (which have natural community structure) than ER graphs.

**Why this is novel**: While GraphAgent-Reasoner uses per-node agents (expensive), we test whether simple community-based decomposition with a single LLM achieves meaningful accuracy gains. This is more practical and reveals whether the decomposition principle itself (vs. the multi-agent architecture) drives the improvement.

### 4.3 Experiment 6C: Structure Probing — Does Graph Topology Predict Difficulty?

**Research Question**: Holding size constant, which topological properties of a graph make it harder for LLMs to reason about?

**Motivation**: All existing papers treat graph size as the primary difficulty driver. But two 50-node graphs can have very different structures. We systematically vary structural properties while keeping n fixed to isolate their effects.

#### Design

Generate 50-node graphs with controlled variation in:

| Property | Low | Medium | High |
|---|---|---|---|
| **Clustering coefficient** | 0.0–0.1 | 0.2–0.4 | 0.5–0.8 |
| **Diameter** | 2–3 | 5–7 | 10–15 |
| **Degree variance** | σ < 1 (regular) | σ ≈ 2–3 | σ > 5 (heterogeneous) |
| **Community structure** (modularity) | Q < 0.1 (none) | Q ≈ 0.3 | Q > 0.5 (strong) |

| Parameter | Value |
|---|---|
| Size | 50 (fixed) |
| Density | ~0.2 (fixed, adjusted per structure) |
| Format | Adjacency List |
| Tasks | Connectivity, Shortest Path, Triangle Counting |
| Models | GPT-4o, Claude 3.5 |
| Trials | 20 per condition (4 properties × 3 levels × 3 tasks × 2 models) |
| **Total** | 4 × 3 × 3 × 2 × 20 = **1,440** API calls |

**Hypotheses**:
- High degree variance (hub nodes) makes tasks harder because the LLM must track high-degree nodes with long adjacency lists
- High diameter makes shortest path harder (longer paths to track)
- High clustering makes triangle counting easier (more triangles to find, pattern is more regular) but cycle detection harder (more cycles to enumerate)
- Strong community structure helps connectivity (communities are easy to reason about) but makes cross-community paths harder

**Why this is novel**: This is the first systematic study isolating topological properties from graph size. Existing papers conflate these by varying size, which simultaneously changes all structural properties. Our approach uses graph generation algorithms that control for specific properties (e.g., Watts-Strogatz for clustering, configuration model for degree distribution).

### 4.4 Summary of Novel Contributions

| Experiment | Key Innovation | Expected Impact |
|---|---|---|
| **6A: Auxiliary Info** | First systematic test of structural hints | Quantifies the extraction vs. reasoning bottleneck |
| **6B: Decomposition** | Simple decomposition (no multi-agent) | Practical scaling strategy for single-LLM setups |
| **6C: Structure Probing** | Isolating topology from size | Identifies which structural features predict difficulty |

---

## Appendix A: Implementation Details

### A.1 Graph Generation

```python
# All graphs generated using NetworkX with fixed seeds
import networkx as nx
import random

def generate_graph(graph_type, n, seed, **params):
    """Generate a graph with reproducible randomness."""
    rng = random.Random(seed)
    
    if graph_type == "ER":
        return nx.erdos_renyi_graph(n, params.get("p", 0.2), seed=seed)
    elif graph_type == "BA":
        return nx.barabasi_albert_graph(n, params.get("m", 3), seed=seed)
    elif graph_type == "SBM":
        sizes = [n // 3] * 3  # 3 communities
        p_matrix = [[params.get("p_in", 0.4), 0.05, 0.05],
                     [0.05, params.get("p_in", 0.4), 0.05],
                     [0.05, 0.05, params.get("p_in", 0.4)]]
        return nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    elif graph_type == "regular":
        return nx.random_regular_graph(params.get("d", 4), n, seed=seed)
    elif graph_type == "tree":
        return nx.random_tree(n, seed=seed)
    elif graph_type == "star":
        return nx.star_graph(n - 1)
```

### A.2 Serialization Functions

```python
def serialize_adjacency_list(G):
    """Adjacency list format."""
    lines = []
    for node in sorted(G.nodes()):
        neighbors = sorted(G.neighbors(node))
        lines.append(f"Node {node}: [{', '.join(str(n) for n in neighbors)}]")
    return "\n".join(lines)

def serialize_edge_list(G):
    """Edge list format."""
    edges = sorted(G.edges())
    return "\n".join(f"({u}, {v})" for u, v in edges)

def serialize_natural_language(G):
    """Natural language format."""
    lines = [f"This graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."]
    for node in sorted(G.nodes()):
        neighbors = sorted(G.neighbors(node))
        if len(neighbors) == 0:
            lines.append(f"Node {node} has no connections.")
        elif len(neighbors) == 1:
            lines.append(f"Node {node} is connected to node {neighbors[0]}.")
        else:
            neighbor_str = ", ".join(str(n) for n in neighbors[:-1]) + f", and {neighbors[-1]}"
            lines.append(f"Node {node} is connected to nodes {neighbor_str}.")
    return "\n".join(lines)

def serialize_dot(G):
    """DOT notation format."""
    lines = ["graph G {"]
    for u, v in sorted(G.edges()):
        lines.append(f"    {u} -- {v};")
    lines.append("}")
    return "\n".join(lines)

def serialize_parenthetical(G):
    """Python dict-style format."""
    adj = {}
    for node in sorted(G.nodes()):
        adj[node] = sorted(list(G.neighbors(node)))
    return str(adj)
```

### A.3 Prompt Template

```
[SYSTEM]
You are a graph analysis assistant. Answer the following question about the given graph. Provide only the answer, with no explanation unless specifically asked.

[USER]
The following describes an undirected graph:

{serialized_graph}

Question: {task_question}

Answer:
```

### A.4 Task Question Templates

| Task | Question Template |
|---|---|
| Edge Existence | "Is there an edge between node {u} and node {v}? Answer yes or no." |
| Degree Counting | "What is the degree of node {v}? Answer with a number." |
| Neighbor Listing | "List all neighbors of node {v}. Answer with a comma-separated list of node numbers." |
| Connectivity | "Is there a path between node {u} and node {v}? Answer yes or no." |
| Shortest Path | "What is the length of the shortest path between node {u} and node {v}? Answer with a number." |
| Cycle Detection | "Does this graph contain a cycle? Answer yes or no." |
| Triangle Counting | "How many triangles are in this graph? Answer with a number." |
| Connected Components | "How many connected components does this graph have? Answer with a number." |
| Graph Diameter | "What is the diameter of this graph? Answer with a number." |
| Bipartiteness | "Is this graph bipartite? Answer yes or no." |

### A.5 Answer Extraction and Grading

```python
import re

def extract_answer(response, task_type):
    """Extract the answer from model response."""
    response = response.strip().lower()
    
    if task_type in ["edge_existence", "connectivity", "cycle_detection", "bipartiteness"]:
        # Binary tasks
        if "yes" in response:
            return True
        elif "no" in response:
            return False
        return None  # Failed to extract
    
    elif task_type in ["degree", "shortest_path", "triangle_count", 
                        "components", "diameter"]:
        # Numeric tasks
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[0])
        return None
    
    elif task_type == "neighbor_listing":
        # Set tasks
        numbers = re.findall(r'\d+', response)
        return set(int(n) for n in numbers)
    
    return None

def grade_answer(predicted, ground_truth, task_type):
    """Grade the answer."""
    if predicted is None:
        return {"exact_match": 0, "partial_credit": 0}
    
    if task_type == "neighbor_listing":
        pred_set = set(predicted) if isinstance(predicted, set) else {predicted}
        true_set = set(ground_truth) if isinstance(ground_truth, set) else {ground_truth}
        jaccard = len(pred_set & true_set) / len(pred_set | true_set) if pred_set | true_set else 1
        return {
            "exact_match": 1 if pred_set == true_set else 0,
            "partial_credit": jaccard
        }
    else:
        return {
            "exact_match": 1 if predicted == ground_truth else 0,
            "partial_credit": 1 if predicted == ground_truth else 0
        }
```

---

## Appendix B: Risk Mitigation

| Risk | Mitigation |
|---|---|
| **API rate limits** | Implement exponential backoff; stagger experiments across days |
| **API cost overrun** | Phase experiments by priority; set budget caps per phase |
| **Model API changes** | Record exact model version strings; timestamp all results |
| **Non-deterministic outputs** | Use temperature=0; record all raw responses for reproducibility |
| **Answer extraction failures** | Log extraction failures separately; manually review edge cases |
| **Graph generation bias** | Use multiple random seeds; verify graph property distributions |
| **Multiple comparisons** | Apply Bonferroni or Benjamini-Hochberg correction for multiple hypothesis tests |

---

## Appendix C: Timeline

| Week | Phase | Deliverable |
|---|---|---|
| 1 | Infrastructure | Graph generation pipeline, serialization functions, API wrappers |
| 2 | Phase 1a | Core scaling experiment (small/medium models) |
| 3 | Phase 1b | Core scaling experiment (frontier models) |
| 4 | Phase 2 | Format comparison experiment |
| 5 | Phase 3 | Graph structure effects |
| 6 | Phase 4+5 | Density effects + permutation sensitivity |
| 7 | Phase 6 | Novel experiments |
| 8 | Analysis | Statistical analysis, visualization, paper draft |

---

## References

- Wang, H., et al. "Can Language Models Solve Graph Problems in Natural Language?" NeurIPS 2023 (NLGraph)
- Fatemi, B., et al. "Talk like a Graph: Encoding Graphs for Large Language Models." ICLR 2024 (GraphQA)
- Yuan, Z., et al. "GraCoRe: Evaluating Graph Comprehension and Complex Reasoning in LLMs." COLING 2025
- Chen, N., et al. "GraphWiz: An Instruction-Following Language Model for Graph Computational Problems." KDD 2024
- G1 Authors. "G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning." arXiv:2505.18499, 2025
- Hu, Y., et al. "Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents." arXiv:2410.05130, 2024
- Jin, B., et al. "Graph Chain-of-Thought: Augmenting LLMs by Reasoning on Graphs." ACL Findings 2024
- Jin, B., et al. "Large Language Models on Graphs: A Comprehensive Survey." IEEE TKDE 2024
