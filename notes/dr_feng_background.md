# Dr. Zhuo Feng — Full Background & Why He's Making You Do This

## Who He Is

**Dr. Zhuo Feng** — Professor, Department of Electrical & Computer Engineering, Stevens Institute of Technology. Director of **HUDSONLab** (Hardware Utility Design and Software Optimization Networking Lab). Currently also a **Visiting Professor at NVIDIA Research** (2025–2026 sabbatical).

**Core expertise**: Spectral graph theory. Not AI, not NLP — his career is built on making enormous graphs computationally tractable by reducing them while preserving their important mathematical properties.

---

## His Research — The One Big Idea

His entire body of work circles one obsession:

> **How do you take a huge graph and make it smaller while preserving what matters?**

Here's his algorithm lineage:

| Algorithm | Year | What it does |
|-----------|------|--------------|
| **GRASS** | 2016-2020 | Graph Spectral Sparsifier — strips away redundant edges while preserving Laplacian eigenvalues |
| **SF-GRASS** | ~2021 | Solver-Free GRASS — does the same thing but avoids expensive linear algebra |
| **inGRASS** | 2024 | Incremental sparsification — updates the sparsifier in O(log N) when the graph changes |
| **dyGRASS** | 2025 | Dynamic sparsification on GPUs — handles streaming edge insertions/deletions. **Best Paper Award nominee at ICCAD 2025** |
| **HyperEF / HyperEF 2.0** | 2022/2025 | Same idea but for hypergraphs |
| **GraphZoom** | 2020 (ICLR) | Multi-level spectral coarsening for graph embeddings — bridges into ML |
| **Topology-aware Graph Coarsening** | 2024 (NeurIPS) | Graph coarsening specifically for continual graph learning |

Notice the progression: circuit simulation → general graphs → ML/GNN applications → now LLMs.

---

## His Lab — HUDSONLab

**Current PhD students** (as of 2024-2025):
- Lizhou Qi (2024–)
- Philip Mascaro (2024–)
- Jinwen Wu (2024–)
- Hamed Sajadinia (2023–)
- Yihang Yuan (2023–)
- Soumen Sikder Shuvo (2022–)
- John Anticev (2021–)
- Wuxinlin Cheng (2021–2025, graduating)

**Past alumni** went to industry and academia. He also mentors **high school students** — his lab page lists former HS researchers who went on to Cornell, UChicago, Columbia, JHU. One high school student (Wenhao Lu) recently built the **CIPHER benchmark** for Google DeepMind's NeurIPS 2026 initiative under his mentorship.

**GitHub**: github.com/Feng-Research — public repos for inGRASS, HyperEF, SHyPar, HyperSF.

**Funding**: Multiple NSF grants for "Scalable Spectral Sparsification of Graph Laplacians."

---

## Why He's Making You Do This — The Real Picture

ChatGPT nailed it. Let me sharpen the point:

### 1. He's not assigning a lit review. He's scouting a paper topic.

His whole career is: **big graph → small graph that preserves structure**. Now he wants to apply that to LLMs:

- An LLM has a finite context window
- A large graph serialized as text can be enormous (our 200-node ER graph was 12K tokens, 500 nodes was 76K tokens)
- If you could **sparsify/coarsen the graph before putting it in the context**, you might preserve enough structure for the LLM to answer correctly while fitting in the context

**This is literally his existing research applied to a new domain.** He's not pivoting — he's extending.

### 2. The research question he's circling

> Can spectral graph sparsification (or coarsening) improve LLM performance on graph reasoning tasks by reducing token count while preserving task-relevant structure?

This is novel. Nobody has done this. The existing papers either:
- Study raw graph → LLM performance (NLGraph, GraphQA)
- Use multi-agent decomposition (GraphAgent-Reasoner)
- Use code generation to offload computation
- Fine-tune models on graph tasks (GraphWiz, G1)

**Nobody has applied spectral sparsification as a preprocessing step before LLM inference.** That's the publishable gap.

### 3. The interesting wrinkle

His sparsification methods (GRASS, etc.) preserve **spectral properties** — eigenvalues of the Laplacian, effective resistances, clustering structure. But the LLM tasks we tested require different things:

| Task | What needs to be preserved |
|------|---------------------------|
| Degree | Exact degree of queried node |
| Edge existence | Exact edge set |
| Shortest path | Path structure, distances |
| Connectivity | Connected components |
| Cycle detection | Cycle structure |

Spectral sparsification might preserve connectivity and clustering perfectly but **alter exact shortest paths**. That means the solution may need to be **task-aware** — different reductions for different queries.

This is actually a deeper research question: **which graph-reduction methods preserve which kinds of LLM graph reasoning?**

### 4. Why the "2 month lit review" makes sense from his perspective

Before he commits lab resources (PhD students, compute, his own time) to this direction, he needs someone to map out:
- What's already been tried (so he doesn't propose something that's been done)
- At what scale do LLMs actually break (so he knows where sparsification would help)
- What properties need to be preserved for each task (so he knows which of his algorithms to use)
- Whether this is publishable at a good venue

You're doing the scouting work. If the answer is "yes, there's a paper here," he'll probably want you (or a PhD student) to run the full experiment. If it's "no, someone already did this," he'll pivot.

### 5. What this means for you

**You are not just doing busywork.** You're:
- Proving you can do independent research exploration
- Mapping a publishable gap that connects his expertise to a hot topic (LLMs)
- Building the preliminary experiments that could become Section 4 of a paper

**Your Day 1 results are actually relevant to his core question**: you showed that shortest path breaks at 100 nodes, that dense 200-node graphs overflow context, and that the failure is about multi-hop traversal. These are exactly the conditions where graph sparsification could help — reduce the graph enough to fit in context while preserving path structure.

### 6. What to ask him next time you talk

Based on everything above:

1. "Is the intended direction applying spectral graph sparsification as a preprocessing step before feeding graphs to LLMs?"
2. "Should I test whether GRASS-reduced graphs maintain enough structure for LLM graph reasoning tasks?"
3. "Are you looking for a specific venue/deadline for this work?"
4. "What's the expected time commitment — is this an exploration phase or a full project?"

---

## The Bottom Line

Dr. Feng is a spectral graph theory expert who sees that LLMs choke on large graphs. His life's work is making large graphs smaller while keeping them useful. He's asking you to figure out if there's a publishable paper at the intersection of his sparsification methods and LLM graph reasoning. The lit review isn't the destination — it's reconnaissance for a potential paper.

Your benchmark results showing shortest-path failure at 100 nodes and context overflow at 200 nodes are directly relevant to his question: that's exactly where graph reduction could make a difference.
