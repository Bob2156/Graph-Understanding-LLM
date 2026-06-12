# Paper Summaries: LLMs and Graph Understanding

> Compiled June 2026. Covers key papers (2023–2025) on LLMs processing, reasoning about, and understanding graph-structured data represented as text.

---

## 1. Foundational Benchmarks & Evaluations

### Can Language Models Solve Graph Problems in Natural Language? (2023)
- **Authors**: Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov
- **Venue/Link**: NeurIPS 2023 (Spotlight) — arXiv:2305.10037
- **Key idea**: Introduces **NLGraph**, a benchmark with 29,370 problems spanning 8 graph reasoning tasks (connectivity, shortest path, maximum flow, bipartite matching, Hamiltonian path, GNN simulation, etc.) to evaluate LLM graph reasoning via natural language.
- **Relevance**: Foundational benchmark for our research — directly tests whether LLMs can solve graph problems when graphs are serialized as text. Establishes baseline capabilities and failure modes.
- **Graph serialization method**: Edge-list text descriptions in natural language (e.g., "Node 0 is connected to Node 1, Node 2, …")
- **Results**: LLMs show preliminary graph-reasoning ability (above random baselines). GPT-4 significantly outperforms GPT-3. Chain-of-thought prompting helps on simpler tasks but effectiveness diminishes on complex problems. Two proposed prompting strategies (Build-a-Graph, Algorithmic Prompting) improve accuracy by 3–17%.
- **Limitations**: Performance brittle and inconsistent — models often rely on spurious correlations rather than genuine structural reasoning. Complex tasks (max flow, Hamiltonian path) remain largely unsolved.
- **Notes**: Seminal paper that catalyzed the field. The NLGraph benchmark is widely used in subsequent work. Code at github.com/Arthur-Heng/NLGraph.

---

### Talk like a Graph: Encoding Graphs for Large Language Models (2024)
- **Authors**: Bahare Fatemi, Jonathan Halcrow, Bryan Perozzi (Google Research)
- **Venue/Link**: ICLR 2024 — arXiv:2310.04560
- **Key idea**: Comprehensive study of how different **graph-to-text encoding methods** affect LLM reasoning. Introduces the **GraphQA** benchmark to systematically evaluate encoding × task × graph-structure interactions.
- **Relevance**: **Directly central** to our research question — provides the most thorough analysis of how serialization format impacts LLM graph understanding. Shows that encoding choice can shift accuracy by up to 61.8%.
- **Graph serialization method**: Tests multiple formats systematically: adjacency list, edge list, incident edges, GraphML, natural language descriptions, and others.
- **Results**: (1) No single encoding is universally best — optimal choice depends on both task and graph structure. (2) Encoding can improve performance by 4.8%–61.8%. (3) LLMs are sensitive to node ordering and edge presentation order. (4) Certain encodings like incident edges work better for local tasks while adjacency matrices help global tasks.
- **Limitations**: Primarily tests GPT-3.5/4 and PaLM-2; may not generalize to other model families. Limited to relatively small graphs (dozens of nodes).
- **Notes**: The GraphQA benchmark is open-sourced and has become a standard evaluation tool. Key reference for any encoding/serialization study.

---

### GraCoRe: Evaluating Graph Comprehension and Complex Reasoning in LLMs (2025)
- **Authors**: Zike Yuan, Ming Liu, et al.
- **Venue/Link**: COLING 2025 — arXiv:2407.02936
- **Key idea**: A **hierarchical benchmark** with a three-tier taxonomy covering 10 capability areas and 19 tasks, evaluating both graph comprehension and complex reasoning across pure and heterogeneous graphs.
- **Relevance**: Latest comprehensive benchmark — tests both structural understanding and logical reasoning, revealing that these are often in tension for LLMs.
- **Graph serialization method**: Text-based graph descriptions with varying levels of semantic enrichment.
- **Results**: (1) Semantic context improves reasoning performance. (2) Node ordering in input significantly impacts results. (3) Longer context windows do NOT inherently improve graph comprehension. (4) OpenAI's o1 shows advanced capabilities but still struggles with balancing structural and logical reasoning. Covers 5,140 graphs across 11 datasets.
- **Limitations**: Evaluation limited to a snapshot of models at time of publication; rapid model evolution may change findings.
- **Notes**: Open-sourced at github.com/ZIKEYUAN/GraCoRe. Important finding: simply scaling context length is insufficient for better graph understanding.

---

### Beyond Text: A Deep Dive into LLMs' Ability on Understanding Graph Data (2023/2024)
- **Authors**: Yuntong Hu, Zheng Zhang, Liang Zhao
- **Venue/Link**: arXiv:2310.04944 (October 2023, widely cited in 2024 literature)
- **Key idea**: Evaluates LLM capabilities on graph-based prediction tasks (node, edge, and graph-level classification) by processing graph information purely through natural language prompts.
- **Relevance**: Directly investigates the core question of whether text-based graph representations are sufficient for graph understanding tasks. Provides GNN baselines for comparison.
- **Graph serialization method**: Natural language neighborhood descriptions (e.g., describing a node's neighbors and their attributes in text).
- **Results**: (1) LLMs show preliminary graph understanding but underperform specialized semi-supervised GNNs on classification tasks. (2) Models tend to treat graph prompts as unstructured context rather than relational topological data. (3) Zero-shot and few-shot settings are particularly challenging.
- **Limitations**: Focused on classification rather than algorithmic graph tasks. Limited graph sizes.
- **Notes**: Foundational reference for understanding the gap between LLM text processing and genuine graph comprehension.

---

## 2. Methods for Improving LLM Graph Understanding

### GraphWiz: An Instruction-Following Language Model for Graph Computational Problems (2024)
- **Authors**: Nuo Chen, Yuhan Li, Jianheng Tang, Jia Li
- **Venue/Link**: KDD 2024 — arXiv:2402.16029
- **Key idea**: Introduces **GraphInstruct**, a 72K-sample instruction-tuning dataset covering 9 graph problem types, and **GraphWiz**, a fine-tuned LLM that generates explicit step-by-step reasoning paths. Uses **Direct Preference Optimization (DPO)** to align reasoning quality.
- **Relevance**: Demonstrates that instruction tuning with explicit reasoning paths can substantially improve LLM graph reasoning. GraphWiz-DPO outperforms GPT-4 on these tasks.
- **Graph serialization method**: Text-based adjacency representations with explicit step-by-step solutions.
- **Results**: GraphWiz-DPO achieves 65% average accuracy across 9 tasks vs. GPT-4's 43.8%. DPO alignment significantly boosts reliability over standard SFT. Tasks range from connectivity to NP-complete problems.
- **Limitations**: Models may overfit to GraphInstruct's distribution and fail to generalize to different graph structures or larger graphs not seen in training.
- **Notes**: Code/data at github.com. GraphInstruct is widely reused by subsequent papers (e.g., GraphSolver, GraphAgent-Reasoner).

---

### G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning (2025)
- **Authors**: (Multiple authors)
- **Venue/Link**: arXiv:2505.18499 (May 2025)
- **Key idea**: Uses **reinforcement learning (GRPO — Group Relative Policy Optimization)** on synthetic graph-theoretic data to elicit graph reasoning in LLMs, without degrading general reasoning capabilities. Introduces the **Erdős dataset** — the largest graph reasoning dataset (100K train / 5K test, 50 tasks).
- **Relevance**: Represents the cutting-edge shift from supervised fine-tuning to RL-based approaches for graph reasoning. Small RL-trained models outperform much larger SFT models.
- **Graph serialization method**: Text-based graph encodings; model demonstrates generalization across encoding schemes.
- **Results**: RL-trained 3B model outperforms Qwen2.5-72B-Instruct on graph reasoning tasks. Strong zero-shot generalization to unseen tasks, domains, and graph encoding schemes. Does not degrade general reasoning (GSM8K, MATH scores maintained).
- **Limitations**: Very recent; needs independent replication. Computational cost of RL training not fully characterized.
- **Notes**: Named after Paul Erdős. Demonstrates that RL can unlock latent graph reasoning abilities more effectively than supervised approaches.

---

### Graph Chain-of-Thought: Augmenting LLMs by Reasoning on Graphs (2024)
- **Authors**: Bowen Jin, et al.
- **Venue/Link**: Findings of ACL 2024 — arXiv:2404.07103
- **Key idea**: Proposes **Graph-CoT**, a framework enabling LLMs to **iteratively traverse** graph-structured data through a cycle of: (1) LLM reasoning → (2) graph query generation → (3) graph execution → repeat. Introduces **GRBench** (1,740 questions, 10 domain graphs).
- **Relevance**: Offers a practical framework for LLMs to interact with graphs at inference time without needing the entire graph in context. Addresses the context-length bottleneck.
- **Graph serialization method**: Not full serialization — instead, the LLM queries local subgraph neighborhoods iteratively. Partial graph information provided as text at each step.
- **Results**: Significantly reduces hallucinations on knowledge-intensive tasks requiring multi-hop reasoning across graph-structured data (academic, e-commerce, healthcare, legal domains). Outperforms single-shot graph-in-context approaches.
- **Limitations**: Higher inference latency due to iterative graph traversal. Requires an external graph execution engine.
- **Notes**: GRBench provides a diverse, real-world benchmark. Framework applicable to knowledge graphs, citation networks, social networks, etc.

---

### CodeGraph / Reasoning-Then-Coding Approaches (2024)
- **Authors**: Various groups
- **Venue/Link**: Multiple papers, 2024
- **Key idea**: Instead of asking LLMs to reason about graphs in natural language, have them **generate executable code** (Python with NetworkX, etc.) to solve graph problems. The "code-first" paradigm offloads precise computation to a deterministic interpreter.
- **Relevance**: Offers a practical alternative to direct text-based graph reasoning — circumvents many of the arithmetic and structural hallucination issues.
- **Graph serialization method**: Graphs encoded as Python data structures (dictionaries, adjacency lists) within code.
- **Results**: Code generation approaches significantly outperform natural language reasoning on graph tasks requiring precise computation (shortest paths, connectivity, counting). Pseudocode injection further improves generated code quality.
- **Limitations**: Requires a code execution environment at inference time. Model must generate correct, runnable code — compilation/runtime errors can be failure modes. Less applicable when the "answer" is a qualitative description rather than a computation.
- **Notes**: Represents a paradigm shift in how to use LLMs for graph problems. Particularly relevant for engineering applications.

---

## 3. Scaling & Graph Size Effects

### Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents (GraphAgent-Reasoner) (2024)
- **Authors**: Yuwei Hu, Runlin Lei, Xinyi Huang, Zhewei Wei, Yongchao Liu
- **Venue/Link**: arXiv:2410.05130
- **Key idea**: Proposes **GraphAgent-Reasoner**, a multi-agent framework inspired by distributed graph computation. Decomposes graph problems into **node-centric tasks** — each node gets its own agent that processes local information and communicates with neighbor agents.
- **Relevance**: **Directly addresses scaling** — the central challenge for our research. Demonstrates that multi-agent decomposition can handle graphs with 1,000+ nodes, far beyond single-LLM capacity.
- **Graph serialization method**: Each agent receives only local neighborhood information as text, not the full graph.
- **Results**: Near-perfect accuracy on polynomial-time graph reasoning tasks (GraphInstruct benchmark). Outperforms both closed-source and fine-tuned open-source models. Scales to 1,000+ node graphs. Fine-tuning-free — uses existing LLMs as reasoning engines.
- **Limitations**: High inference cost (many LLM calls per problem). Communication overhead between agents. May not scale well to NP-hard problems.
- **Notes**: Key paper for scaling research. Shows that the single-context bottleneck can be overcome through distributed reasoning.

---

### LLM Graph Reasoning Scaling Behavior (Synthesized from Multiple 2024–2025 Studies)
- **Authors**: Various (findings across NLGraph, GraphQA, GraCoRe, and related work)
- **Venue/Link**: Multiple venues
- **Key idea**: Performance on graph reasoning tasks degrades systematically as graph size (nodes/edges) increases, due to: (1) context length limitations, (2) linearization information loss, (3) structural hallucination, and (4) the "curse of complexity."
- **Relevance**: **Core to our research question** — documents the scaling wall that current LLMs hit when processing larger graphs.
- **Graph serialization method**: Various text formats (all exhibit degradation).
- **Results**: (1) Performance decreases approximately linearly with graph size for most tasks. (2) Even reasoning models (o1, DeepSeek-R1) face scaling limits beyond certain complexity thresholds. (3) Increased context windows help but don't solve the fundamental problem — GraCoRe shows longer context ≠ better graph comprehension. (4) Structural hallucinations increase with graph size.
- **Limitations**: Results are aggregated across studies with different methodologies.
- **Notes**: The scaling challenge is the primary open problem in this field. Solutions being explored include multi-agent decomposition, hierarchical summarization, and hybrid GNN-LLM approaches.

---

## 4. GNN vs. LLM Comparative Studies

### Large Language Models on Graphs: A Comprehensive Survey (2024)
- **Authors**: Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, Jiawei Han
- **Venue/Link**: IEEE TKDE 2024, Vol. 36, No. 12, pp. 8622–8642 — arXiv:2312.02783
- **Key idea**: Comprehensive survey categorizing LLM-graph integration into four paradigms: (1) GNN-as-prefix (GNNs encode, LLMs predict), (2) LLM-as-prefix (LLMs encode, GNNs predict), (3) LLM-Graph Integration (joint architectures), (4) LLM-only approaches.
- **Relevance**: Provides the definitive taxonomy for understanding where text-only graph approaches fit relative to hybrid GNN-LLM methods. Essential context for positioning our work.
- **Graph serialization method**: Reviews all major serialization approaches across the literature.
- **Results**: (1) GNNs remain superior for structural/topological reasoning; LLMs excel at semantic understanding. (2) Hybrid approaches generally outperform either alone. (3) LLM-only approaches work best for text-attributed graphs where semantic content is rich. (4) The field is shifting toward integration rather than replacement.
- **Limitations**: Survey scope — cannot provide deep experimental comparison across all methods.
- **Notes**: Highly cited (IEEE TKDE). Essential reading for understanding the full landscape. Companion GitHub repository maintained with updated paper lists.

---

### A Survey of Graph Meets Large Language Model: Progress and Future Directions (2024)
- **Authors**: Various
- **Venue/Link**: IJCAI 2024
- **Key idea**: Surveys the intersection of graph learning and LLMs, proposing a framework for categorizing integration approaches and identifying future research directions.
- **Relevance**: Provides complementary perspective to the TKDE survey, with emphasis on future directions and open challenges.
- **Graph serialization method**: Reviews multiple serialization strategies.
- **Results**: Identifies key open challenges: scalability to large graphs, permutation invariance, balancing efficiency with accuracy. Highlights the growing importance of "graph foundation models."
- **Limitations**: Survey-level coverage; individual experimental results cited from other papers.
- **Notes**: IJCAI venue provides strong visibility in the AI community.

---

## 5. Knowledge Graph Integration & Hallucination Reduction

### Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey (2024)
- **Authors**: Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, Huan Liu (Arizona State University)
- **Venue/Link**: NAACL 2024 — arXiv:2311.07914
- **Key idea**: Systematically categorizes KG-based augmentation methods for reducing LLM hallucinations into: (1) Knowledge-Aware Inference, (2) Knowledge-Aware Learning, (3) Knowledge-Aware Validation.
- **Relevance**: While focused on knowledge graphs specifically, the techniques for integrating structured graph information with LLMs are directly applicable to our work.
- **Graph serialization method**: Various KG text representations (triples, subgraph descriptions, structured queries).
- **Results**: KG augmentation demonstrably reduces hallucinations across multiple domains. Knowledge-aware inference (e.g., KG-augmented retrieval) is the most practically deployed approach.
- **Limitations**: Most techniques evaluated on factoid QA rather than structural graph reasoning tasks.
- **Notes**: First comprehensive survey at the KG-hallucination intersection. Published at a top NLP venue.

---

## 6. Permutation Sensitivity & Invariance

### Large Language Models for Graph Reasoning: Capabilities, Limitations and Future Directions (2024–2026)
- **Authors**: Various
- **Venue/Link**: Authorea preprint (building on 2024 discourse)
- **Key idea**: Categorizes LLM limitations for graph reasoning into a 2×2 framework: **Permutation Sensitivity** (Global-Architectural) and **Linearization Fragility** (Local-Representational). Proposes that these are the two fundamental barriers.
- **Relevance**: Provides a theoretical framework for understanding why LLMs struggle with graphs — the mismatch between permutation-invariant graph structures and order-sensitive sequential processing.
- **Graph serialization method**: Analyzes all major serialization formats through the lens of permutation sensitivity.
- **Results**: (1) Standard LLMs suffer from "linearization fragility" — changing node/edge order in text changes outputs. (2) This is an architectural limitation, not just a data issue. (3) Current solutions include input augmentation (random permutations during training) and architectural alignment (graph-specific attention mechanisms).
- **Limitations**: More of a conceptual framework than empirical study.
- **Notes**: The permutation sensitivity issue is arguably the most fundamental challenge for text-based graph processing.

---

### PEARL: Permutation-Resilient LLMs (2024)
- **Authors**: Various
- **Venue/Link**: arXiv (2024)
- **Key idea**: Fine-tunes LLMs using **distributionally robust optimization** to ensure consistent performance across all possible permutations of input demonstrations / graph node orderings.
- **Relevance**: Directly addresses the permutation sensitivity problem that plagues text-serialized graph inputs.
- **Graph serialization method**: Standard text formats with robustness training across permutations.
- **Results**: Improved consistency of LLM performance when the same graph is presented with different node orderings.
- **Limitations**: Computational cost of training across all permutations. May not fully solve the invariance problem for very large graphs.
- **Notes**: Practical approach to mitigating a known weakness.

---

## 7. Graph Generation

### Exploring the Potential of Large Language Models in Graph Generation (2024)
- **Authors**: Yao et al.
- **Venue/Link**: arXiv (March 2024)
- **Key idea**: Proposes **LLM4GraphGen**, a framework for evaluating LLMs on graph generation tasks: rule-based generation, distribution-based generation, and property-based generation (e.g., molecular graphs).
- **Relevance**: Complements the reasoning/comprehension literature by testing whether LLMs can also *generate* valid graph structures. Tests a different facet of graph understanding.
- **Graph serialization method**: Text-based graph descriptions as both input (prompts) and output (generated graphs).
- **Results**: GPT-4 shows preliminary but notable capability in rule-based and distribution-based graph generation. Few-shot and chain-of-thought prompting provide inconsistent improvements. Property-based generation (e.g., drug discovery) shows potential.
- **Limitations**: Generated graphs often violate structural constraints. Evaluation limited to small-to-medium graphs.
- **Notes**: Important for understanding the full scope of LLM graph capabilities beyond analysis/reasoning.

---

## 8. Scalable Benchmarks (Latest)

### KG-LLM-Bench (2025)
- **Authors**: Various
- **Venue/Link**: arXiv 2025
- **Key idea**: A scalable benchmark for evaluating LLM reasoning on **text-serialized knowledge graphs**, highlighting the "reliability gap" across different models.
- **Relevance**: Directly tests our research setting — KGs serialized as text fed to LLMs. Quantifies the gap between what models claim to know and what they reliably process.
- **Graph serialization method**: Text-serialized knowledge graph triples and subgraphs.
- **Results**: Significant reliability gap across models — performance varies substantially depending on how knowledge is serialized. Models that perform well on general NLP benchmarks don't necessarily transfer to KG reasoning.
- **Limitations**: Focused on knowledge graphs rather than arbitrary graph structures.
- **Notes**: Useful for understanding the practical reliability of LLM-based graph processing in real-world KG applications.

---

### CausalGraphBench (2025)
- **Authors**: Various
- **Venue/Link**: arXiv 2025
- **Key idea**: Benchmark focused on LLM ability to **discover and construct causal graphs**, testing across varying graph sizes and complexities.
- **Relevance**: Tests a specialized but important application of graph understanding — causal reasoning. Provides data on how performance scales with graph complexity.
- **Graph serialization method**: Textual descriptions of variables and their relationships.
- **Results**: Performance degrades with increasing graph size and complexity. LLMs show some ability to identify simple causal structures but struggle with larger, more complex causal networks.
- **Limitations**: Causal discovery is inherently ambiguous; evaluation can be challenging.
- **Notes**: Bridges the graph reasoning and causal inference literatures.
