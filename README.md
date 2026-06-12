# Graph-Understanding-LLM

**Literature Study on Graph Theory with LLMs**

Advisor: Dr. Feng — Stevens Institute of Technology

---

## Research Question

> How well can current LLMs process large graphs serialized directly as text within their context windows, and how can this be improved?

**Key constraint**: Large graphs fed into LLMs directly as text/context — no external tools or auxiliary methods.

## Plan

1. **Foundations** — Learn about graph theory, refresh on LLM tokenization and reasoning
2. **Literature Review** — Find current papers on graph theory in ML, specifically LLMs
3. **Benchmarking** — Test how well current LLMs work with large graphs with many distinct nodes, list results
4. **Improvement Ideas** — Find papers/ideas on improving LLM graph understanding, create new ideas if necessary
5. **Experimentation** — Test ideas if compute is small, or implement scaled-down simulations

## Repository Structure

```
├── Agents.md              # AI agent guidelines
├── README.md              # This file
├── literature/
│   ├── papers/            # Paper summaries and notes
│   └── foundations/       # Graph theory & LLM fundamentals
├── experiments/
│   ├── benchmarks/        # Scripts for testing LLMs on graph tasks
│   ├── improvements/      # Scripts for testing improvement ideas
│   └── graphs/            # Graph generation and serialization utilities
├── results/               # Experiment outputs, tables, analysis
├── data/                  # Graph datasets and test cases
└── notes/                 # Meeting notes, brainstorms, misc
```
