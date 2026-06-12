"""
Graph Reasoning Tasks Module
=============================

Defines graph reasoning tasks with ground-truth solvers for benchmarking
LLM performance on graph understanding.  Each task provides:

    - A prompt generator that combines graph text with a question
    - A ground-truth solver using NetworkX
    - An answer parser / evaluator that checks LLM output against truth
"""

import random as _random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class GraphTask:
    """Descriptor for a single graph-reasoning task.

    Attributes:
        name: Machine-readable task identifier.
        description: Human-readable description.
        needs_node: Whether the task requires selecting a random node.
        needs_node_pair: Whether the task requires selecting a random pair.
        generate_question: Callable that produces the question string.
        solve: Callable that computes the ground-truth answer.
        evaluate: Callable that checks an LLM response against truth.
    """
    name: str
    description: str
    needs_node: bool = False
    needs_node_pair: bool = False


# ---------------------------------------------------------------------------
# Answer parsing helpers
# ---------------------------------------------------------------------------

def _parse_integer(text: str) -> Optional[int]:
    """Extract the answer integer from a text response.

    Strategy:
    1. Look for explicit 'final answer' patterns
    2. Look for bold/formatted answers at the end
    3. Fall back to the last integer in the response
    """
    # 1. Look for "Final Answer: X" or "Answer: X" patterns
    final_match = re.search(
        r"(?:final\s+answer|answer)\s*[:=]\s*\**\s*(-?\d+)",
        text, re.IGNORECASE,
    )
    if final_match:
        return int(final_match.group(1))

    # 2. Look for bold number at the end (e.g., "**5**" or "**2**")
    bold_match = re.search(r"\*\*(-?\d+)\*\*\s*$", text.strip())
    if bold_match:
        return int(bold_match.group(1))

    # 3. Fall back to last standalone integer in the response
    matches = re.findall(r"-?\d+", text)
    if matches:
        return int(matches[-1])
    return None


def _parse_yes_no(text: str) -> Optional[bool]:
    """Determine yes/no from a text response.

    Looks for common affirmative / negative keywords.
    """
    text_lower = text.lower().strip()
    # Check first line / first few words for the answer
    if re.search(r"\byes\b", text_lower):
        return True
    if re.search(r"\bno\b", text_lower):
        return False
    if re.search(r"\btrue\b", text_lower):
        return True
    if re.search(r"\bfalse\b", text_lower):
        return False
    return None


def _parse_integer_list(text: str) -> Optional[List[int]]:
    """Extract a list of integers from a text response."""
    # Look for numbers separated by commas, spaces, or listed
    numbers = re.findall(r"\d+", text)
    if numbers:
        return [int(n) for n in numbers]
    return None


# ---------------------------------------------------------------------------
# Task solvers
# ---------------------------------------------------------------------------

def _solve_node_degree(graph: nx.Graph, node: int, **kwargs: Any) -> int:
    """Return the degree of the given node."""
    return graph.degree(node)


def _solve_edge_existence(
    graph: nx.Graph, node_a: int, node_b: int, **kwargs: Any
) -> bool:
    """Return whether an edge exists between two nodes."""
    return graph.has_edge(node_a, node_b)


def _solve_neighbor_listing(
    graph: nx.Graph, node: int, **kwargs: Any
) -> List[int]:
    """Return sorted list of neighbours of the given node."""
    return sorted(graph.neighbors(node))


def _solve_shortest_path(
    graph: nx.Graph, node_a: int, node_b: int, **kwargs: Any
) -> int:
    """Return shortest path length, or -1 if no path exists."""
    try:
        return nx.shortest_path_length(graph, node_a, node_b)
    except nx.NetworkXNoPath:
        return -1


def _solve_connectivity(
    graph: nx.Graph, node_a: int, node_b: int, **kwargs: Any
) -> bool:
    """Return whether two nodes are connected (path exists)."""
    return nx.has_path(graph, node_a, node_b)


def _solve_cycle_detection(graph: nx.Graph, **kwargs: Any) -> bool:
    """Return whether the graph contains at least one cycle."""
    return len(nx.cycle_basis(graph)) > 0


def _solve_triangle_counting(graph: nx.Graph, **kwargs: Any) -> int:
    """Return the number of triangles in the graph."""
    return sum(nx.triangles(graph).values()) // 3


def _solve_graph_diameter(graph: nx.Graph, **kwargs: Any) -> int:
    """Return graph diameter, or -1 if the graph is disconnected."""
    if nx.is_connected(graph):
        return nx.diameter(graph)
    return -1


def _solve_node_count(graph: nx.Graph, **kwargs: Any) -> int:
    """Return the number of nodes."""
    return graph.number_of_nodes()


def _solve_edge_count(graph: nx.Graph, **kwargs: Any) -> int:
    """Return the number of edges."""
    return graph.number_of_edges()


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------

def _question_node_degree(node: int, **kwargs: Any) -> str:
    return f"What is the degree of node {node}?"


def _question_edge_existence(node_a: int, node_b: int, **kwargs: Any) -> str:
    return f"Is there an edge between node {node_a} and node {node_b}? Answer yes or no."


def _question_neighbor_listing(node: int, **kwargs: Any) -> str:
    return f"List all neighbors of node {node}."


def _question_shortest_path(node_a: int, node_b: int, **kwargs: Any) -> str:
    return (
        f"What is the shortest path length from node {node_a} to node {node_b}? "
        f"If there is no path, answer -1."
    )


def _question_connectivity(node_a: int, node_b: int, **kwargs: Any) -> str:
    return (
        f"Are node {node_a} and node {node_b} connected "
        f"(i.e., is there a path between them)? Answer yes or no."
    )


def _question_cycle_detection(**kwargs: Any) -> str:
    return "Does this graph contain a cycle? Answer yes or no."


def _question_triangle_counting(**kwargs: Any) -> str:
    return "How many triangles are in this graph?"


def _question_graph_diameter(**kwargs: Any) -> str:
    return (
        "What is the diameter of this graph? "
        "If the graph is disconnected, answer -1."
    )


def _question_node_count(**kwargs: Any) -> str:
    return "How many nodes are in this graph?"


def _question_edge_count(**kwargs: Any) -> str:
    return "How many edges are in this graph?"


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _evaluate_integer(response: str, ground_truth: int) -> Dict[str, Any]:
    """Evaluate a response that should contain an integer."""
    parsed = _parse_integer(response)
    return {
        "correct": parsed == ground_truth,
        "expected": ground_truth,
        "parsed": parsed,
        "raw_response": response,
    }


def _evaluate_yes_no(response: str, ground_truth: bool) -> Dict[str, Any]:
    """Evaluate a response that should be yes/no."""
    parsed = _parse_yes_no(response)
    return {
        "correct": parsed == ground_truth,
        "expected": ground_truth,
        "parsed": parsed,
        "raw_response": response,
    }


def _evaluate_neighbor_list(
    response: str, ground_truth: List[int]
) -> Dict[str, Any]:
    """Evaluate a response that should list node neighbours."""
    parsed = _parse_integer_list(response)
    if parsed is not None:
        correct = set(parsed) == set(ground_truth)
    else:
        correct = False
    return {
        "correct": correct,
        "expected": ground_truth,
        "parsed": parsed,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

# Each entry: (question_fn, solve_fn, evaluate_fn, needs_node, needs_node_pair)
_TASK_DEFS: Dict[str, Dict[str, Any]] = {
    "node_degree": {
        "description": "What is the degree of a given node?",
        "question_fn": _question_node_degree,
        "solve_fn": _solve_node_degree,
        "evaluate_fn": _evaluate_integer,
        "needs_node": True,
        "needs_node_pair": False,
    },
    "edge_existence": {
        "description": "Is there an edge between two given nodes?",
        "question_fn": _question_edge_existence,
        "solve_fn": _solve_edge_existence,
        "evaluate_fn": _evaluate_yes_no,
        "needs_node": False,
        "needs_node_pair": True,
    },
    "neighbor_listing": {
        "description": "List all neighbors of a given node.",
        "question_fn": _question_neighbor_listing,
        "solve_fn": _solve_neighbor_listing,
        "evaluate_fn": _evaluate_neighbor_list,
        "needs_node": True,
        "needs_node_pair": False,
    },
    "shortest_path": {
        "description": "What is the shortest path length between two nodes?",
        "question_fn": _question_shortest_path,
        "solve_fn": _solve_shortest_path,
        "evaluate_fn": _evaluate_integer,
        "needs_node": False,
        "needs_node_pair": True,
    },
    "connectivity": {
        "description": "Are two given nodes connected (is there a path)?",
        "question_fn": _question_connectivity,
        "solve_fn": _solve_connectivity,
        "evaluate_fn": _evaluate_yes_no,
        "needs_node": False,
        "needs_node_pair": True,
    },
    "cycle_detection": {
        "description": "Does the graph contain a cycle?",
        "question_fn": _question_cycle_detection,
        "solve_fn": _solve_cycle_detection,
        "evaluate_fn": _evaluate_yes_no,
        "needs_node": False,
        "needs_node_pair": False,
    },
    "triangle_counting": {
        "description": "How many triangles are in the graph?",
        "question_fn": _question_triangle_counting,
        "solve_fn": _solve_triangle_counting,
        "evaluate_fn": _evaluate_integer,
        "needs_node": False,
        "needs_node_pair": False,
    },
    "graph_diameter": {
        "description": "What is the diameter of the graph?",
        "question_fn": _question_graph_diameter,
        "solve_fn": _solve_graph_diameter,
        "evaluate_fn": _evaluate_integer,
        "needs_node": False,
        "needs_node_pair": False,
    },
    "node_count": {
        "description": "How many nodes are in the graph?",
        "question_fn": _question_node_count,
        "solve_fn": _solve_node_count,
        "evaluate_fn": _evaluate_integer,
        "needs_node": False,
        "needs_node_pair": False,
    },
    "edge_count": {
        "description": "How many edges are in the graph?",
        "question_fn": _question_edge_count,
        "solve_fn": _solve_edge_count,
        "evaluate_fn": _evaluate_integer,
        "needs_node": False,
        "needs_node_pair": False,
    },
}


TASKS: Dict[str, Dict[str, Any]] = _TASK_DEFS
"""Registry of all available graph-reasoning tasks."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_task_prompt(
    graph: nx.Graph,
    graph_text: str,
    task_name: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a complete prompt for a graph-reasoning task.

    Combines the serialized graph text with a task-specific question.
    For tasks that require selecting specific nodes, nodes are chosen
    pseudo-randomly using the provided seed.

    Args:
        graph: The NetworkX graph (used for solver and node selection).
        graph_text: Pre-serialized text representation of the graph.
        task_name: Name of the task (key in :data:`TASKS`).
        seed: Random seed for reproducible node selection.

    Returns:
        Dictionary with keys:

        - ``prompt``: The full prompt string.
        - ``ground_truth``: The correct answer.
        - ``task_name``: Echo of the task name.
        - ``params``: Parameters used (e.g., selected nodes).

    Raises:
        ValueError: If *task_name* is not in :data:`TASKS`.
    """
    if task_name not in TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"
        )

    task_def = TASKS[task_name]
    rng = _random.Random(seed)
    nodes = sorted(graph.nodes())
    params: Dict[str, Any] = {}

    # Select nodes if needed
    if task_def["needs_node"]:
        node = rng.choice(nodes)
        params["node"] = node
    elif task_def["needs_node_pair"]:
        node_a, node_b = rng.sample(nodes, 2)
        params["node_a"] = node_a
        params["node_b"] = node_b

    # Generate question
    question = task_def["question_fn"](**params)

    # Solve
    solve_kwargs = {**params, "graph": graph}  # type: ignore[arg-type]
    ground_truth = task_def["solve_fn"](**solve_kwargs)

    # Build full prompt
    prompt = (
        "Consider the following graph:\n\n"
        f"{graph_text}\n\n"
        f"Question: {question}\n\n"
        "Please provide your answer concisely."
    )

    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "task_name": task_name,
        "params": params,
    }


def evaluate_response(
    task_name: str,
    response: str,
    ground_truth: Any,
) -> Dict[str, Any]:
    """Evaluate an LLM response against the ground-truth answer.

    Args:
        task_name: Name of the task.
        response: Raw text response from the LLM.
        ground_truth: The correct answer from the solver.

    Returns:
        Dictionary with keys:

        - ``correct``: Whether the answer is correct.
        - ``expected``: The ground-truth value.
        - ``parsed``: The parsed answer from the response.
        - ``raw_response``: Echo of the raw response.
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'.")

    eval_fn = TASKS[task_name]["evaluate_fn"]
    return eval_fn(response, ground_truth)


if __name__ == "__main__":
    # Quick self-test
    G = nx.erdos_renyi_graph(10, 0.3, seed=42)
    from experiments.graphs.graph_serializer import serialize_adjacency_list

    graph_text = serialize_adjacency_list(G)
    for task_name in TASKS:
        result = generate_task_prompt(G, graph_text, task_name, seed=42)
        print(f"=== {task_name} ===")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Params: {result['params']}")
        print()
