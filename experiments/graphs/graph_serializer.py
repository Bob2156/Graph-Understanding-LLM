"""
Graph Serializer Module
=======================

Converts NetworkX graphs to various text representations suitable for
feeding to LLMs.  Also provides utilities for token counting and
computing ground-truth statistics.

Supported formats:
    - adjacency_list
    - edge_list
    - natural_language
    - dot
    - adjacency_matrix
    - graphml
"""

from typing import Any, Callable, Dict, List

import networkx as nx


# ---------------------------------------------------------------------------
# Serialization functions
# ---------------------------------------------------------------------------

def serialize_adjacency_list(graph: nx.Graph) -> str:
    """Serialize a graph as an adjacency list.

    Format::

        Node 0: [1, 3, 5]
        Node 1: [0, 2]
        ...

    Args:
        graph: A NetworkX Graph.

    Returns:
        Multi-line string in adjacency-list format.
    """
    lines: List[str] = []
    for node in sorted(graph.nodes()):
        neighbors = sorted(graph.neighbors(node))
        lines.append(f"Node {node}: {neighbors}")
    return "\n".join(lines)


def serialize_edge_list(graph: nx.Graph) -> str:
    """Serialize a graph as an edge list.

    Format::

        (0, 1)
        (0, 3)
        (1, 2)
        ...

    Args:
        graph: A NetworkX Graph.

    Returns:
        Multi-line string with one edge per line.
    """
    edges = sorted(graph.edges())
    return "\n".join(f"({u}, {v})" for u, v in edges)


def serialize_natural_language(graph: nx.Graph) -> str:
    """Serialize a graph using natural-language sentences.

    Format::

        Node 0 is connected to Node 1, Node 3, and Node 5.
        Node 1 is connected to Node 0 and Node 2.
        Node 4 is connected to Node 7.
        Node 6 has no connections.

    Uses proper English list formatting with commas and "and".

    Args:
        graph: A NetworkX Graph.

    Returns:
        Multi-line natural-language description of the graph.
    """
    lines: List[str] = []
    for node in sorted(graph.nodes()):
        neighbors = sorted(graph.neighbors(node))
        if len(neighbors) == 0:
            lines.append(f"Node {node} has no connections.")
        elif len(neighbors) == 1:
            lines.append(f"Node {node} is connected to Node {neighbors[0]}.")
        elif len(neighbors) == 2:
            lines.append(
                f"Node {node} is connected to Node {neighbors[0]} "
                f"and Node {neighbors[1]}."
            )
        else:
            parts = [f"Node {n}" for n in neighbors[:-1]]
            last = f"Node {neighbors[-1]}"
            lines.append(
                f"Node {node} is connected to {', '.join(parts)}, and {last}."
            )
    return "\n".join(lines)


def serialize_dot(graph: nx.Graph) -> str:
    """Serialize a graph in DOT (Graphviz) format.

    Format::

        graph G {
          0 -- 1;
          0 -- 3;
          ...
        }

    Args:
        graph: A NetworkX Graph.

    Returns:
        String in DOT format.
    """
    lines = ["graph G {"]
    for u, v in sorted(graph.edges()):
        lines.append(f"  {u} -- {v};")
    lines.append("}")
    return "\n".join(lines)


def serialize_adjacency_matrix(graph: nx.Graph) -> str:
    """Serialize a graph as a full adjacency matrix.

    Includes row and column headers.  Entries are 0 or 1.

    Args:
        graph: A NetworkX Graph.

    Returns:
        Multi-line string representation of the adjacency matrix.
    """
    nodes = sorted(graph.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Build matrix manually (no scipy dependency)
    matrix = [[0] * n for _ in range(n)]
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        matrix[i][j] = 1
        matrix[j][i] = 1

    # Column header
    col_width = max(len(str(nd)) for nd in nodes) + 1
    header = " " * (col_width + 1) + "  ".join(str(nd).rjust(col_width) for nd in nodes)
    lines = [header]

    for i, node in enumerate(nodes):
        row_vals = "  ".join(str(int(matrix[i][j])).rjust(col_width) for j in range(n))
        lines.append(f"{str(node).rjust(col_width)}  {row_vals}")

    return "\n".join(lines)


def serialize_graphml(graph: nx.Graph) -> str:
    """Serialize a graph in GraphML-style XML format.

    Args:
        graph: A NetworkX Graph.

    Returns:
        String in GraphML XML format.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml>',
        '  <graph id="G" edgedefault="undirected">',
    ]
    for node in sorted(graph.nodes()):
        lines.append(f'    <node id="n{node}"/>')
    for u, v in sorted(graph.edges()):
        lines.append(f'    <edge source="n{u}" target="n{v}"/>')
    lines.append("  </graph>")
    lines.append("</graphml>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SERIALIZATION_FORMATS: Dict[str, Callable[[nx.Graph], str]] = {
    "adjacency_list": serialize_adjacency_list,
    "edge_list": serialize_edge_list,
    "natural_language": serialize_natural_language,
    "dot": serialize_dot,
    "adjacency_matrix": serialize_adjacency_matrix,
    "graphml": serialize_graphml,
}
"""Mapping from format name to serialization function."""


def serialize_graph(graph: nx.Graph, format_name: str) -> str:
    """Serialize a graph using the named format.

    Args:
        graph: A NetworkX Graph.
        format_name: One of the keys in :data:`SERIALIZATION_FORMATS`.

    Returns:
        Serialized text representation.

    Raises:
        ValueError: If *format_name* is unknown.
    """
    if format_name not in SERIALIZATION_FORMATS:
        raise ValueError(
            f"Unknown format '{format_name}'. "
            f"Choose from: {list(SERIALIZATION_FORMATS.keys())}"
        )
    return SERIALIZATION_FORMATS[format_name](graph)


# ---------------------------------------------------------------------------
# Token / size utilities
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Approximate token count by splitting on whitespace.

    This is a rough heuristic—real tokenisers will differ—but it
    provides a fast, dependency-free estimate.

    Args:
        text: Input text.

    Returns:
        Number of whitespace-separated tokens.
    """
    return len(text.split())


def get_text_stats(graph: nx.Graph) -> Dict[str, Dict[str, Any]]:
    """Compute text-size statistics for every serialization format.

    Args:
        graph: A NetworkX Graph.

    Returns:
        Dictionary keyed by format name, each mapping to::

            {
                "text": <serialized string>,
                "token_count": <int>,
                "char_count": <int>,
            }
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for fmt_name, fmt_fn in SERIALIZATION_FORMATS.items():
        text = fmt_fn(graph)
        stats[fmt_name] = {
            "text": text,
            "token_count": count_tokens(text),
            "char_count": len(text),
        }
    return stats


# ---------------------------------------------------------------------------
# Ground-truth computation
# ---------------------------------------------------------------------------

def compute_ground_truth(graph: nx.Graph) -> Dict[str, Any]:
    """Compute ground-truth answers for common graph properties.

    Args:
        graph: A NetworkX Graph.

    Returns:
        Dictionary with keys:

        - ``node_count``
        - ``edge_count``
        - ``is_connected``
        - ``has_cycle``
        - ``diameter`` (``None`` if disconnected)
        - ``average_degree``
        - ``max_degree``
        - ``min_degree``
        - ``triangle_count``
    """
    degrees = [d for _, d in graph.degree()]

    is_connected = nx.is_connected(graph)

    # Diameter only defined for connected graphs
    diameter = nx.diameter(graph) if is_connected else None

    # Cycle detection
    has_cycle = len(nx.cycle_basis(graph)) > 0

    # Triangles
    triangle_count = sum(nx.triangles(graph).values()) // 3

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "is_connected": is_connected,
        "has_cycle": has_cycle,
        "diameter": diameter,
        "average_degree": sum(degrees) / len(degrees) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "min_degree": min(degrees) if degrees else 0,
        "triangle_count": triangle_count,
    }


if __name__ == "__main__":
    # Quick demo
    G = nx.erdos_renyi_graph(8, 0.4, seed=42)
    for fmt in SERIALIZATION_FORMATS:
        text = serialize_graph(G, fmt)
        tokens = count_tokens(text)
        print(f"--- {fmt} ({tokens} tokens, {len(text)} chars) ---")
        print(text[:300])
        print()
