"""
Graph Generator Module
======================

Generates test graphs of various types and sizes using NetworkX.
Supports Erdős-Rényi, Barabási-Albert, Watts-Strogatz, complete,
tree, grid/lattice, bipartite, path, and cycle graphs.

Each generator takes `n_nodes` as a primary parameter and an optional
`seed` for reproducibility.
"""

import math
import random as _random
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Individual graph generators
# ---------------------------------------------------------------------------

def generate_erdos_renyi(
    n_nodes: int,
    p: float = 0.3,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a random Erdős-Rényi graph G(n, p).

    Each possible edge is included independently with probability *p*.

    Args:
        n_nodes: Number of nodes.
        p: Probability of edge creation (0 ≤ p ≤ 1).
        seed: Random seed for reproducibility.

    Returns:
        A NetworkX Graph instance.
    """
    return nx.erdos_renyi_graph(n_nodes, p, seed=seed)


def generate_barabasi_albert(
    n_nodes: int,
    m: int = 3,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a scale-free Barabási-Albert preferential-attachment graph.

    New nodes attach to *m* existing nodes with probability proportional
    to their current degree.

    Args:
        n_nodes: Number of nodes.
        m: Number of edges to attach from a new node to existing nodes.
        seed: Random seed for reproducibility.

    Returns:
        A NetworkX Graph instance.
    """
    # m must be < n_nodes
    m = min(m, n_nodes - 1) if n_nodes > 1 else 1
    return nx.barabasi_albert_graph(n_nodes, m, seed=seed)


def generate_watts_strogatz(
    n_nodes: int,
    k: int = 4,
    p: float = 0.3,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a small-world Watts-Strogatz graph.

    Starts with a ring lattice where each node is connected to its *k*
    nearest neighbours, then rewires each edge with probability *p*.

    Args:
        n_nodes: Number of nodes.
        k: Each node is joined with its k nearest neighbours in the ring.
        p: Probability of rewiring each edge.
        seed: Random seed for reproducibility.

    Returns:
        A NetworkX Graph instance.
    """
    # k must be even and < n_nodes
    k = min(k, n_nodes - 1)
    if k % 2 != 0:
        k = max(k - 1, 2)
    return nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)


def generate_complete(n_nodes: int) -> nx.Graph:
    """Generate a complete graph K_n.

    Every pair of distinct nodes is connected by an edge.

    Args:
        n_nodes: Number of nodes.

    Returns:
        A NetworkX Graph instance.
    """
    return nx.complete_graph(n_nodes)


def generate_tree(
    n_nodes: int,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a random tree (random Prüfer-sequence tree).

    Args:
        n_nodes: Number of nodes.
        seed: Random seed for reproducibility.

    Returns:
        A NetworkX Graph (tree) instance.
    """
    if n_nodes <= 2:
        return nx.path_graph(n_nodes)
    return nx.random_labeled_tree(n_nodes, seed=seed)


def generate_grid(n_nodes: int) -> nx.Graph:
    """Generate a 2-D grid / lattice graph.

    The grid dimensions are chosen to be as close to square as possible
    while having at least *n_nodes* nodes.  Nodes are relabelled to
    integers ``0 … n-1``.

    Args:
        n_nodes: Approximate number of nodes (actual may be ≥ n_nodes).

    Returns:
        A NetworkX Graph instance.
    """
    rows = int(math.isqrt(n_nodes))
    cols = math.ceil(n_nodes / rows) if rows > 0 else n_nodes
    G = nx.grid_2d_graph(rows, cols)
    # Relabel (row, col) tuples to plain integers
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    return G


def generate_bipartite(
    n_nodes: int,
    m_ratio: float = 0.5,
    p: float = 0.4,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a random bipartite graph.

    Nodes are split into two sets; edges are created between sets with
    probability *p*.

    Args:
        n_nodes: Total number of nodes.
        m_ratio: Fraction of nodes in the first partition (0 < m_ratio < 1).
        p: Probability of edge between nodes in different partitions.
        seed: Random seed for reproducibility.

    Returns:
        A NetworkX Graph instance with bipartite node attribute.
    """
    n1 = max(1, int(n_nodes * m_ratio))
    n2 = max(1, n_nodes - n1)
    return nx.bipartite.random_graph(n1, n2, p, seed=seed)


def generate_path(n_nodes: int) -> nx.Graph:
    """Generate a path graph P_n.

    A linear chain of *n_nodes* nodes connected end-to-end.

    Args:
        n_nodes: Number of nodes.

    Returns:
        A NetworkX Graph instance.
    """
    return nx.path_graph(n_nodes)


def generate_cycle(n_nodes: int) -> nx.Graph:
    """Generate a cycle graph C_n.

    A single cycle through all *n_nodes* nodes.

    Args:
        n_nodes: Number of nodes.

    Returns:
        A NetworkX Graph instance.
    """
    return nx.cycle_graph(n_nodes)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRAPH_TYPES: Dict[str, Callable[..., nx.Graph]] = {
    "erdos_renyi": generate_erdos_renyi,
    "barabasi_albert": generate_barabasi_albert,
    "watts_strogatz": generate_watts_strogatz,
    "complete": generate_complete,
    "tree": generate_tree,
    "grid": generate_grid,
    "bipartite": generate_bipartite,
    "path": generate_path,
    "cycle": generate_cycle,
}
"""Mapping from graph-type name to its generator function."""


# ---------------------------------------------------------------------------
# Test-suite generator
# ---------------------------------------------------------------------------

def generate_test_suite(
    sizes: Optional[List[int]] = None,
    graph_types: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[Tuple[str, int], nx.Graph]:
    """Generate a comprehensive suite of test graphs.

    Produces one graph for every ``(graph_type, size)`` combination.

    Args:
        sizes: List of node counts to generate.  Defaults to
               ``[10, 20, 50, 100, 200, 500]``.
        graph_types: List of graph-type names (keys of :data:`GRAPH_TYPES`).
                     Defaults to all available types.
        seed: Base random seed.  Each graph gets ``seed + i`` for
              reproducibility.

    Returns:
        Dictionary keyed by ``(graph_type_name, n_nodes)`` tuples, with
        :class:`nx.Graph` values.
    """
    if sizes is None:
        sizes = [10, 20, 50, 100, 200, 500]
    if graph_types is None:
        graph_types = list(GRAPH_TYPES.keys())

    suite: Dict[Tuple[str, int], nx.Graph] = {}
    counter = 0

    for gtype in graph_types:
        gen_fn = GRAPH_TYPES[gtype]
        for n in sizes:
            current_seed = seed + counter
            counter += 1

            # Determine which kwargs the generator accepts
            import inspect
            sig = inspect.signature(gen_fn)
            kwargs: dict = {}
            if "seed" in sig.parameters:
                kwargs["seed"] = current_seed

            G = gen_fn(n_nodes=n, **kwargs)
            suite[(gtype, n)] = G

    return suite


if __name__ == "__main__":
    # Quick demo
    suite = generate_test_suite(sizes=[10, 20])
    for (gtype, size), G in sorted(suite.items()):
        print(f"{gtype:20s}  n={size:4d}  nodes={G.number_of_nodes():4d}  edges={G.number_of_edges():4d}")
