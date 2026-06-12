"""
Experiment Configuration Module
===============================

Defines the :class:`ExperimentConfig` dataclass that holds all
experiment parameters (graph sizes, types, serialization formats,
tasks, models, etc.).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for a graph-understanding benchmark experiment.

    Attributes:
        graph_sizes: List of node counts to test.
        graph_types: List of graph-type names (keys of ``GRAPH_TYPES``).
        serialization_formats: List of serialization format names.
        tasks: List of task names (keys of ``TASKS``).
        models: List of LLM model identifiers.
        num_trials: Number of independent trials per configuration.
        seed: Base random seed for reproducibility.
        results_dir: Directory to save results.
        dry_run: If ``True``, generate prompts without calling LLM APIs.
        rate_limit_seconds: Seconds to sleep between API calls.
        max_retries: Maximum number of retries on API failure.
    """

    graph_sizes: List[int] = field(
        default_factory=lambda: [10, 20, 50, 100, 200, 500]
    )
    graph_types: List[str] = field(
        default_factory=lambda: [
            "erdos_renyi",
            "barabasi_albert",
            "watts_strogatz",
            "complete",
            "tree",
            "grid",
        ]
    )
    serialization_formats: List[str] = field(
        default_factory=lambda: ["adjacency_list", "edge_list", "natural_language"]
    )
    tasks: List[str] = field(
        default_factory=lambda: [
            "node_degree",
            "edge_existence",
            "neighbor_listing",
            "shortest_path",
            "connectivity",
            "cycle_detection",
            "triangle_counting",
            "graph_diameter",
            "node_count",
            "edge_count",
        ]
    )
    models: List[str] = field(
        default_factory=lambda: ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
    )
    num_trials: int = 3
    seed: int = 42
    results_dir: str = "results"
    dry_run: bool = False
    rate_limit_seconds: float = 1.0
    max_retries: int = 3


DEFAULT_CONFIG = ExperimentConfig()
"""Default experiment configuration with full parameter grid."""


def get_small_config() -> ExperimentConfig:
    """Return a small configuration for quick testing.

    Uses only 10- and 20-node graphs, adjacency-list format,
    Erdős-Rényi type, all tasks, and a single trial.

    Returns:
        An :class:`ExperimentConfig` instance.
    """
    return ExperimentConfig(
        graph_sizes=[10, 20],
        graph_types=["erdos_renyi"],
        serialization_formats=["adjacency_list"],
        num_trials=1,
        dry_run=True,
    )
