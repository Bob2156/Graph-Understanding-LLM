"""
Benchmark Runner
================

Main entry point for running graph-understanding benchmarks against LLMs.
Supports OpenAI, Anthropic, and Google Gemini APIs with rate limiting,
retries, and dry-run mode.

Usage::

    python -m experiments.benchmarks.run_benchmark --dry-run
    python -m experiments.benchmarks.run_benchmark --models gpt-4o --sizes 10 20
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.benchmarks.config import ExperimentConfig, DEFAULT_CONFIG
from experiments.benchmarks.tasks import TASKS, generate_task_prompt, evaluate_response
from experiments.graphs.graph_generator import GRAPH_TYPES
from experiments.graphs.graph_serializer import serialize_graph, count_tokens

# ---------------------------------------------------------------------------
# Optional API client imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional


class BenchmarkRunner:
    """Orchestrates graph-understanding benchmark experiments.

    Args:
        config: An :class:`ExperimentConfig` instance controlling the
                experimental grid.
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        local_url: Optional[str] = None,
        local_model_name: Optional[str] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        if local_url:
            self.config.local_url = local_url  # type: ignore[attr-defined]
        if local_model_name:
            self.config.local_model_name = local_model_name  # type: ignore[attr-defined]
        self._setup_results_dir()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_results_dir(self) -> None:
        """Create the results directory tree."""
        base = Path(self.config.results_dir)
        base.mkdir(parents=True, exist_ok=True)
        if self.config.dry_run:
            (base / "dry_run").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # LLM API callers
    # ------------------------------------------------------------------

    def _call_llm(self, model: str, prompt: str) -> str:
        """Route an LLM call to the appropriate provider.

        Args:
            model: Model identifier (prefix determines provider).
            prompt: The prompt string.

        Returns:
            The model's text response.
        """
        if model.startswith("gpt"):
            return self._call_openai(model, prompt)
        elif model.startswith("claude"):
            return self._call_anthropic(model, prompt)
        elif model.startswith("gemini"):
            return self._call_gemini(model, prompt)
        elif model == "local" or model.startswith("local:"):
            return self._call_local(model, prompt)
        else:
            raise ValueError(f"Unknown model provider for '{model}'")

    def _call_openai(self, model: str, prompt: str) -> str:
        """Call the OpenAI Chat Completions API.

        Args:
            model: OpenAI model name (e.g., ``gpt-4o``).
            prompt: The user prompt.

        Returns:
            The assistant's response text.
        """
        if not _HAS_OPENAI:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a graph theory expert. Answer the question "
                        "about the given graph concisely and precisely."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, model: str, prompt: str) -> str:
        """Call the Anthropic Messages API.

        Args:
            model: Anthropic model name (e.g., ``claude-sonnet-4-20250514``).
            prompt: The user prompt.

        Returns:
            The assistant's response text.
        """
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=(
                "You are a graph theory expert. Answer the question "
                "about the given graph concisely and precisely."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_gemini(self, model: str, prompt: str) -> str:
        """Call the Google Gemini API.

        Args:
            model: Gemini model name (e.g., ``gemini-2.5-pro``).
            prompt: The user prompt.

        Returns:
            The model's response text.
        """
        if not _HAS_GEMINI:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_obj = genai.GenerativeModel(
            model,
            system_instruction=(
                "You are a graph theory expert. Answer the question "
                "about the given graph concisely and precisely."
            ),
        )
        response = model_obj.generate_content(prompt)
        return response.text

    def _call_local(
        self, model: str, prompt: str,
    ) -> str:
        """Call a local model via OpenAI-compatible API (e.g., LM Studio).

        Args:
            model: Either ``"local"`` or ``"local:<model-name>"``.
            prompt: The user prompt.

        Returns:
            The model's response text.
        """
        if not _HAS_OPENAI:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        base_url = getattr(self.config, "local_url", "http://localhost:1234/v1")
        model_name = getattr(self.config, "local_model_name", "")
        if model.startswith("local:"):
            model_name = model.split(":", 1)[1]
        if not model_name:
            # LM Studio uses whatever model is loaded
            model_name = "qwen3.5-9b"

        client = openai.OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LM Studio ignores the key
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a graph theory expert. Answer the question "
                        "about the given graph concisely and precisely. "
                        "Give your final answer clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    def _call_with_retry(self, model: str, prompt: str) -> str:
        """Call LLM with exponential-backoff retry logic.

        Args:
            model: Model identifier.
            prompt: The prompt.

        Returns:
            The model's response text.

        Raises:
            Exception: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                return self._call_llm(model, prompt)
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(
                    f"  [retry {attempt + 1}/{self.config.max_retries}] "
                    f"Error: {e}. Waiting {wait}s..."
                )
                time.sleep(wait)
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full benchmark experiment.

        Iterates over the Cartesian product of graph types, sizes,
        serialization formats, tasks, models, and trials.

        Returns:
            A dictionary containing all results and metadata.
        """
        results: List[Dict[str, Any]] = []
        total_configs = (
            len(self.config.graph_types)
            * len(self.config.graph_sizes)
            * len(self.config.serialization_formats)
            * len(self.config.tasks)
            * len(self.config.models)
            * self.config.num_trials
        )

        print(f"Benchmark configuration:")
        print(f"  Graph types:  {self.config.graph_types}")
        print(f"  Sizes:        {self.config.graph_sizes}")
        print(f"  Formats:      {self.config.serialization_formats}")
        print(f"  Tasks:        {self.config.tasks}")
        print(f"  Models:       {self.config.models}")
        print(f"  Trials:       {self.config.num_trials}")
        print(f"  Dry run:      {self.config.dry_run}")
        print(f"  Total configs: {total_configs}")
        print()

        counter = 0
        for graph_type in self.config.graph_types:
            gen_fn = GRAPH_TYPES[graph_type]
            for size in self.config.graph_sizes:
                for trial in range(self.config.num_trials):
                    # Generate graph with unique seed per trial
                    trial_seed = self.config.seed + counter
                    import inspect
                    sig = inspect.signature(gen_fn)
                    gen_kwargs: dict = {}
                    if "seed" in sig.parameters:
                        gen_kwargs["seed"] = trial_seed

                    graph = gen_fn(n_nodes=size, **gen_kwargs)

                    for fmt in self.config.serialization_formats:
                        graph_text = serialize_graph(graph, fmt)
                        token_count = count_tokens(graph_text)

                        for task_name in self.config.tasks:
                            task_prompt_data = generate_task_prompt(
                                graph, graph_text, task_name,
                                seed=trial_seed,
                            )

                            for model in self.config.models:
                                counter += 1
                                result_entry: Dict[str, Any] = {
                                    "id": counter,
                                    "graph_type": graph_type,
                                    "n_nodes": size,
                                    "actual_nodes": graph.number_of_nodes(),
                                    "actual_edges": graph.number_of_edges(),
                                    "format": fmt,
                                    "token_count": token_count,
                                    "task": task_name,
                                    "task_params": task_prompt_data["params"],
                                    "ground_truth": _serialize_value(
                                        task_prompt_data["ground_truth"]
                                    ),
                                    "model": model,
                                    "trial": trial,
                                    "seed": trial_seed,
                                    "prompt": task_prompt_data["prompt"],
                                }

                                if self.config.dry_run:
                                    result_entry["response"] = None
                                    result_entry["evaluation"] = None
                                    result_entry["status"] = "dry_run"
                                    _progress = f"[{counter}] "
                                    print(
                                        f"{_progress}{graph_type} n={size} "
                                        f"{fmt} {task_name} {model} "
                                        f"trial={trial} [DRY RUN]"
                                    )
                                else:
                                    try:
                                        print(
                                            f"[{counter}/{total_configs}] "
                                            f"{graph_type} n={size} {fmt} "
                                            f"{task_name} {model} trial={trial}",
                                            end=" ... ",
                                        )
                                        response = self._call_with_retry(
                                            model,
                                            task_prompt_data["prompt"],
                                        )
                                        evaluation = evaluate_response(
                                            task_name,
                                            response,
                                            task_prompt_data["ground_truth"],
                                        )
                                        result_entry["response"] = response
                                        result_entry["evaluation"] = evaluation
                                        result_entry["status"] = "success"
                                        print(
                                            f"{'✓' if evaluation['correct'] else '✗'}"
                                        )

                                        # Rate limiting
                                        time.sleep(
                                            self.config.rate_limit_seconds
                                        )
                                    except Exception as e:
                                        result_entry["response"] = None
                                        result_entry["evaluation"] = None
                                        result_entry["status"] = f"error: {e}"
                                        print(f"ERROR: {e}")

                                results.append(result_entry)

        output = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "graph_sizes": self.config.graph_sizes,
                    "graph_types": self.config.graph_types,
                    "serialization_formats": self.config.serialization_formats,
                    "tasks": self.config.tasks,
                    "models": self.config.models,
                    "num_trials": self.config.num_trials,
                    "seed": self.config.seed,
                    "dry_run": self.config.dry_run,
                },
                "total_results": len(results),
            },
            "results": results,
        }

        return output

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """Save results to a JSON file.

        Args:
            results: The results dictionary from :meth:`run`.
            filename: Optional filename override.  Defaults to a
                      timestamped name.

        Returns:
            Absolute path to the saved file.
        """
        base = Path(self.config.results_dir)
        if self.config.dry_run:
            base = base / "dry_run"

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "dry_run" if self.config.dry_run else "benchmark"
            filename = f"{mode}_{ts}.json"

        filepath = base / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")
        return str(filepath.resolve())


def _serialize_value(value: Any) -> Any:
    """Convert non-JSON-serializable values for storage."""
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, bool):
        return value
    return value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run graph-understanding LLM benchmarks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts without calling LLM APIs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override the list of models.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Override graph sizes.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Override graph types.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Override serialization formats.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Override tasks.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results/).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--local-url",
        default="http://localhost:1234/v1",
        help="Base URL for local model API (default: http://localhost:1234/v1).",
    )
    parser.add_argument(
        "--local-model-name",
        default="qwen3.5-9b",
        help="Model name for local server (default: qwen3.5-9b).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of trials per configuration.",
    )

    args = parser.parse_args()

    # Build config
    if args.config_file:
        with open(args.config_file, "r") as f:
            config_data = json.load(f)
        config = ExperimentConfig(**config_data)
    else:
        config = ExperimentConfig()

    # Apply CLI overrides
    if args.dry_run:
        config.dry_run = True
    if args.models:
        config.models = args.models
    if args.sizes:
        config.graph_sizes = args.sizes
    if args.types:
        config.graph_types = args.types
    if args.formats:
        config.serialization_formats = args.formats
    if args.tasks:
        config.tasks = args.tasks
    if args.output_dir:
        config.results_dir = args.output_dir
    if args.num_trials:
        config.num_trials = args.num_trials

    # Run
    runner = BenchmarkRunner(
        config,
        local_url=args.local_url,
        local_model_name=args.local_model_name,
    )
    results = runner.run()
    runner.save_results(results)


if __name__ == "__main__":
    main()
