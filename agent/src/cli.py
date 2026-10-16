"""
cli - the agent's command line (handout section 5).

    ./isso-agent --input agent/queries/dev --output runs/dev-full --config configs/full.toml

Reads the TOML config, checks it, builds the tracer and the run context, and
runs the graph once. Writes report.json and trace.jsonl into --output.

A bad config or input stops here, before the tracer exists: no run started,
so there is no trace or report to write. Every run that does start ends in
finalize, planned or crashed.

Exit status: 0 for every planned ending (completed, budget_exhausted,
max_steps, no_progress), since those are the agent working as designed; a
crash re-raises, so Python exits non-zero with the traceback.
"""

import argparse
import sys
import tomllib
from pathlib import Path

from graph import build_graph
from query import corpus_version
from run import run
from state import RunContext
from tracer import Tracer

# section -> key -> type. Every key is required; other keys are allowed (section 5).
REQUIRED = {
    "scan": {"corpus_version": str, "k": int},
    "model": {"provider": str, "name": str},
    "budget": {
        "max_tokens": int,
        "output_allowance": int,
        "max_steps": int,
        "no_progress_limit": int,
    },
}
TYPE_NAMES = {str: "string", int: "integer"}
# The smallest value each integer key may take (section 5).
MINIMUM = {"k": 1, "max_tokens": 0, "output_allowance": 1, "max_steps": 1, "no_progress_limit": 1}


def load_config(path):
    """The parsed config, or exits with every problem it found."""
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(f"config {path}: {e}")

    problems = []
    for section, keys in REQUIRED.items():
        table = config.get(section)
        if not isinstance(table, dict):
            problems.append(f"missing [{section}]")
            continue
        for key, kind in keys.items():
            value = table.get(key)
            # bool is a subclass of int in Python, so true would pass as 1.
            if not isinstance(value, kind) or isinstance(value, bool):
                problems.append(f"[{section}] {key} must be a {TYPE_NAMES[kind]}")
            elif kind is int and value < MINIMUM[key]:
                problems.append(f"[{section}] {key} must be >= {MINIMUM[key]}")

    if config.get("model", {}).get("provider") not in (None, "openai"):
        problems.append('[model] provider must be "openai", the only provider the agent calls')

    if problems:
        sys.exit(f"config {path}:\n  " + "\n  ".join(problems))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="query set folder, holding queries.json")
    parser.add_argument("--output", type=Path, required=True, help="folder for report.json and trace.jsonl")
    parser.add_argument("--config", type=Path, required=True, help="TOML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    scan, budget = config["scan"], config["budget"]

    if not (args.input / "queries.json").is_file():
        sys.exit(f"--input {args.input}: no queries.json in it")

    # The labels are tied to one corpus; a run against any other is not comparable.
    actual = corpus_version()
    if scan["corpus_version"] != actual:
        sys.exit(f"[scan] corpus_version is {scan['corpus_version']}, but the corpus on disk is {actual}")

    args.output.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(args.output / "trace.jsonl")
    context = RunContext(
        input_dir=args.input,
        output_dir=args.output,
        corpus_version=actual,
        k=scan["k"],
        model=config["model"]["name"],
        token_budget=budget["max_tokens"],
        output_allowance=budget["output_allowance"],
        max_steps=budget["max_steps"],
        no_progress_limit=budget["no_progress_limit"],
        tracer=tracer,
    )
    status = run(build_graph(), context)
    print(f"{status}: {args.output / 'report.json'}, {args.output / 'trace.jsonl'}")


if __name__ == "__main__":
    main()
