"""
cli - the HW2 agent's command line (MCP agent; CodeAct is not built yet).

    agent/isso-agent-hw2 --input agent/prepared/dev --output agent/runs/hw2-mcp-dev \
        --config configs/mcp.toml

--input is a prepared query set folder (<prepared>/<set>). The host reads only
its candidates.json (IDs) and the frozen manifest's hash and corpus version,
the control metadata it needs (brief line 152); passages, questions, and
packets reach it only through the server's tools.

Startup, in order: validate the config against schemas/config.schema.json;
check the input; start the server from mcp.json ([mcp] config) over stdio;
initialize; discover its tools with tools/list; offer the model every tool
except the host-only controls; run the graph inside the open session.
Writes report.json and trace.jsonl into --output.

Exit status 0 for every planned ending (completed, budget_exhausted,
max_steps); a crash re-raises.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from graph import build_graph
from nodes import HOST_ONLY
from run import run
from state import RunContext
from tracer import Tracer

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCHEMA = REPO_ROOT / "agent" / "schemas" / "config.schema.json"
RUNS_DIR = REPO_ROOT / "agent" / "runs"
QUERY_SETS = ("dev", "heldout-a", "heldout-b")

# Refused classify proposals allowed per item, as in HW1. Overridable with
# [budget] classify_attempts (the config schema allows extra settings).
DEFAULT_CLASSIFY_ATTEMPTS = 3


def load_config(path: Path) -> dict:
    """The parsed config, or exits listing every schema problem."""
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(f"config {path}: {e}")

    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    problems = [f"{'/'.join(map(str, e.absolute_path)) or '(top)'}: {e.message}"
                for e in Draft202012Validator(schema).iter_errors(config)]
    if config.get("model", {}).get("provider") != "openai":
        problems.append('model/provider: "openai" is the only provider the agent calls')
    if problems:
        sys.exit(f"config {path}:\n  " + "\n  ".join(problems))
    return config


def server_params(config: dict, mcp_json: Path | None = None) -> tuple[str, StdioServerParameters]:
    """
    The single server entry of the mcp.json named by [mcp] config (relative to
    the repository root), as stdio parameters. mcp_json overrides that file:
    tests use it to point the server at a temporary evidence folder.
    """
    mcp_json = mcp_json or REPO_ROOT / config["mcp"]["config"]
    servers = json.loads(mcp_json.read_text(encoding="utf-8"))["mcpServers"]
    if len(servers) != 1:
        sys.exit(f"{mcp_json}: expected exactly one server, found {sorted(servers)}")
    name, entry = next(iter(servers.items()))
    # Relative paths in mcp.json are relative to the repository root.
    return name, StdioServerParameters(command=entry["command"], args=entry.get("args", []),
                                       env=entry.get("env") or None, cwd=REPO_ROOT)


def openai_tool(tool) -> dict:
    """A discovered MCP tool in the OpenAI Chat Completions tool format."""
    return {"type": "function",
            "function": {"name": tool.name, "description": tool.description or "",
                         "parameters": tool.input_schema}}


def default_model_client():
    """AsyncOpenAI with the key from the shell or the repository's .env."""
    from openai import AsyncOpenAI
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set (shell env or .env file)")
    return AsyncOpenAI(api_key=api_key)


async def run_agent(input_dir: Path, output_dir: Path, config_path: Path,
                    model_client=None, mcp_json: Path | None = None) -> str:
    config = load_config(config_path)
    if config["agent"]["kind"] != "mcp":
        sys.exit(f'[agent] kind {config["agent"]["kind"]!r}: only "mcp" is implemented so far')

    query_set = input_dir.name
    if query_set not in QUERY_SETS or not (input_dir / "candidates.json").is_file():
        sys.exit(f"--input {input_dir}: expected a prepared query set folder ({', '.join(QUERY_SETS)}) "
                 "holding candidates.json")
    manifest_path = input_dir.parent / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"--input {input_dir}: no manifest.json next to it")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_version = json.loads(manifest_path.read_text(encoding="utf-8")).get("corpus_version")
    if manifest_version != config["scan"]["corpus_version"]:
        sys.exit(f"[scan] corpus_version {config['scan']['corpus_version']} does not match the "
                 f"manifest's {manifest_version}")

    model_client = model_client or default_model_client()
    budget = config["budget"]
    server_name, params = server_params(config, mcp_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            names = {t.name for t in listed}
            missing = HOST_ONLY - names
            if missing:
                sys.exit(f"server {server_name!r} does not offer the host-only controls {sorted(missing)}")
            # The model sees every discovered tool except the host-only controls (brief line 147).
            offered = [t for t in listed if t.name not in HOST_ONLY]

            context = RunContext(
                input_dir=input_dir,
                output_dir=output_dir,
                query_set=query_set,
                manifest_sha256=manifest_sha256,
                k=config["scan"]["k"],
                model=config["model"]["name"],
                temperature=config["model"].get("temperature", 0),
                token_budget=budget["max_tokens"],
                output_allowance=budget["output_allowance"],
                max_steps=budget["max_steps"],
                no_progress_limit=budget["no_progress_limit"],
                classify_attempts=budget.get("classify_attempts", DEFAULT_CLASSIFY_ATTEMPTS),
                server_name=server_name,
                session=session,
                tool_defs=[openai_tool(t) for t in offered],
                tool_names=frozenset(t.name for t in offered),
                model_client=model_client,
                tracer=Tracer(output_dir / "trace.jsonl"),
            )
            return await run(build_graph(), context)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="prepared query set folder, e.g. agent/prepared/dev")
    parser.add_argument("--output", type=Path,
                        help="folder for report.json and trace.jsonl (default: agent/runs/hw2-<config>-<set>)")
    parser.add_argument("--config", type=Path, required=True, help="TOML config, e.g. configs/mcp.toml")
    args = parser.parse_args()
    output = args.output or RUNS_DIR / f"hw2-{args.config.stem}-{args.input.name}"

    status = asyncio.run(run_agent(args.input, output, args.config))
    print(f"{status}: {output / 'report.json'}, {output / 'trace.jsonl'}")


if __name__ == "__main__":
    main()
