"""
state - the LangGraph state schema and run context for the HW2 MCP host.

Adapted from src/hw1/state.py. What changed for HW2:
  - RunContext carries the agent (MCP or CodeAct), which holds the tools and
    the host's trusted operations, and no workspace: the MCP host never opens
    target files (brief line 151).
  - Items are the frozen candidate rows, read from <set>/candidates.json (IDs
    only, which the host treats as control metadata, brief line 152).
  - Each item has a phase, classify then (only after an accepted label 1)
    evidence, and each finished item gets one evidence update.

Fields with a reducer accumulate: a node returns only its new piece. Fields
without one are overwritten. The trace is not in the state: nodes append to
trace.jsonl as they go, so a crash keeps every event written so far.
"""

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from typing_extensions import TypedDict

from tracer import Tracer


@dataclass(frozen=True)
class RunContext:
    """
    Settings fixed for the whole run, built by the CLI before the graph starts.
    Nodes read them as runtime.context; no node can replace them. The tracer,
    the agent, and the model client are the mutable objects here: every node
    uses the same instances for the whole run.
    """

    input_dir: Path        # the prepared query set folder, <prepared>/<set>
    output_dir: Path       # where report.json and trace.jsonl are written
    query_set: str         # "dev", "heldout-a", or "heldout-b"
    manifest_sha256: str   # SHA-256 of the frozen manifest the report echoes
    k: int                 # [scan] candidates per question
    model: str             # the model id the agent calls
    temperature: float     # sent on every model call
    token_budget: int      # [budget] max_tokens: run-wide input + output charge
    output_allowance: int  # max output tokens per model call
    max_steps: int         # run-wide model calls
    no_progress_limit: int  # per phase: turns without progress before the item ends
    classify_attempts: int  # refused classify proposals allowed per item
    agent: Any             # MCPAgent or CodeActAgent (agents.py, codeact.py): the tools, the
                           # host's trusted operations, and how a model tool call runs
    model_client: Any      # AsyncOpenAI, or a stand-in with the same interface in tests
    tracer: Tracer


TerminalStatus = Literal["completed", "budget_exhausted", "max_steps", "crashed"]
Phase = Literal["classify", "evidence"]


def insert_once(old: dict[str, dict], new: dict[str, dict]) -> dict[str, dict]:
    """
    Reducer for evidence_updates: each item gets exactly one, never replaced.
    Writing a second one for an item is a bug and ends the run as crashed.
    """
    res = dict(old)
    for key, value in new.items():
        if key in old:
            raise ValueError(f"evidence update for {key} is already recorded")
        res[key] = value
    return res


def merge_rows(old: dict[str, dict], new: dict[str, dict]) -> dict[str, dict]:
    """
    Reducer for report_rows, as in HW1: prefilled as pending, then each row
    is settled once. Settling an unknown or already settled row raises.
    """
    if old == {}:
        return new
    res = dict(old)
    for key in new:
        if key not in old:
            raise ValueError(f"row {key} was never prefilled")
        if old[key]["status"] != "pending":
            raise ValueError(f"row {key} is already settled")
        res[key] = new[key]
    return res


class AgentState(TypedDict):
    # Per run, written once by discovery: the frozen candidate rows in order.
    alert_list: list[dict]

    # The item being worked on; None before the first and after the last.
    current_alert_id: str | None

    # Per run. Classification rows, prefilled as pending and settled once.
    report_rows: Annotated[dict[str, dict], merge_rows]

    # Per run. One evidence update per finished item (accepted, rejected, or
    # skipped). An item is finished exactly when it has one.
    evidence_updates: Annotated[dict[str, dict], insert_once]

    # Per run, never reset.
    used_tokens: Annotated[int, operator.add]
    step_count: Annotated[int, operator.add]

    # Per item. The conversation sent to the model; next_alert replaces it,
    # and the evidence phase continues it.
    message_history: Annotated[list[dict], operator.add]

    # Per item, overwritten.
    phase: Phase | None
    observed: set[str]            # chunk ids the server returned to the model this item
    no_progress_streak: int       # turns without progress in the current phase
    classify_refusals: int        # refused classify proposals this item
    seen_proposals: list[str]     # classify proposals this item, each as a canonical JSON string
    write_attempted: bool         # whether write_evidence was called this item
    last_refusal: str | None      # the last refusal text this item, for a rejected update's reason

    # Set by the node that ends the run; read by finalize.
    terminal_status: TerminalStatus
