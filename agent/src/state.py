"""
state - the LangGraph state schema for one agent run (A4 in DESIGN.md).

Fields with a reducer accumulate: a node returns only its new piece, and the
reducer merges it in. Fields without one are overwritten: a node returns the
whole new value.

The trace is not in the state. Every node appends its events to trace.jsonl as
they happen, so a crash part way through a run keeps every event written so far.
"""

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from typing_extensions import TypedDict

from discovery import Alert
from tracer import Tracer


@dataclass(frozen=True)
class RunContext:
    """
    Settings fixed for the whole run, decided by the CLI and config before the
    graph starts. Nodes and routers read them as runtime.context; they are not
    state, so no node can replace them. The tracer is the one mutable object
    here: every node appends to the same instance for the whole run.
    """

    input_dir: Path  # the target: a query set folder holding queries.json
    output_dir: Path  # where report.json and trace.jsonl are written
    workspace: Path  # the temporary folder holding the corpus copy the tools read (A7)
    corpus_version: str  # [scan] the pinned corpus (SHA-256 prefix of chunks.json)
    k: int           # [scan] chunks retrieved per query
    model: str       # the OpenAI model id the agent calls
    temperature: float  # sent on every model call; 0 so runs of one prompt are comparable
    token_budget : int
    output_allowance: int  # max output tokens per model call, sent to the provider
    max_steps : int
    no_progress_limit : int
    tracer: Tracer  # writes trace.jsonl; built by the caller before the run


TerminalStatus = Literal["completed", "budget_exhausted", "max_steps", "no_progress", "crashed"]


def merge_rows(old: dict[str, dict], new: dict[str, dict]) -> dict[str, dict]:
    """
    Reducer for report_rows: insert-once.

    old - every row so far, keyed by alert_id. Empty before discovery runs.
    new - the rows one node is writing, keyed by alert_id.

    Returns a new dict with old and new merged; neither argument is changed.
    A row that is still pending may be settled once. Settling a row that is
    already settled is a bug, and raises, which ends the run as crashed (A6).
    """
    if old == {}:
        return new
    
    res = dict(old)

    for key in new:
        if key not in old:
            # this can never happen, all the alerts should be prefilled.
            raise ValueError(f"This key: {key} should not exist.")
        
        # check if its prefilled
        if old[key]["status"] == "pending": # assuming the dict has a 'status' key
            # update the key with new.
            res[key] = new[key]
        # if not prefilled, and alr settled, raise error
        else:
            raise ValueError(f"row {key} is already settled") # again, its a placeholder. 
    
    return res


class AgentState(TypedDict):
    # Per run, written once by discovery.
    alert_list: list[Alert]

    # The alert being worked on. Overwritten by next_alert; None before the
    # first alert and after the last.
    current_alert_id: str | None

    # Per run. Prefilled as pending by discovery, then settled once each.
    report_rows: Annotated[dict[str, dict], merge_rows]

    # Per run, never reset. Nodes return the amount to add.
    used_tokens: Annotated[int, operator.add]
    step_count: Annotated[int, operator.add]

    # Per alert. Plain OpenAI message dicts, exactly what is sent to the model.
    # Nodes return only the messages they add. next_alert empties it with
    # langgraph.types.Overwrite([]), which bypasses the reducer.
    message_history: Annotated[list[dict], operator.add]

    # Per alert, overwritten. Only the tool node and next_alert write these,
    # and both are deterministic code, so each returns the whole new value.
    shown_chunks: set[str]
    no_growth_streak: int
    classify_refusal_count: int

    # Set by the node that ends the run; read by finalize. Before the graph
    # starts, the runner sets it to "crashed" (A6).
    terminal_status: TerminalStatus
