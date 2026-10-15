"""
nodes - one function per node of the graph (A3 in DESIGN.md).

Each node takes the state, and the run's fixed settings as runtime.context,
and returns only the state fields it changes.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

from typing import Literal
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from discovery import discover
from report import build_row
from state import AgentState, RunContext
from prompt import opening_messages
from tools import TOOL_DEFINITIONS

_client_cache = None


def openai_client():
    """One OpenAI client per process, created on first use, like get_client() in src/query.py."""
    global _client_cache
    if _client_cache is None:
        # The repo's .env, the same file src/query.py reads the Gemini key from.
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (shell env or .env file)")
        _client_cache = OpenAI(api_key=api_key)
    return _client_cache

def discovery_node(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """
    Runs once: builds every alert and prefills its report row as pending.

    shape of report_rows:

    {
    "alert_id" : row of type {"alert_id": alert.alert_id, "test_id": alert.query_id,}
        (see report.py for shape of row.)
    }

    """
    alerts = discover(runtime.context.input_dir / "queries.json", runtime.context.k)
    return {
        "alert_list": alerts,
        "report_rows": {alert.alert_id: build_row(alert) for alert in alerts},
    }

def next_alert(state: AgentState, runtime: Runtime[RunContext]):
    """
    Picks the next pending alert and gives it a fresh per-alert state, or sets
    completed when none is left. Alerts are taken in discovery's order.
    """
    rows = state["report_rows"]
    for alert in state["alert_list"]:
        if rows[alert.alert_id]["status"] == "pending":
            return {
                "current_alert_id": alert.alert_id,
                # Overwrite bypasses the append reducer: a fresh conversation.
                "message_history": Overwrite(opening_messages(alert)),
                # The alert's own chunk is shown from the start (A4).
                "shown_chunks": {alert.chunk.chunk_id},
                "no_growth_streak": 0,
                "classify_refusal_count": 0,
            }
    return {
        "current_alert_id": None,
        "terminal_status": "completed",
    }


def guard(state: AgentState, runtime: Runtime[RunContext]):
    """
    Runs before every model call. Writes terminal_status when a limit is
    reached, and nothing otherwise; the routing function after it decides
    where the run goes.

    Each limit fires on reaching it, not on passing it: max_steps = 40 allows
    40 model calls, and no_progress_limit = 3 fires on the third step in a row
    that showed nothing new.

    When two limits are reached on the same step, no_progress wins: a stuck
    alert is the more specific diagnosis, and it settles that alert as error
    rather than budget_exhausted (A6).
    """
    limits = runtime.context

    if state["no_growth_streak"] >= limits.no_progress_limit:
        return {"terminal_status": "no_progress"}

    if state["step_count"] >= limits.max_steps:
        return {"terminal_status": "max_steps"}

    if state["used_tokens"] >= limits.token_budget:
        return {"terminal_status": "budget_exhausted"}

    return {}

def next_alert_router(state: AgentState) -> Literal["guard", "__end__"]:
    # next_alert sets completed when no pending alert is left; otherwise the
    # alert it picked goes to the guard before its first model call.
    if state.get("terminal_status") is not None:
        return END
    return "guard"


def guard_router(state: AgentState) -> Literal["model", "__end__"] :
    if state.get("terminal_status") is not None:
        return END
    return "model"

def model_node(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """
    One step = one model call (A3). Sends this alert's whole conversation to
    the model and records the reply. Never routes: a fixed edge takes the
    reply to the dispatcher.

    Reads:
        state["message_history"] - this alert's conversation so far. The
            first call of an alert must already find the system and user
            messages here (decide which node adds them).
        runtime.context - the model id and any request settings, once they
            are added to RunContext.

    Calls the OpenAI API with:
        - the message history, unchanged
        - the three tool definitions (read, read_neighbors, classify), each
          with a JSON schema for its arguments
        - tool_choice set so that a tool call is required
        - parallel_tool_calls=False

    Returns:
        "message_history": [the assistant message, as a plain dict], appended
            by the reducer. Keep its tool_calls exactly as received: the
            dispatcher answers each one by its id.
        "step_count": 1, summed by the reducer.
        "used_tokens": the tokens this call used, from the response's usage,
            summed by the reducer.

    Errors from the API (network, rate limit, bad request) are not caught
    here: they end the run as crashed (A6), unless you decide to retry.
    """
    response = openai_client().chat.completions.create(
        model=runtime.context.model,
        messages=state["message_history"],
        tools=TOOL_DEFINITIONS,
        tool_choice="required",
        parallel_tool_calls=False,
    )
    message = response.choices[0].message

    # Built by hand rather than dumped from the SDK object, so the history
    # holds only fields the API accepts back on the next call.
    assistant = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        assistant["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]

    return {
        "message_history": [assistant],
        "step_count": 1,
        # Input and output together: every call resends the whole
        # conversation, and all of it is paid for.
        "used_tokens": response.usage.total_tokens,
    }


