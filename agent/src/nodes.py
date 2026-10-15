"""
nodes - one function per node of the graph (A3 in DESIGN.md).

Each node takes the state, and the run's fixed settings as runtime.context,
and returns only the state fields it changes.
"""
import os
import json
from dotenv import load_dotenv
from pathlib import Path

from typing import Literal
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from discovery import discover
from report import build_row, classified_row, error_row
from state import AgentState, RunContext
from prompt import opening_messages
from tools import TOOL_DEFINITIONS, read, read_neighbors, classify

# A refused classify may be retried; the third refusal settles the alert as error (A5).
CLASSIFY_ATTEMPTS = 3

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

def next_alert_router(state: AgentState) -> Literal["guard", "__end__"]:
    # next_alert sets completed when no pending alert is left; otherwise the
    # alert it picked goes to the guard before its first model call.
    if state.get("terminal_status") is not None:
        return END
    return "guard"


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

def tool_reply(call_id: str, result: dict) -> dict:
    """The tool message answering one tool call. Every call the model makes needs one, refused or not."""
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)}


def dispatcher(state: AgentState, runtime: Runtime[RunContext]):
    """
    Handles the model's last reply, which should be exactly one tool call.
    Refusals go back to the model as messages, never as crashes. A refused
    read, read_neighbors or malformed reply counts as a step that showed
    nothing new; a refused classify counts toward its own cap instead (A5).
    The router after this node sends a settled alert to next_alert, and
    everything else back to the guard.
    """
    streak = state["no_growth_streak"]

    def refuse(messages):
        return {"message_history": messages, "no_growth_streak": streak + 1}

    model_response = state["message_history"][-1]
    calls = model_response.get("tool_calls")

    # No tool call means no id to answer, so this refusal is a user message.
    if not calls:
        return refuse([{"role": "user", "content": "Your reply must be exactly one tool call."}])

    # parallel_tool_calls is off, so this should not happen. If it does, every
    # call still needs its own reply, or the next request is rejected.
    if len(calls) != 1:
        result = {"ok": False, "msg": "Make exactly one tool call per reply."}
        return refuse([tool_reply(call["id"], result) for call in calls])

    call_id = calls[0]["id"]
    name = calls[0]["function"]["name"]

    if name not in {"read", "read_neighbors", "classify"}:
        return refuse([tool_reply(call_id, {"ok": False, "msg": f"There is no tool named {name!r}."})])

    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except json.JSONDecodeError as e:
        return refuse([tool_reply(call_id, {"ok": False, "msg": f"The arguments are not valid JSON: {e}"})])
    if not isinstance(arguments, dict):
        return refuse([tool_reply(call_id, {"ok": False, "msg": "The arguments must be a JSON object."})])

    if name in ("read", "read_neighbors"):
        if "chunk_id" not in arguments:
            return refuse([tool_reply(call_id, {"ok": False, "msg": f"{name} needs a chunk_id argument."})])

        tool = read if name == "read" else read_neighbors
        tool_output = tool(arguments["chunk_id"])
        if not tool_output["ok"]:
            return refuse([tool_reply(call_id, tool_output)])

        # The set grew if any returned chunk was not already shown. A read of
        # an already shown chunk succeeds but shows nothing new.
        returned = {chunk["chunk_id"] for chunk in tool_output["chunks"]}
        shown = state["shown_chunks"]
        grew = not returned <= shown
        return {
            "message_history": [tool_reply(call_id, tool_output)],
            # Overwritten, so the whole set: everything shown so far, plus these.
            "shown_chunks": shown | returned,
            "no_growth_streak": 0 if grew else streak + 1,
        }

    # name is "classify" here: the name check above allows only three tools.
    alert_id = state["current_alert_id"]
    pending = state["report_rows"][alert_id]
    label = arguments.get("label")
    rationale = arguments.get("rationale")
    evidence = arguments.get("evidence")

    tool_output = classify(label, rationale, evidence, state["shown_chunks"])

    if tool_output["ok"]:
        return {
            "message_history": [tool_reply(call_id, tool_output)],
            "report_rows": {alert_id: classified_row(pending, label, rationale.strip(), evidence)},
        }

    attempt = state["classify_refusal_count"] + 1
    reason = tool_output["msg"]
    update = {
        "message_history": [
            tool_reply(call_id, {"ok": False, "msg": f"{reason} (classify attempt {attempt} of {CLASSIFY_ATTEMPTS})"})
        ],
        "classify_refusal_count": attempt,
    }
    if attempt >= CLASSIFY_ATTEMPTS:
        # The alert ends as error; the run goes on to the next alert (A5).
        update["report_rows"] = {
            alert_id: error_row(pending, f"classify was refused {attempt} times; last reason: {reason}.")
        }
    return update

def dispatcher_router(state: AgentState) -> Literal["next_alert", "guard"]:
    # A settled row (classified, or error after the third refused classify)
    # means this alert is over; anything else goes back through the guard
    # before the next model call.
    if state["report_rows"][state["current_alert_id"]]["status"] != "pending":
        return "next_alert"
    return "guard"