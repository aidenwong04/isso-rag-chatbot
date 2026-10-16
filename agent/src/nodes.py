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
    runtime.context.tracer.emit("state_change", node="discovery", alert_count=len(alerts))
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
            runtime.context.tracer.emit("state_change", alert_id=alert.alert_id, node="next_alert")
            return {
                "current_alert_id": alert.alert_id,
                # Overwrite bypasses the append reducer: a fresh conversation.
                "message_history": Overwrite(opening_messages(alert, runtime.context.workspace)),
                # The alert's own chunk is shown from the start (A4).
                "shown_chunks": {alert.chunk.chunk_id},
                "no_growth_streak": 0,
                "classify_refusal_count": 0,
            }
    runtime.context.tracer.emit("state_change", node="next_alert", terminal_status="completed")
    return {
        "current_alert_id": None,
        "terminal_status": "completed",
    }

def route(runtime: Runtime[RunContext], state: AgentState, decision: str, by: str) -> str:
    """Writes the routing event for a choice an edge function made, then returns the choice."""
    runtime.context.tracer.emit("routing", alert_id=state.get("current_alert_id"), decision=decision, by=by)
    return decision

def next_alert_router(state: AgentState, runtime: Runtime[RunContext]) -> Literal["guard", "__end__"]:
    # next_alert sets completed when no pending alert is left; otherwise the
    # alert it picked goes to the guard before its first model call.
    if state.get("terminal_status") is not None:
        return route(runtime, state, END, "runtime")
    return route(runtime, state, "guard", "runtime")


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
        status = "no_progress"
    elif state["step_count"] >= limits.max_steps:
        status = "max_steps"
    elif state["used_tokens"] >= limits.token_budget:
        status = "budget_exhausted"
    else:
        return {}

    limits.tracer.emit("state_change", alert_id=state["current_alert_id"], node="guard", terminal_status=status)
    return {"terminal_status": status}

def guard_router(state: AgentState, runtime: Runtime[RunContext]) -> Literal["model", "__end__"] :
    if state.get("terminal_status") is not None:
        return route(runtime, state, END, "runtime")
    return route(runtime, state, "model", "runtime")


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
        max_completion_tokens=runtime.context.output_allowance,
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

    # The charge is input + output exactly as the provider reported them (B6).
    # OpenAI reports cached input inside prompt_tokens; it is recorded as
    # cache_read but not charged separately. It has no cache_write counter.
    usage = {"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens}
    details = response.usage.prompt_tokens_details
    if details is not None and details.cached_tokens is not None:
        usage["cache_read"] = details.cached_tokens
    runtime.context.tracer.emit(
        "model_call",
        alert_id=state["current_alert_id"],
        role="agent",
        model=runtime.context.model,
        usage=usage,
    )

    return {
        "message_history": [assistant],
        "step_count": 1,
        # Input and output together: every call resends the whole
        # conversation, and all of it is paid for.
        "used_tokens": usage["input"] + usage["output"],
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
    tracer = runtime.context.tracer
    alert_id = state["current_alert_id"]
    streak = state["no_growth_streak"]

    def refuse(messages):
        return {"message_history": messages, "no_growth_streak": streak + 1}

    def request(name, raw_arguments):
        """Writes the tool_request event. args is the parsed object when it parses, else the raw text."""
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError:
            args = raw_arguments
        tracer.emit("tool_request", alert_id=alert_id, tool=name, args=args)

    def answer(call_id, name, result):
        """Writes the tool_result event right after its request (invariant 6), and returns the reply."""
        error = {} if result["ok"] else {"error": result["msg"]}
        tracer.emit("tool_result", alert_id=alert_id, tool=name, ok=result["ok"], **error)
        return tool_reply(call_id, result)

    model_response = state["message_history"][-1]
    calls = model_response.get("tool_calls")

    # No tool call means no id to answer, so this refusal is a user message.
    # There is no request to trace, so the refusal is a state change.
    if not calls:
        msg = "Your reply must be exactly one tool call."
        tracer.emit("state_change", alert_id=alert_id, node="dispatcher", refused=msg)
        return refuse([{"role": "user", "content": msg}])

    # parallel_tool_calls is off, so this should not happen. If it does, every
    # call still needs its own reply, or the next request is rejected.
    if len(calls) != 1:
        result = {"ok": False, "msg": "Make exactly one tool call per reply."}
        replies = []
        for call in calls:
            request(call["function"]["name"], call["function"]["arguments"])
            replies.append(answer(call["id"], call["function"]["name"], result))
        return refuse(replies)

    call_id = calls[0]["id"]
    name = calls[0]["function"]["name"]
    # Which tool runs is the model's choice; the runtime only validates it (A3).
    tracer.emit("routing", alert_id=alert_id, decision=name, by="model")
    request(name, calls[0]["function"]["arguments"])

    if name not in {"read", "read_neighbors", "classify"}:
        return refuse([answer(call_id, name, {"ok": False, "msg": f"There is no tool named {name!r}."})])

    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except json.JSONDecodeError as e:
        return refuse([answer(call_id, name, {"ok": False, "msg": f"The arguments are not valid JSON: {e}"})])
    if not isinstance(arguments, dict):
        return refuse([answer(call_id, name, {"ok": False, "msg": "The arguments must be a JSON object."})])

    if name in ("read", "read_neighbors"):
        if "chunk_id" not in arguments:
            return refuse([answer(call_id, name, {"ok": False, "msg": f"{name} needs a chunk_id argument."})])

        tool = read if name == "read" else read_neighbors
        tool_output = tool(arguments["chunk_id"], runtime.context.workspace)
        if not tool_output["ok"]:
            return refuse([answer(call_id, name, tool_output)])

        # The set grew if any returned chunk was not already shown. A read of
        # an already shown chunk succeeds but shows nothing new.
        returned = {chunk["chunk_id"] for chunk in tool_output["chunks"]}
        shown = state["shown_chunks"]
        grew = not returned <= shown
        return {
            "message_history": [answer(call_id, name, tool_output)],
            # Overwritten, so the whole set: everything shown so far, plus these.
            "shown_chunks": shown | returned,
            "no_growth_streak": 0 if grew else streak + 1,
        }

    # name is "classify" here: the name check above allows only three tools.
    pending = state["report_rows"][alert_id]
    label = arguments.get("label")
    rationale = arguments.get("rationale")
    evidence = arguments.get("evidence")

    tool_output = classify(label, rationale, evidence, state["shown_chunks"], runtime.context.workspace)

    if tool_output["ok"]:
        reply = answer(call_id, name, tool_output)
        row = classified_row(pending, label, rationale.strip(), evidence)
        tracer.emit("state_change", alert_id=alert_id, node="dispatcher", alert_status=row["status"], label=row["label"])
        return {"message_history": [reply], "report_rows": {alert_id: row}}

    attempt = state["classify_refusal_count"] + 1
    reason = tool_output["msg"]
    update = {
        "message_history": [
            answer(call_id, name, {"ok": False, "msg": f"{reason} (classify attempt {attempt} of {CLASSIFY_ATTEMPTS})"})
        ],
        "classify_refusal_count": attempt,
    }
    if attempt >= CLASSIFY_ATTEMPTS:
        # The alert ends as error; the run goes on to the next alert (A5).
        row = error_row(pending, f"classify was refused {attempt} times; last reason: {reason}.")
        tracer.emit("state_change", alert_id=alert_id, node="dispatcher", alert_status=row["status"], label=row["label"])
        update["report_rows"] = {alert_id: row}
    return update

def dispatcher_router(state: AgentState, runtime: Runtime[RunContext]) -> Literal["next_alert", "guard"]:
    # A settled row (classified, or error after the third refused classify)
    # means this alert is over; anything else goes back through the guard
    # before the next model call.
    if state["report_rows"][state["current_alert_id"]]["status"] != "pending":
        return route(runtime, state, "next_alert", "runtime")
    return route(runtime, state, "guard", "runtime")
