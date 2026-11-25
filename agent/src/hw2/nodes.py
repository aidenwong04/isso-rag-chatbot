"""
nodes - the HW2 hosts' graph nodes, shared by the MCP and CodeAct agents.
Adapted from src/hw1/nodes.py.

Same loop as HW1 (next_alert -> guard -> model -> dispatcher), with:
  - an agent (agents.py, codeact.py) that supplies the model's tools, the
    host's trusted operations (its controller), and how a model tool call
    runs: through the MCP server, or as a script / harness action in CodeAct;
  - host-only controls (set_alert, reset_packets) never offered to the model
    and refused by name in the dispatcher (brief line 183);
  - a phase per item: classify, then only after an accepted label 1, evidence,
    continuing the same conversation; label 0 and unclassified items skip
    evidence, traced as runtime decisions (lines 194-195);
  - no-progress ending the item, not the run; step limit and budget ending
    the run.

Progress, per phase (the counter resets on progress and on a phase change):
  classify: a call delivers a passage not delivered before this item, or a
            classify proposal differs from every earlier one this item;
  evidence: a quote is accepted, which also ends the item.
A refused classify also counts toward classify_attempts, a hard cap per item.
"""

import json
from typing import Literal

from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from agents import HOST_ONLY, Outcome
from report import (
    accepted_update,
    build_row,
    classified_row,
    emit_update,
    error_row,
    rejected_update,
    skipped_update,
)
from state import AgentState, RunContext

UNCLASSIFIED = "The item was not classified, so it skips evidence assembly."


def proposal_key(arguments: dict) -> str:
    """A classify proposal as a canonical string, so the same proposal compares equal."""
    evidence = arguments.get("evidence")
    if isinstance(evidence, list) and all(isinstance(e, str) for e in evidence):
        evidence = sorted(evidence)
    return json.dumps(
        [arguments.get("label"), arguments.get("rationale"), evidence],
        sort_keys=True, ensure_ascii=False, default=repr,
    )


def finished(state: AgentState) -> bool:
    """An item is finished exactly when it has its evidence update."""
    return state.get("current_alert_id") in state.get("evidence_updates", {})


def route(runtime: Runtime[RunContext], state: AgentState, decision: str, by: str, **fields) -> str:
    """Writes the routing event for a choice, then returns it."""
    runtime.context.tracer.emit("routing", alert_id=state.get("current_alert_id"),
                                decision=decision, by=by, **fields)
    return decision


# --- discovery -------------------------------------------------------------------------

async def discovery_node(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """
    Runs once. Records the model-facing tools, loads the frozen candidate rows
    (IDs only), empties the set's packets through the controller's host-only
    reset_packets, and prefills every row as pending.
    """
    context = runtime.context
    context.agent.discovery_event(context.tracer)

    with open(context.input_dir / "candidates.json", encoding="utf-8") as f:
        candidates = json.load(f)

    reset = await context.agent.controller.reset_packets(context.query_set)
    if not reset.ok:
        # Packets that may hold earlier evidence would make the run invalid.
        raise RuntimeError(f"reset_packets({context.query_set!r}) failed: {reset.error}")
    context.tracer.emit("state_change", node="discovery", alert_count=len(candidates),
                        packets_reset=reset.payload["query_ids"])

    return {
        "alert_list": candidates,
        "report_rows": {c["alert_id"]: build_row(c) for c in candidates},
        "evidence_updates": {},
        "used_tokens": 0,
        "step_count": 0,
    }


# --- next_alert ------------------------------------------------------------------------

async def next_alert(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """
    Starts the next unfinished item in frozen order: set_alert(classify), then
    current_item for the opening message, then a fresh conversation and
    per-item state. A refused set_alert (or current_item) settles that item as
    error with a skipped update, never retried, and the next one is tried.
    """
    context, tracer = runtime.context, runtime.context.tracer
    controller = context.agent.controller
    done = state.get("evidence_updates", {})
    rows_out, updates_out = {}, {}

    for candidate in state["alert_list"]:
        alert_id = candidate["alert_id"]
        if alert_id in done:
            continue

        outcome = await controller.set_alert(alert_id, "classify")
        if outcome.ok:
            outcome = await controller.current_item()
        if not outcome.ok:
            row = error_row(state["report_rows"][alert_id], f"The host could not start this item: {outcome.error}")
            update = skipped_update(UNCLASSIFIED)
            rows_out[alert_id], updates_out[alert_id] = row, update
            tracer.emit("state_change", alert_id=alert_id, node="next_alert",
                        alert_status=row["status"], label=row["label"])
            emit_update(tracer, alert_id, update)
            continue

        await context.agent.start_item(alert_id)
        tracer.emit("state_change", alert_id=alert_id, node="next_alert", phase="classify")
        return {
            "current_alert_id": alert_id,
            "report_rows": rows_out,
            "evidence_updates": updates_out,
            "message_history": Overwrite(context.agent.opening_messages({**outcome.payload, "alert_id": alert_id})),
            "phase": "classify",
            "observed": set(),
            "no_progress_streak": 0,
            "classify_refusals": 0,
            "seen_proposals": [],
            "write_attempted": False,
            "last_refusal": None,
        }

    tracer.emit("state_change", node="next_alert", terminal_status="completed")
    return {
        "current_alert_id": None,
        "report_rows": rows_out,
        "evidence_updates": updates_out,
        "terminal_status": "completed",
    }


def next_alert_router(state: AgentState, runtime: Runtime[RunContext]) -> Literal["guard", "__end__"]:
    if state.get("terminal_status") is not None:
        return route(runtime, state, END, "runtime")
    return route(runtime, state, "guard", "runtime")


# --- guard -------------------------------------------------------------------------------

def guard(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """
    Runs before every model call. No progress ends the current item; the step
    limit and the token budget end the run. Each fires on reaching its limit.
    No progress is checked first: a stuck item is the more specific diagnosis.
    """
    context, tracer = runtime.context, runtime.context.tracer
    alert_id = state["current_alert_id"]
    limit = context.no_progress_limit

    if state["no_progress_streak"] >= limit:
        if state["phase"] == "classify":
            row = error_row(state["report_rows"][alert_id],
                            f"No progress in {limit} consecutive turns of the classify phase.")
            update = skipped_update(UNCLASSIFIED)
            tracer.emit("state_change", alert_id=alert_id, node="guard",
                        alert_status=row["status"], label=row["label"])
            emit_update(tracer, alert_id, update)
            return {"report_rows": {alert_id: row}, "evidence_updates": {alert_id: update}}

        # Evidence phase: the accepted label stays; only the update fails.
        if state["write_attempted"]:
            update = rejected_update(f"No quote was accepted in {limit} turns of the evidence phase; "
                                     f"last refusal: {state['last_refusal']}")
        else:
            update = skipped_update(f"No quote was proposed in {limit} turns of the evidence phase.")
        emit_update(tracer, alert_id, update)
        return {"evidence_updates": {alert_id: update}}

    if state["step_count"] >= context.max_steps:
        status = "max_steps"
    elif state["used_tokens"] >= context.token_budget:
        status = "budget_exhausted"
    else:
        return {}
    tracer.emit("state_change", alert_id=alert_id, node="guard", terminal_status=status)
    return {"terminal_status": status}


def guard_router(state: AgentState, runtime: Runtime[RunContext]) -> Literal["model", "next_alert", "__end__"]:
    if state.get("terminal_status") is not None:
        return route(runtime, state, END, "runtime")
    if finished(state):
        return route(runtime, state, "next_alert", "runtime")
    return route(runtime, state, "model", "runtime")


# --- model -------------------------------------------------------------------------------

async def model_node(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """
    One step = one model call, with this item's whole conversation and the
    agent's model-facing tools. A tool call is required, one at a time.
    """
    context = runtime.context
    response = await context.model_client.chat.completions.create(
        model=context.model,
        messages=state["message_history"],
        max_completion_tokens=context.output_allowance,
        temperature=context.temperature,
        tools=context.agent.tool_defs,
        tool_choice="required",
        parallel_tool_calls=False,
    )
    message = response.choices[0].message

    # Built by hand so the history holds only fields the API accepts back.
    assistant = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        assistant["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in message.tool_calls
        ]

    usage = {"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens}
    details = getattr(response.usage, "prompt_tokens_details", None)
    if details is not None and getattr(details, "cached_tokens", None) is not None:
        usage["cache_read"] = details.cached_tokens
    context.tracer.emit("model_call", alert_id=state["current_alert_id"], role="agent",
                        model=context.model, usage=usage)

    return {
        "message_history": [assistant],
        "step_count": 1,
        "used_tokens": usage["input"] + usage["output"],
    }


# --- dispatcher --------------------------------------------------------------------------

def tool_reply(call_id: str, content: str) -> dict:
    """The tool message answering one tool call. Every call needs one, refused or not."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


async def dispatcher(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """
    Handles the model's reply, which must be exactly one tool call to an
    offered tool with a JSON-object argument. Everything else is refused here
    (refused_by: dispatcher). Valid calls run through the agent: the MCP
    server, or in CodeAct a script in the worker or a harness action. Their
    refusals come back as errors with their source. Refusals are messages to
    the model, never crashes.
    """
    context, tracer, agent = runtime.context, runtime.context.tracer, runtime.context.agent
    alert_id = state["current_alert_id"]
    phase = state["phase"]
    streak = state["no_progress_streak"]

    def request(name: str, raw: str):
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = None
        # The trace schema needs an object; unparsable arguments are kept raw.
        tracer.emit("tool_request", alert_id=alert_id, tool=name, via=agent.via,
                    args=args if isinstance(args, dict) else {"raw": raw})

    def result_event(name: str, outcome: Outcome):
        fields = dict(outcome.trace)
        if not outcome.ok:
            fields.update(error=outcome.error, error_kind=outcome.error_kind)
            if outcome.refused_by:
                fields["refused_by"] = outcome.refused_by
        tracer.emit("tool_result", alert_id=alert_id, tool=name, via=agent.via, ok=outcome.ok, **fields)

    def dispatcher_refusal(call_id: str, name: str, reason: str) -> dict:
        outcome = Outcome(False, error=f"refused: {reason}", error_kind="refusal", refused_by="dispatcher")
        result_event(name, outcome)
        return tool_reply(call_id, outcome.error)

    def no_progress(messages: list[dict], refusal: str | None = None, **extra) -> dict:
        update = {"message_history": messages, "no_progress_streak": streak + 1, **extra}
        if refusal is not None:
            update["last_refusal"] = refusal
        return update

    calls = state["message_history"][-1].get("tool_calls")

    # No tool call means no id to answer, so the refusal is a user message.
    if not calls:
        msg = "Your reply must be exactly one tool call."
        tracer.emit("state_change", alert_id=alert_id, node="dispatcher", refused=msg)
        return no_progress([{"role": "user", "content": msg}], msg)

    # parallel_tool_calls is off, so this should not happen; every call still needs a reply.
    if len(calls) != 1:
        replies = []
        for call in calls:
            request(call["function"]["name"], call["function"]["arguments"])
            replies.append(dispatcher_refusal(call["id"], call["function"]["name"],
                                              "make exactly one tool call per reply"))
        return no_progress(replies, "refused: make exactly one tool call per reply")

    call_id, name, raw = calls[0]["id"], calls[0]["function"]["name"], calls[0]["function"]["arguments"]
    tracer.emit("routing", alert_id=alert_id, decision=name, by="model")
    request(name, raw)

    # Host-only controls are refused by name even though they were never offered.
    if name in HOST_ONLY:
        return no_progress([dispatcher_refusal(call_id, name, f"{name} is a host-only control, not a tool you can call")],
                           f"refused: {name} is host-only")
    if name not in agent.tool_names:
        return no_progress([dispatcher_refusal(call_id, name, f"there is no tool named {name!r}")],
                           f"refused: no tool named {name!r}")
    try:
        arguments = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        return no_progress([dispatcher_refusal(call_id, name, f"the arguments are not valid JSON: {e}")],
                           "refused: arguments are not valid JSON")
    if not isinstance(arguments, dict):
        return no_progress([dispatcher_refusal(call_id, name, "the arguments must be a JSON object")],
                           "refused: arguments must be a JSON object")

    outcome = await agent.execute(name, arguments, state)
    result_event(name, outcome)
    reply = tool_reply(call_id, outcome.reply())

    if not outcome.ok:
        if name == "classify":
            attempts = state["classify_refusals"] + 1
            key = proposal_key(arguments)
            new = key not in state["seen_proposals"]
            update = {
                "message_history": [reply],
                "classify_refusals": attempts,
                "seen_proposals": state["seen_proposals"] + ([key] if new else []),
                "no_progress_streak": 0 if new else streak + 1,
                "last_refusal": outcome.error,
            }
            if attempts >= context.classify_attempts:
                row = error_row(state["report_rows"][alert_id],
                                f"classify was refused {attempts} times; last reason: {outcome.error}")
                skipped = skipped_update(UNCLASSIFIED)
                tracer.emit("state_change", alert_id=alert_id, node="dispatcher",
                            alert_status=row["status"], label=row["label"])
                emit_update(tracer, alert_id, skipped)
                update["report_rows"] = {alert_id: row}
                update["evidence_updates"] = {alert_id: skipped}
            return update

        extra = {"write_attempted": True} if name == "write_evidence" else {}
        # A failed script can still have shown passages before it failed.
        if outcome.delivered:
            extra["observed"] = state["observed"] | outcome.delivered
        return no_progress([reply], outcome.error, **extra)

    if name == "classify":
        # The check passed; the host settles the model's own arguments.
        label = arguments["label"]
        row = classified_row(state["report_rows"][alert_id], label,
                             arguments["rationale"].strip(), arguments["evidence"])
        tracer.emit("state_change", alert_id=alert_id, node="classify",
                    alert_status="classified", label=label)

        if label != 1:
            route(runtime, state, "skip_evidence", "runtime")
            skipped = skipped_update("Label 0 skips evidence assembly.")
            emit_update(tracer, alert_id, skipped)
            return {"message_history": [reply], "report_rows": {alert_id: row},
                    "evidence_updates": {alert_id: skipped}}

        started = await agent.controller.set_alert(alert_id, "evidence")
        if not started.ok:
            skipped = skipped_update(f"The host could not start the evidence phase: {started.error}")
            emit_update(tracer, alert_id, skipped)
            return {"message_history": [reply], "report_rows": {alert_id: row},
                    "evidence_updates": {alert_id: skipped}}

        route(runtime, state, "assemble_evidence", "runtime", phase="evidence")
        return {
            "message_history": [reply, agent.evidence_message(row["chunk_id"])],
            "report_rows": {alert_id: row},
            "phase": "evidence",
            "no_progress_streak": 0,
        }

    if name == "write_evidence":
        accepted = accepted_update(outcome.payload["diff"], outcome.payload["checks"],
                                   "One item for the current chunk was appended to the current query packet.")
        emit_update(tracer, alert_id, accepted)
        return {"message_history": [reply], "write_attempted": True,
                "evidence_updates": {alert_id: accepted}}

    # Reads (MCP read/read_neighbors, CodeAct run_python): progress when a new
    # passage was delivered, in the classify phase only.
    grew = not outcome.delivered <= state["observed"]
    progressed = grew and phase == "classify"
    return {
        "message_history": [reply],
        "observed": state["observed"] | outcome.delivered,
        "no_progress_streak": 0 if progressed else streak + 1,
    }


def dispatcher_router(state: AgentState, runtime: Runtime[RunContext]) -> Literal["next_alert", "guard"]:
    if finished(state):
        return route(runtime, state, "next_alert", "runtime")
    return route(runtime, state, "guard", "runtime")
