"""
report - report rows, evidence updates, and finalize for the HW2 MCP host.

Adapted from src/hw1/report.py to report.schema.json: rows carry query_id,
chunk_id, chunk-id evidence, and an evidence_update; the report echoes the
corpus and frozen-manifest hashes; rows keep the frozen candidate order
(brief line 93) instead of HW1's path/line sort.
"""

import json

CORPUS_SHA256 = "f186563d6a1e702703386a25e5047e44eac3d666a7fe382d516c86dd47e5bf8e"

# Checks for an update with no proposal to judge (skipped), or whose checks
# the host could not see (a refused write returns only its refusal text).
NO_CHECKS = {"applies": None, "current_item_only": None, "grounded": None}


# --- Classification rows -----------------------------------------------------------

def build_row(candidate: dict) -> dict:
    """The pending row discovery writes for a frozen candidate."""
    return {
        "alert_id": candidate["alert_id"],
        "query_id": candidate["query_id"],
        "chunk_id": candidate["chunk_id"],
        "status": "pending",
        "label": None,
        "rationale": "",
        "evidence": [],
    }


def classified_row(pending: dict, label: int, rationale: str, evidence: list[str]) -> dict:
    """The pending row, settled with the model's accepted proposal."""
    return {**pending, "status": "classified", "label": label, "rationale": rationale, "evidence": list(evidence)}


def error_row(pending: dict, rationale: str) -> dict:
    """The pending row, settled as error, with the runtime's reason as rationale."""
    return {**pending, "status": "error", "label": None, "rationale": rationale[:600], "evidence": []}


def budget_exhausted_row(pending: dict) -> dict:
    """The pending row, settled because a run-wide limit stopped the run first."""
    return {**pending, "status": "budget_exhausted", "label": None, "rationale": "", "evidence": []}


# --- Evidence updates ----------------------------------------------------------------

def skipped_update(reason: str) -> dict:
    return {"status": "skipped", "diff": "", "checks": dict(NO_CHECKS), "reason": reason}


def accepted_update(diff: str, checks: dict, reason: str) -> dict:
    return {"status": "accepted", "diff": diff, "checks": dict(checks), "reason": reason}


def rejected_update(reason: str) -> dict:
    """
    No proposal was accepted. The server refused each write with text only,
    so the host has no diff or check values to record for it.
    """
    return {"status": "rejected", "diff": "", "checks": dict(NO_CHECKS), "reason": reason}


def emit_update(tracer, alert_id: str, update: dict) -> None:
    """The evidence_update state_change event (no alert_status or label: schema rule)."""
    tracer.emit(
        "state_change", alert_id=alert_id, node="evidence_update",
        status=update["status"], checks=update["checks"], reason=update["reason"],
    )


# --- Leftovers at the end of a run -------------------------------------------------

CRASH_RATIONALE = "The run crashed before this item was classified."
LEFTOVER_REASON = {
    "crashed": "The run crashed before this item's evidence stage finished.",
    "budget_exhausted": "The token budget ran out before this item's evidence stage finished.",
    "max_steps": "The step limit was reached before this item's evidence stage finished.",
}
UNCLASSIFIED_REASON = "The item was not classified, so it skips evidence assembly."


def settle_leftover_row(pending: dict, terminal_status: str) -> dict:
    if terminal_status == "crashed":
        return error_row(pending, CRASH_RATIONALE)
    if terminal_status in ("budget_exhausted", "max_steps"):
        return budget_exhausted_row(pending)
    # completed means next_alert found no unfinished item, so one here is a bug.
    raise ValueError(f"row {pending['alert_id']} is pending at terminal status {terminal_status}")


def leftover_update(row: dict, terminal_status: str) -> dict:
    """The update for an item that ended the run without one."""
    if row["status"] == "classified" and row["label"] == 1:
        return skipped_update(LEFTOVER_REASON.get(terminal_status, f"The run ended ({terminal_status})."))
    return skipped_update(UNCLASSIFIED_REASON if row["status"] != "classified" else "Label 0 skips evidence assembly.")


# --- Writing ---------------------------------------------------------------------------

def write_report(alert_order, rows, updates, used_tokens, terminal_status, context):
    alerts = [{**rows[alert_id], "evidence_update": updates[alert_id]} for alert_id in alert_order]
    report = {
        "target": {
            "corpus_sha256": CORPUS_SHA256,
            "query_set": context.query_set,
            "prepared_manifest_sha256": context.manifest_sha256,
        },
        "scan": {"k": context.k, "alert_count": len(alerts)},
        "agent_kind": "mcp",
        "budget": {
            "max_tokens": context.token_budget,
            "used_tokens": used_tokens,
            "exhausted": terminal_status == "budget_exhausted",
        },
        "alerts": alerts,
    }
    with open(context.output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


def finalize(state, terminal_status, context, error=None):
    """
    The end of every run, planned or not. Called once from the runner's
    finally block, so it also runs after a crash. In order:
      1. settle every row still pending, with its state_change event
      2. give every item without an evidence update a skipped one, with its event
      3. write report.json
      4. write the terminal event, always last
    """
    tracer = context.tracer
    order = [c["alert_id"] for c in state.get("alert_list", [])]
    rows = dict(state.get("report_rows", {}))
    updates = dict(state.get("evidence_updates", {}))

    for alert_id in order:
        row = rows[alert_id]
        if row["status"] == "pending":
            row = rows[alert_id] = settle_leftover_row(row, terminal_status)
            tracer.emit("state_change", alert_id=alert_id, node="finalize",
                        alert_status=row["status"], label=row["label"])
        if alert_id not in updates:
            update = updates[alert_id] = leftover_update(row, terminal_status)
            emit_update(tracer, alert_id, update)

    try:
        write_report(order, rows, updates, state.get("used_tokens", 0), terminal_status, context)
    except BaseException:
        terminal_status = "crashed"
        raise
    finally:
        tracer.emit("terminal", status=terminal_status)
