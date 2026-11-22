import json

from discovery import Alert

def build_row(alert):
    """The pending row discovery writes for an alert, before the agent sees it."""
    return {
        "alert_id": alert.alert_id,
        "test_id": alert.query_id,
        "path": alert.page,
        "line": alert.chunk_index,
        # Bandit's severity has no counterpart
        "severity": None,
        # The retriever's own confidence that the chunk matches the query.
        "confidence": alert.chunk.score,
        "status": "pending",
        # Settled later. Pending rows match error rows: no label, no evidence.
        "label": None,
        "rationale": "",
        "evidence": [],
    }

def chunk_pair(chunk_id):
    """A chunk id ("page::3") as the report points at chunks: {"path": "page", "line": 3}."""
    page, index = chunk_id.rsplit("::", 1)
    return {"path": page, "line": int(index)}


def classified_row(pending, label, rationale, evidence):
    """
    The alert's pending row, settled with the model's accepted judgment.

    pending  - the row discovery wrote; it is copied, never changed.
    evidence - chunk ids that classify has already validated against the
               shown set, converted here to {path, line} pairs.
    """
    return {
        **pending,
        "status": "classified",
        "label": label,
        "rationale": rationale,
        "evidence": [chunk_pair(chunk_id) for chunk_id in evidence],
    }


def error_row(pending, rationale):
    """
    The alert's pending row, settled as error. The rationale is written by the
    runtime and says what went wrong (A5, "Error rows").
    """
    return {
        **pending,
        "status": "error",
        "label": None,
        "rationale": rationale,
        "evidence": [],
    }

def budget_exhausted_row(pending):
    """
    The alert's pending row, settled because the run stopped before the alert
    was classified, whichever limit stopped it (A6).
    """
    return {
        **pending,
        "status": "budget_exhausted",
        "label": None,
        "rationale": "",
        "evidence": [],
    }


CRASH_RATIONALE = "The run crashed before this alert was classified."
NO_PROGRESS_RATIONALE = "No new chunk was shown in {n} consecutive steps."


def settle_leftover(pending, terminal_status, in_progress, context):
    """
    The settled row for an alert still pending when the run ended, by the A6
    table: the alert in progress and the unstarted alerts can differ.
    """
    if terminal_status == "crashed":
        return error_row(pending, CRASH_RATIONALE)
    if terminal_status == "no_progress" and in_progress:
        return error_row(pending, NO_PROGRESS_RATIONALE.format(n=context.no_progress_limit))
    if terminal_status in ("budget_exhausted", "max_steps", "no_progress"):
        return budget_exhausted_row(pending)
    # completed means next_alert found no pending row, so one here is a bug.
    raise ValueError(f"row {pending['alert_id']} is pending at terminal status {terminal_status}")


def row_order(row):
    """Section 6 order: path as a plain string, then line as an int, then test_id."""
    return (row["path"], row["line"], row["test_id"])


def write_report(rows, used_tokens, terminal_status, context):
    alerts = sorted(rows.values(), key=row_order)
    report = {
        "target": {"input": str(context.input_dir)},
        "scan": {"corpus_version": context.corpus_version, "k": context.k, "alert_count": len(alerts)},
        "budget": {
            "max_tokens": context.token_budget,
            "used_tokens": used_tokens,
            "exhausted": terminal_status == "budget_exhausted",
        },
        "alerts": alerts,
    }
    with open(context.output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)


def finalize(state, terminal_status, context, error=None):
    """
    The end of every run, planned or not (A6). Called once, from the runner's
    finally block, never as a graph node, so it also runs after a crash.

    state           - the last full state the runner saw. After a crash this is
                      the state before the failing step; it may be empty if the
                      run failed before discovery finished.
    terminal_status - completed, budget_exhausted, max_steps, no_progress or
                      crashed.
    context         - the RunContext: output_dir, the tracer, and the settings
                      the report echoes.
    error           - the exception, when terminal_status is crashed.

    In this order (A6):
    1. settle every row still pending, each with its state_change event
    2. write report.json
    3. write the terminal event, always last. If writing report.json fails,
       the run is crashed after all, and the terminal event says so.
    """
    tracer = context.tracer
    rows = dict(state.get("report_rows", {}))
    current = state.get("current_alert_id")

    for alert_id, row in rows.items():
        if row["status"] != "pending":
            continue
        settled = settle_leftover(row, terminal_status, alert_id == current, context)
        rows[alert_id] = settled
        tracer.emit(
            "state_change",
            alert_id=alert_id,
            node="finalize",
            alert_status=settled["status"],
            label=settled["label"],
        )

    try:
        write_report(rows, state.get("used_tokens", 0), terminal_status, context)
    except BaseException:
        terminal_status = "crashed"
        raise
    finally:
        tracer.emit("terminal", status=terminal_status)
