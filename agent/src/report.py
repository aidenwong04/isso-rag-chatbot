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


def finalize(state, terminal_status, error=None):
    """
    The end of every run, planned or not (A6). Called once, from the runner's
    finally block, never as a graph node, so it also runs after a crash.

    state           - the last full state the runner saw. After a crash this is
                      the state before the failing step; it may be empty if the
                      run failed before discovery finished.
    terminal_status - completed, budget_exhausted, max_steps, no_progress or
                      crashed.
    error           - the exception, when terminal_status is crashed.

    To do, in this order (A6):
    1. settle every row still pending, by terminal_status (the A6 table:
       the alert in progress and the unstarted alerts can differ)
    2. write the terminal event to the trace
    3. write report.json, rows sorted by test_id, path, then line
    """
    for alert_id, row in state["report_rows"].items():
        if row["status"] == "pending":
            row["status"] = terminal_status

    # todos 2 and 3...
    raise NotImplementedError
