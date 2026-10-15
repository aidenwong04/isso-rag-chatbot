"""
run - runs the graph once and always finishes the run (A6).

The crash default lives here, outside the graph. Inside the graph,
terminal_status is unset until a node ends the run on purpose (next_alert with
completed, guard with a limit), so a routing function can treat "a status is
set" as "go to END". If the graph raises instead, terminal_status here is
still crashed.

invoke() returns nothing when a node raises, so the run is streamed and the
last full state is kept: that is what finalize settles and writes after a
crash.
"""

from report import finalize
from state import RunContext


# Supersteps per model call: guard, model, dispatcher, tool, and next_alert
# once per alert. Doubled for margin, so that max_steps always ends the run
# first. If LangGraph's own limit is ever hit, that is a bug, and the run ends
# crashed.
SUPERSTEPS_PER_MODEL_CALL = 10


def run(app, context: RunContext):
    terminal_status = "crashed"
    last_state = {}
    error = None
    try:
        config = {"recursion_limit": SUPERSTEPS_PER_MODEL_CALL * context.max_steps + 10}
        for last_state in app.stream({}, context=context, stream_mode="values", config=config):
            pass
        # A graph that reached END without any node setting a status is a bug
        # in the graph, so it is left as crashed.
        terminal_status = last_state.get("terminal_status") or "crashed"
    except BaseException as exc:
        error = exc
        raise
    finally:
        finalize(last_state, terminal_status, error)
    return terminal_status
