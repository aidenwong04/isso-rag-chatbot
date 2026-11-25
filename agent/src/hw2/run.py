"""
run - runs the HW2 graph once and always finishes the run, as src/hw1/run.py.

Async, because the MCP client is: the graph runs with astream inside the
CLI's open MCP session. The crash default lives here, outside the graph; the
last full state is kept so finalize can settle and write it after a crash.
"""

from report import finalize
from state import RunContext

# Supersteps per model call (guard, model, dispatcher, next_alert), doubled
# for margin so max_steps always ends the run before LangGraph's own limit.
SUPERSTEPS_PER_MODEL_CALL = 10


async def run(app, context: RunContext) -> str:
    """
    context must already hold the run's Tracer, MCP session, and model client.
    run() owns closing the tracer.
    """
    terminal_status = "crashed"
    last_state = {}
    error = None
    try:
        config = {"recursion_limit": SUPERSTEPS_PER_MODEL_CALL * context.max_steps + 100}
        async for last_state in app.astream({}, context=context, stream_mode="values", config=config):
            pass
        terminal_status = last_state.get("terminal_status") or "crashed"
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            finalize(last_state, terminal_status, context, error)
        finally:
            context.tracer.close()
    return terminal_status
