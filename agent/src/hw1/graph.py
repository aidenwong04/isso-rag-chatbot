"""
graph - wires the nodes into the state machine (A2/A3 in DESIGN.md).

Fixed edges: START -> discovery -> next_alert, and model -> dispatcher.
Conditional edges, all decided by the runtime: next_alert, guard and
dispatcher each go through their router. Which tool runs inside the dispatcher
is the model's choice, traced as routing by model.

finalize is not a node: run.py calls it in its finally block, so planned
endings route to END and crashes reach it the same way.
"""

from langgraph.graph import START, StateGraph

from nodes import (
    discovery_node,
    dispatcher,
    dispatcher_router,
    guard,
    guard_router,
    model_node,
    next_alert,
    next_alert_router,
)
from state import AgentState, RunContext


def build_graph():
    graph = StateGraph(AgentState, context_schema=RunContext)

    graph.add_node("discovery", discovery_node)
    graph.add_node("next_alert", next_alert)
    graph.add_node("guard", guard)
    graph.add_node("model", model_node)
    graph.add_node("dispatcher", dispatcher)

    graph.add_edge(START, "discovery")
    graph.add_edge("discovery", "next_alert")
    graph.add_conditional_edges("next_alert", next_alert_router)
    graph.add_conditional_edges("guard", guard_router)
    graph.add_edge("model", "dispatcher")
    graph.add_conditional_edges("dispatcher", dispatcher_router)

    return graph.compile()
