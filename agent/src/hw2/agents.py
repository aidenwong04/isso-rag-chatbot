"""
agents - what differs between the MCP and CodeAct agents, behind one interface.

The graph (nodes.py) is shared: discovery, next_alert, guard, model, the
dispatcher's validation and settlement, report, finalize. Each agent supplies:

  controller   the host's trusted operations: set_alert, current_item,
               reset_packets, classify, write_evidence. MCPController goes
               through the MCP session (the host never opens target files);
               BackendController calls the CodeAct parent's own Backend.
  tool_defs    the tools offered to the model, in the provider's format
  execute()    runs one validated model tool call and returns an Outcome plus
               the passages it delivered to the model

CodeActAgent lives in codeact.py.
"""

import json
from dataclasses import dataclass, field

# Server controls only the host calls. Never offered to the model; the
# dispatcher refuses them by name if the model asks anyway (brief line 183).
HOST_ONLY = frozenset({"set_alert", "reset_packets"})


@dataclass
class Outcome:
    """One operation's result, the same shape for both agents."""
    ok: bool
    payload: dict | None = None      # structured result on success
    error: str | None = None         # the error text the model sees on failure
    error_kind: str | None = None    # "refusal" or "execution"
    refused_by: str | None = None    # "dispatcher", "server", or "kernel" for refusals
    delivered: set[str] = field(default_factory=set)  # chunk ids shown to the model
    trace: dict = field(default_factory=dict)          # extra fields for the tool_result event
    text: str | None = None          # plain-text reply instead of JSON (CodeAct script output)

    def reply(self) -> str:
        """The tool message content the model sees."""
        if self.text is not None:
            return self.text
        if self.ok:
            return json.dumps(self.payload, ensure_ascii=False)
        return self.error


def mcp_error_kind(tool: str, text: str) -> tuple[str, str | None]:
    """
    (error_kind, refused_by) for an MCP error result. mcp 2.3.0 prefixes
    every ToolError with "Error executing tool <name>: ", so the reason is
    read after that prefix. A bare "Error executing tool <name>" is a crash
    inside the tool, whose details stay in the server's log.
    """
    prefix = f"Error executing tool {tool}: "
    if not text.startswith(prefix):
        return "execution", None
    detail = text[len(prefix):]
    if detail.startswith("refused:") or "validation error for" in detail:
        return "refusal", "server"
    return "execution", None


def backend_error_kind(msg: str) -> tuple[str, str | None]:
    """(error_kind, refused_by) for a Backend result run by the CodeAct harness itself."""
    if msg.startswith("refused:"):
        return "refusal", "dispatcher"
    return "execution", None


# --- Controllers ----------------------------------------------------------------------

class MCPController:
    """The host's operations, as calls to the MCP server."""

    def __init__(self, session):
        self.session = session

    async def call(self, name: str, arguments: dict) -> Outcome:
        result = await self.session.call_tool(name, arguments)
        if result.is_error:
            text = result.content[0].text if result.content else f"Error executing tool {name}"
            kind, by = mcp_error_kind(name, text)
            return Outcome(False, error=text, error_kind=kind, refused_by=by)
        payload = result.structured_content
        if payload is None:
            payload = {"text": "\n".join(getattr(c, "text", "") for c in result.content)}
        return Outcome(True, payload=payload)

    async def set_alert(self, alert_id: str, phase: str) -> Outcome:
        return await self.call("set_alert", {"alert_id": alert_id, "phase": phase})

    async def current_item(self) -> Outcome:
        return await self.call("current_item", {})

    async def reset_packets(self, query_set: str) -> Outcome:
        return await self.call("reset_packets", {"query_set": query_set})


class BackendController:
    """The host's operations, as calls to the CodeAct parent's own Backend."""

    def __init__(self, backend):
        self.backend = backend

    @staticmethod
    def wrap(result: dict, shape_keys: tuple[str, ...] | None = None) -> Outcome:
        if not result["ok"]:
            kind, by = backend_error_kind(result["msg"])
            return Outcome(False, error=result["msg"], error_kind=kind, refused_by=by)
        payload = {k: v for k, v in result.items() if k != "ok" and (shape_keys is None or k in shape_keys)}
        return Outcome(True, payload=payload)

    async def set_alert(self, alert_id: str, phase: str) -> Outcome:
        return self.wrap(self.backend.set_alert(alert_id, phase))

    async def current_item(self) -> Outcome:
        return self.wrap(self.backend.current_item())

    async def reset_packets(self, query_set: str) -> Outcome:
        return self.wrap(self.backend.reset_packets(query_set))

    async def classify(self, arguments: dict, delivered: set[str]) -> Outcome:
        """
        HW1's checks, against the harness's delivered set: the passages it saw
        in the scripts' output, never anything the worker reported about
        itself. The parent Backend's observed set is replaced with it first.
        """
        self.backend.observed = set(delivered)
        return self.wrap(self.backend.classify(
            arguments.get("label"), arguments.get("rationale"), arguments.get("evidence")))

    async def write_evidence(self, quote) -> Outcome:
        return self.wrap(self.backend.write_evidence(quote), ("msg", "item", "diff", "checks"))


# --- The MCP agent ----------------------------------------------------------------------

class MCPAgent:
    kind = "mcp"
    via = "mcp"

    def __init__(self, session, server_name: str, listed_tools):
        self.server_name = server_name
        self.controller = MCPController(session)
        offered = [t for t in listed_tools if t.name not in HOST_ONLY]
        self.tool_names = frozenset(t.name for t in offered)
        self.tool_defs = [{"type": "function",
                           "function": {"name": t.name, "description": t.description or "",
                                        "parameters": t.input_schema}} for t in offered]

    def discovery_event(self, tracer) -> None:
        tracer.emit("discovery", server=self.server_name, tools=sorted(self.tool_names))

    def opening_messages(self, item: dict) -> list[dict]:
        from prompt import opening_messages
        return opening_messages(item)

    def evidence_message(self, chunk_id: str) -> dict:
        from prompt import evidence_phase_message
        return evidence_phase_message(chunk_id)

    async def start_item(self, alert_id: str) -> None:
        """Nothing per item: the server's set_alert resets its own item state."""

    async def execute(self, name: str, arguments: dict, state: dict) -> Outcome:
        """Every model tool goes to the server (brief line 148)."""
        outcome = await self.controller.call(name, arguments)
        if outcome.ok and name in ("read", "read_neighbors"):
            outcome.delivered = {chunk["chunk_id"] for chunk in outcome.payload.get("chunks", [])}
        return outcome

    async def close(self) -> None:
        """The CLI's async with blocks close the session and stop the server."""
