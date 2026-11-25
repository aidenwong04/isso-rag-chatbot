"""
mcp_server - the stdio MCP server for the HW2 MCP agent (brief section 3.2).

A thin layer over Backend (api.py): each tool calls one Backend method and
turns its {"ok", "msg", ...} result into an MCP result. All behavior lives in
the Backend; this file is protocol only.

Adapted from the staff reference pattern, comse6998-019/hw2-tool-boundary
reference/cldk_mcp_server.py (MCPServer, @mcp.tool, ToolError).

Started by the host from the repository-root mcp.json, for example:
    env/bin/python agent/src/hw2/mcp_server.py \
        --prepared-dir agent/prepared --evidence-dir /sandbox/work/evidence

Transport rules (brief line 155):
  - stdout carries protocol JSON only. Never print() here: a stray line on
    stdout breaks the transport. Diagnostics go to stderr through `log`.
  - Answer initialize promptly (line 154): the Backend is created lazily, on
    the first tool call, not at startup.

Model-facing tools: current_item, read, read_neighbors, classify,
read_packet, write_evidence. Host-only: set_alert, reset_packets. The host
filters them out of what it offers the model, and its dispatcher rejects a
model request naming either (line 183); the server itself still serves them,
because the host and staff call them directly.

The Backend verifies the prepared inputs against manifest.json when it is
created, on the first tool call; if that fails, every tool refuses.

Open design questions:
  - Which parts of a Backend result does the model see as tool content?
  - ok: False becomes a ToolError. How does the trace tell a refusal
    ("refused: ...") from an execution error ("error: ...")?
  - mcp.example.json passes separate corpus, queries, and manifests roots;
    Backend takes one prepared_dir. Which one changes for the sandbox layout?
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import TypedDict

from pydantic import StrictInt

from mcp.server import MCPServer                       # the SDK's main server class
from mcp.server.mcpserver.exceptions import ToolError  # raised when a tool refuses

from api import Backend

log = logging.getLogger("isso-mcp")

mcp = MCPServer("isso")

# Set from the command line in main(); the Backend itself is created on first use.
_args: argparse.Namespace | None = None
_backend: Backend | None = None

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPARED_DIR = REPO_ROOT / "agent" / "prepared"

def backend() -> Backend:
    """The server's one Backend, created on the first tool call (not at startup)."""
    global _backend, _args
    if not _backend:
        _backend = Backend(prepared_dir=_args.prepared_dir, evidence_dir=_args.evidence_dir)

    return _backend


# --- Result shapes -----------------------------------------------------------------
# One TypedDict per tool result. The SDK turns a tool's return annotation into
# that tool's output schema in tools/list (brief line 175: result schemas for
# every tool) and checks each return value against it.

class Chunk(TypedDict):
    """One canonical passage, as HW1's generate_chunk builds it."""
    chunk_id: str
    text: str
    heading: str
    breadcrumb: list[str]
    audience: str


class ChunksResult(TypedDict):
    """read and read_neighbors. msg notes omitted out-of-scope neighbors."""
    msg: str
    chunks: list[Chunk]


class CurrentItemResult(TypedDict):
    question: str
    audience: str
    chunk_id: str
    candidate_ids: list[str]


class ClassifyResult(TypedDict):
    """A passed check only: the host decides whether to accept the label."""
    msg: str


class PacketItem(TypedDict):
    chunk_id: str
    quote: str
    source: str
    url: str


class Packet(TypedDict):
    query_id: str
    items: list[PacketItem]


class PacketResult(TypedDict):
    packet: Packet


class Checks(TypedDict):
    applies: bool
    current_item_only: bool
    grounded: bool


class EvidenceResult(TypedDict):
    """write_evidence. The host records diff and checks in the report."""
    msg: str
    item: PacketItem
    diff: str
    checks: Checks


class ResetResult(TypedDict):
    """reset_packets, host-only."""
    msg: str
    query_set: str
    query_ids: list[str]


class AlertResult(TypedDict):
    """set_alert, host-only."""
    msg: str
    alert_id: str
    query_id: str
    chunk_id: str
    phase: str


def unwrap(result: dict, shape: type):
    """
    Turns a Backend result into a tool return value, or raises ToolError.

    Failure (ok is False): raises ToolError with the Backend's message, so its
    "refused:" or "error:" prefix reaches the host after the SDK's own
    "Error executing tool <name>: " prefix.

    Success: returns exactly the keys `shape` (a TypedDict) declares. `ok` and
    anything else the Backend returned are dropped, so the model sees only
    the declared result, and a key the Backend forgot fails loudly here.
    """
    if not result["ok"]:
        raise ToolError(result["msg"])

    missing = [key for key in shape.__annotations__ if key not in result]
    if missing:
        # A Backend/shape mismatch is a server bug, not the model's mistake:
        # a plain exception becomes the SDK's generic crash result and is
        # logged with its traceback on stderr.
        raise RuntimeError(f"{shape.__name__} result is missing {missing}")
    return {key: result[key] for key in shape.__annotations__}


# --- Host-only control ---------------------------------------------------------

@mcp.tool(description=(
    "HOST-ONLY control, never offered to the model. Selects the current item (an alert_id from the frozen candidate lists) and phase ('classify' or 'evidence'). Every other tool acts on the item set here. A refused call clears the current item."
))
def set_alert(alert_id: str, phase: str) -> AlertResult:
    """Host-only: selects the current item and phase. Never offered to the model."""
    b = backend()
    result = b.set_alert(alert_id, phase)
    return unwrap(result, AlertResult)


@mcp.tool(description=(
    "HOST-ONLY control, never offered to the model. Replaces every packet of a "
    "query set ('dev', 'heldout-a', or 'heldout-b') with an empty one. Called "
    "once before a run's first item. Clears the current item."
))
def reset_packets(query_set: str) -> ResetResult:
    """Host-only: empties the query set's packets before a run."""
    b = backend()
    result = b.reset_packets(query_set)
    return unwrap(result, ResetResult)


# --- Model-facing tools ----------------------------------------------------------

@mcp.tool(description=(
    "Show the passage you are judging: the question, its audience, the target chunk id, and the ids of the question's candidate passages."
))
def current_item() -> CurrentItemResult:
    """The question, audience, target chunk ID, and the question's candidate IDs."""
    b = backend()
    result = b.current_item()
    return unwrap(result, CurrentItemResult)


@mcp.tool(description=(
    "Show one chunk: its text, heading, breadcrumb, and audience. Only the question's candidate chunks and their immediate neighbors can be read; any other id is refused."
))
def read(chunk_id: str) -> ChunksResult:
    """Canonical text and metadata for an allowed chunk."""
    b = backend()
    result = b.read(chunk_id)
    return unwrap(result, ChunksResult)


@mcp.tool(description=(
    'Show the chunks just before and after a chunk on the same page, for context that was cut off. Only neighbors you are allowed to read are returned, and the result says if any were left out. Neighbors of a neighbor cannot be read.'
))
def read_neighbors(chunk_id: str) -> ChunksResult:
    """The allowed frozen neighbors of an allowed chunk."""
    b = backend()
    result = b.read_neighbors(chunk_id)
    return unwrap(result, ChunksResult)


@mcp.tool(description=(
    'Hand in your judgment for the passage you are working on: label 1 if the target passage itself answers all or part of the question for its audience, 0 if not. The rationale is at most 600 characters. Evidence lists chunk ids you have read for this passage and must include the target. Only allowed in the classify phase.'
))
# StrictInt: the SDK validates arguments with pydantic, which would otherwise
# turn "1", 1.0, and True into 1 before the Backend sees them. The HW1
# contract (brief line 200) accepts only the integer 0 or 1.
def classify(label: StrictInt, rationale: str, evidence: list[str]) -> ClassifyResult:
    """Checks a proposed label; the host decides whether to accept it."""
    b = backend()
    result = b.classify(label, rationale, evidence)
    return unwrap(result, ClassifyResult)


@mcp.tool(description=(
    'Show the evidence already collected for this question. Only allowed in the evidence phase.'
))
def read_packet() -> PacketResult:
    """The current question's evidence packet (evidence phase only)."""
    b = backend()
    result = b.read_packet()
    return unwrap(result, PacketResult)


@mcp.tool(description=(
    "Add one quote from the target passage to this question's evidence. The quote must be copied exactly, character for character, from the target passage's text, and be at most 600 characters. Source and URL are filled in for you. Only allowed in the evidence phase, once per passage."
))
def write_evidence(quote: str) -> EvidenceResult:
    """Writes the current chunk's evidence item if every check passes (evidence phase only)."""
    b = backend()
    result = b.write_evidence(quote)
    return unwrap(result, EvidenceResult)


# --- Entry point -----------------------------------------------------------------

def main() -> None:
    global _args
    parser = argparse.ArgumentParser(description="ISSO HW2 MCP server (stdio)")
    parser.add_argument("--prepared-dir", type=Path, required=True,
                        help="folder written by prepare.py: chunks.json, neighbors.json, <set>/...")
    parser.add_argument("--evidence-dir", type=Path, required=True,
                        help="writable folder holding <query_id>.json packets")
    _args = parser.parse_args()

    # stderr only: stdout belongs to the protocol.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log.info("starting: prepared=%s evidence=%s", _args.prepared_dir, _args.evidence_dir)

    mcp.run()


if __name__ == "__main__":
    main()
