"""
api - the backend both HW2 agents use (brief section 3.1).

CodeAct: the model's generated Python imports this and calls it directly.
MCP: the server's tools call it, and the model only sees the tools.

Reads the frozen inputs written by prepare.py (agent/prepared/) and the
per-question evidence packets. Data roots are constructor arguments, so the
same code runs locally and inside the sandbox (/target/..., /sandbox/work/...).

Model-facing operations (brief section 3.3):
  current_item     the question, audience, selected chunk ID, permitted candidate IDs
  read             canonical text and metadata for an allowed chunk ID
  read_neighbors   the permitted immediate neighbors of a chunk
  classify         propose the label, rationale, and observed evidence IDs
  read_packet      the current question's packet (evidence phase only)
  write_evidence   propose the current chunk's evidence item (evidence phase only)

Results follow the HW1 tools' shape: a dict with "ok" and "msg", plus data.

Open design questions (decide before filling these in):
  - What is the allowed read set for an item, and what changes from HW1's
    read and read_neighbors, which could reach any chunk in the corpus?
  - HW1's read_neighbors told the model to "call it again on a neighbor's
    id". What does brief line 263 say about that now?
  - Line 132: item selection, budgets, and acceptance state must stay outside
    the generated code's namespace. In CodeAct this object lives in that
    namespace. Which of its state is authoritative, and which is only a
    convenience the host re-checks?
"""
import difflib
import json
import os
import sys
import tempfile
from pathlib import Path

# The HW1 tools (validate_id, generate_chunk, classify) are reused as is. They
# use flat imports (`from workspace import ...`), so their folder goes on the
# import path; appended, so it never shadows a module of the same name here.
HW1_DIR = Path(__file__).resolve().parents[1] / "hw1"
if str(HW1_DIR) not in sys.path:
    sys.path.append(str(HW1_DIR))
import tools as hw1_tools  # noqa: E402

from prepare import verify_prepared  # the manifest check prepare.py writes and runs

PHASES = ("classify", "evidence")


def serialize_packet(packet: dict) -> str:
    """
    A packet's file text: UTF-8 JSON, indent=2, one trailing newline, the
    format of examples/evidence.example.json. The harness creates packets and
    write_evidence writes them in this same form, so diffs show only real
    changes, never formatting noise.
    """
    return json.dumps(packet, ensure_ascii=False, indent=2) + "\n"

QUERY_SETS = ("dev", "heldout-a", "heldout-b")

# Defaults for Backend() with no arguments, set by the CodeAct harness in the
# worker's environment, so generated code can write Backend() (brief line 126).
PREPARED_DIR_ENV = "ISSO_PREPARED_DIR"
EVIDENCE_DIR_ENV = "ISSO_EVIDENCE_DIR"

# How show() prints a passage, and how the CodeAct harness finds passages in
# a script's output (brief line 137). The text between the two lines is the
# passage's canonical text exactly, so the harness can check it byte for byte.
PASSAGE_START = "<<<chunk {chunk_id}>>>"
PASSAGE_END = "<<<end {chunk_id}>>>"


def passage_block(chunk_id: str, text: str) -> str:
    """One passage as show() prints it: start line, exact text, end line."""
    return f"{PASSAGE_START.format(chunk_id=chunk_id)}\n{text}\n{PASSAGE_END.format(chunk_id=chunk_id)}\n"


class Backend:
    """One item's view of the frozen inputs and its question's packet."""

    def __init__(self, prepared_dir: Path | None = None, evidence_dir: Path | None = None):
        """
        prepared_dir: the folder holding manifest.json, chunks.json,
            neighbors.json, and <set>/{queries,candidates}.json.
        evidence_dir: the writable folder holding <query_id>.json packets.

        The frozen candidate lists and neighbors are loaded once here, so the
        per-item methods only look things up.
        """
        # With no arguments (generated CodeAct code), the folders come from
        # the environment the harness sets for the worker.
        prepared_dir = prepared_dir or os.environ.get(PREPARED_DIR_ENV)
        evidence_dir = evidence_dir or os.environ.get(EVIDENCE_DIR_ENV)
        if not prepared_dir or not evidence_dir:
            raise ValueError(f"Backend needs prepared_dir and evidence_dir, or the "
                             f"{PREPARED_DIR_ENV} and {EVIDENCE_DIR_ENV} environment variables")
        self.prepared_dir = Path(prepared_dir)
        self.evidence_dir = Path(evidence_dir)

        # Check the prepared inputs against the frozen manifest before using
        # them (brief line 86). On failure nothing is loaded and every
        # operation refuses with this message, so a changed input never
        # reaches the model. Lazy in the MCP server: this runs on the first
        # tool call, not at startup.
        self.verification_error = None
        try:
            verify_prepared(self.prepared_dir / "manifest.json")
        except (OSError, ValueError) as e:
            self.verification_error = f"error: prepared inputs failed verification: {e}"
            self.candidates, self.by_alert_id, self.neighbors = [], {}, {}
            self.queries, self.query_ids_by_set = {}, {}
            self._clear_item()
            return

        # All three sets' candidates, each set in its frozen row order.
        self.candidates = []
        for query_set in QUERY_SETS:
            path = self.prepared_dir / query_set / "candidates.json"
            with open(path, encoding="utf-8") as f:
                self.candidates.extend(json.load(f))
        self.by_alert_id = {c["alert_id"]: c for c in self.candidates}

        with open(self.prepared_dir / "neighbors.json", encoding="utf-8") as f:
            self.neighbors = {record["chunk_id"]: record["neighbors"] for record in json.load(f)}

        # query_id -> {"id", "text", "audience"}, from all three sets' frozen query files.
        # query_ids_by_set: set name -> its query ids in file order (for reset_packets).
        self.queries = {}
        self.query_ids_by_set = {}
        for query_set in QUERY_SETS:
            path = self.prepared_dir / query_set / "queries.json"
            with open(path, encoding="utf-8") as f:
                for query in json.load(f):
                    self.queries[query["id"]] = query
                    self.query_ids_by_set.setdefault(query_set, []).append(query["id"])

        # The current item, set only by set_alert. Empty until the first
        # successful set_alert(..., "classify"), so every field always exists.
        self._clear_item()

    def _clear_item(self) -> None:
        """
        Forgets the current item. Called at startup and on every refused
        set_alert, so a refusal never leaves the previous item active: with no
        item set, the model-facing methods must refuse until a valid set_alert.
        """
        self.alert_id = None
        self.query_id = None
        self.chunk_id = None
        self.question = None
        self.audience = None
        self.candidate_ids = None
        self.allowed_chunk_ids = None
        self.packet_path = None
        self.phase = None
        self.observed = set()
        self.pre_stage_packet = None
        self.evidence_written = False

    def _refuse_item(self, msg: str) -> dict:
        """A refused set_alert: clears the current item, then returns the refusal."""
        self._clear_item()
        return {"ok": False, "msg": msg}

    def compute_allowed_set(self, alert_id: str) -> dict:
        """
        computes the allowed set of chunk_ids from the current alert_id:
        the question's three frozen candidates plus their frozen immediate neighbors.
        """
        # Look the alert up in the frozen candidate lists instead of parsing
        # its ID: any input that is not exactly a frozen alert_id (a typo, an
        # unknown set, a non-string, "garbage") is simply not found.
        if not isinstance(alert_id, str) or alert_id not in self.by_alert_id:
            return {"set": None, "ok": False, "msg": "refused: unknown alert_id"}

        # query_id comes from the frozen row, never from splitting the input.
        query_id = self.by_alert_id[alert_id]["query_id"]

        res = set()
        siblings = []

        for c in self.candidates:
            if c["query_id"] == query_id:
                res.add(c["chunk_id"])
                siblings.append(c["chunk_id"])
        # now res is the three sibling candidates
        # next, find their neighbors and add to res.

        for c in siblings:
            res.update(self.neighbors[c])

        # Returns the siblings too, so set_alert can store them; this method
        # only computes and never changes the current item.
        return {"set": res, "candidate_ids": siblings, "ok": True, "msg": "success"}

    # --- Host-only control --------------------------------------------------

    def set_alert(self, alert_id: str, phase: str) -> dict:
        """
        Selects the current item and phase (brief line 180). Never offered to
        the model: in MCP the dispatcher rejects it; in CodeAct, decide where
        it lives so generated code cannot call it to switch items.

        Derives the query, target chunk, allowed read set, and packet path from
        the frozen manifest. phase is "classify" or "evidence".
        """
        if self.verification_error:
            return self._refuse_item(self.verification_error)
        if phase not in PHASES:
            return self._refuse_item("refused: unknown phase")
        if phase == "evidence" and self.alert_id is not None and alert_id != self.alert_id:
            # trying to jump from the current item to a different one
            return self._refuse_item("refused: attempting to enter evidence phase with a new alert.")

        aset = self.compute_allowed_set(alert_id)
        if not aset["ok"]:
            return self._refuse_item(aset["msg"])
        cand = self.by_alert_id[alert_id]

        # A new item, or the evidence phase on a fresh Backend (each CodeAct
        # script starts one): load the item's context first. Whether evidence
        # is allowed at all is the host's decision (brief line 182), not this
        # object's.
        if phase == "classify" or self.alert_id is None:
            self.alert_id = alert_id
            self.query_id = cand["query_id"]
            self.chunk_id = cand["chunk_id"]
            self.question = self.queries[self.query_id]["text"]
            self.audience = self.queries[self.query_id]["audience"]
            self.allowed_chunk_ids = aset["set"]
            self.candidate_ids = aset["candidate_ids"]
            self.packet_path = self.evidence_dir / f"{self.query_id}.json"
            self.phase = "classify"
            self.observed = set()
            self.pre_stage_packet = None
            self.evidence_written = False
        if phase == "evidence":
            self.phase = "evidence"
            # The packet as it stands when this item's evidence stage begins.
            # The report's diff is against this (report.schema.json), and the
            # new packet is built from it, not from whatever is on disk later.
            # None if the packet cannot be read; propose_evidence then fails
            # its applies check.
            snapshot = self.read_packet()
            self.pre_stage_packet = snapshot["packet"] if snapshot["ok"] else None

        return {
            "ok": True,
            "msg": "success",
            "alert_id": self.alert_id,
            "query_id": self.query_id,
            "chunk_id": self.chunk_id,
            "phase": self.phase,
        }

    def reset_packets(self, query_set: str) -> dict:
        """
        Host-only: writes an empty packet {"query_id": q, "items": []} for every
        question in query_set, replacing whatever was there (brief lines 215,
        518: every launch starts from empty packets). Called once before a
        run's first item: by the MCP host through a host-only server tool, by
        the CodeAct parent on its own Backend. Clears the current item, so no
        stage survives a reset.

        Destructive, so never offered to the model, like set_alert.
        """
        if self.verification_error:
            return {"ok": False, "msg": self.verification_error}
        if query_set not in self.query_ids_by_set:
            return {"ok": False, "msg": f"refused: unknown query set {query_set!r}"}

        self._clear_item()
        query_ids = self.query_ids_by_set[query_set]
        try:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            for query_id in query_ids:
                self._write_atomically(
                    self.evidence_dir / f"{query_id}.json",
                    serialize_packet({"query_id": query_id, "items": []}),
                )
        except OSError as e:
            return {"ok": False, "msg": f"error: packets for {query_set} could not be reset ({e.strerror})"}

        return {"ok": True, "msg": "success", "query_set": query_set, "query_ids": list(query_ids)}

    # --- Model-facing operations ---------------------------------------------

    def current_item(self) -> dict:
        """Returns the question, audience, selected chunk ID, and permitted candidate IDs."""
        refusal = self._no_item_refusal()
        if refusal:
            return refusal
        return {
            "ok": True,
            "question": self.question,
            "audience": self.audience,
            "chunk_id": self.chunk_id,
            "candidate_ids": self.candidate_ids,
            "msg": "success"
        }


    def _no_item_refusal(self) -> dict | None:
        """
        A refusal when the inputs failed verification, or no item is set
        (startup, or after a refused set_alert); else None.
        """
        if self.verification_error:
            return {"ok": False, "msg": self.verification_error, "chunks": []}
        if self.alert_id is None:
            return {"ok": False, "msg": "refused: no current item", "chunks": []}
        return None

    def _scope_refusal(self, chunk_id) -> dict | None:
        """
        A refusal when chunk_id is malformed or outside this item's allowed set;
        else None.

        The allowed-set check comes before any lookup in the corpus, so an
        out-of-scope ID is refused the same way whether or not that chunk
        exists: the model learns nothing about chunks it may not read.
        """
        if not isinstance(chunk_id, str):
            return {"ok": False, "msg": "refused: malformed chunk id", "chunks": []}
        if chunk_id not in self.allowed_chunk_ids:
            return {"ok": False, "msg": "refused: chunk_id outside allowed set.", "chunks": []}
        return None

    def read(self, chunk_id: str) -> dict:
        """Returns canonical text and metadata for chunk_id, refusing IDs outside the allowed set."""
        refusal = self._no_item_refusal() or self._scope_refusal(chunk_id)
        if refusal:
            return refusal

        # HW1's validate_id reads the canonical view (prepared_dir/chunks.json).
        # Every allowed ID is in the view, so this only fails if the frozen
        # inputs are broken; refuse rather than crash.
        validation = hw1_tools.validate_id(chunk_id, self.prepared_dir)
        if not validation["ok"]:
            return {"ok": False, "msg": "refused: " + validation["msg"], "chunks": []}

        self.observed.add(chunk_id)
        return {
            "ok": True,
            "msg": "success",
            "chunks": [hw1_tools.generate_chunk(validation["chunk"])],
        }

    def read_neighbors(self, chunk_id: str) -> dict:
        """
        Returns the frozen immediate neighbors of chunk_id that are in the allowed set.

        Unlike HW1, neighbors come from the frozen neighbors.json, not from
        index arithmetic, and the call never widens the allowed set: chunk_id
        itself must be allowed, and neighbors outside the set are dropped with
        a note saying so (brief line 263).
        """
        refusal = self._no_item_refusal() or self._scope_refusal(chunk_id)
        if refusal:
            return refusal

        allowed, omitted = [], 0
        for neighbor_id in self.neighbors.get(chunk_id, []):
            if neighbor_id in self.allowed_chunk_ids:
                allowed.append(neighbor_id)
            else:
                omitted += 1

        chunks = []
        for neighbor_id in allowed:
            validation = hw1_tools.validate_id(neighbor_id, self.prepared_dir)
            if not validation["ok"]:
                return {"ok": False, "msg": "refused: " + validation["msg"], "chunks": []}
            chunks.append(hw1_tools.generate_chunk(validation["chunk"]))
            self.observed.add(neighbor_id)

        msg = "success"
        if omitted:
            msg += f"; {omitted} neighbor(s) outside this item's allowed set were omitted"
        return {"ok": True, "msg": msg, "chunks": chunks}

    def show(self, chunk_id: str) -> dict:
        """
        read(chunk_id), and on success prints the passage in the marker format
        (passage_block). For CodeAct: the harness counts a passage as delivered
        to the model only when it finds this block in the script's output with
        the exact canonical text (brief line 137). Returns read's result.
        """
        result = self.read(chunk_id)
        if result["ok"]:
            for chunk in result["chunks"]:
                sys.stdout.write(passage_block(chunk["chunk_id"], chunk["text"]))
        return result

    def show_neighbors(self, chunk_id: str) -> dict:
        """read_neighbors(chunk_id), printing each returned passage like show()."""
        result = self.read_neighbors(chunk_id)
        if result["ok"]:
            for chunk in result["chunks"]:
                sys.stdout.write(passage_block(chunk["chunk_id"], chunk["text"]))
        return result

    def classify(self, label: int, rationale: str, evidence: list[str]) -> dict:
        """
        Checks a proposed classification against the HW1 contract (brief line 200):
        label is the integer 0 or 1, rationale is nonempty and at most 600
        characters, evidence IDs were delivered during this item and include
        the target chunk. Proposes; the host decides whether to accept.

        A pure check: it changes no state, so calling it from generated code
        settles nothing. The checks themselves are HW1's classify, run against
        this item's observed set and target chunk.
        """
        refusal = self._no_item_refusal()
        if refusal:
            return {"ok": False, "msg": refusal["msg"]}
        if self.phase != "classify":
            return {"ok": False, "msg": "refused: classify is only allowed in the classify phase"}

        result = hw1_tools.classify(
            label, rationale, evidence, self.observed, self.chunk_id, self.prepared_dir
        )
        if not result["ok"]:
            return {"ok": False, "msg": "refused: " + result["msg"]}
        return result

    def read_packet(self) -> dict:
        """Returns the current question's packet. Refused during the classify phase."""
        refusal = self._no_item_refusal()
        if refusal:
            return {"ok": False, "msg": refusal["msg"]}
        if self.phase == "classify":
            return {"ok": False, "msg": "refused: read_packet is only allowed in the evidence phase."}

        # A missing, unparsable, or misshapen packet means the harness setup
        # failed or the file was tampered with. That is an execution error, not
        # a refusal of the model's request, so these messages say "error:".
        # They leave out the file path: the model never deals with packet
        # destinations (brief line 225).
        try:
            with open(self.packet_path, "r", encoding="utf-8") as f:
                packet = json.load(f)
        except FileNotFoundError:
            return {"ok": False, "msg": f"error: the packet for {self.query_id} is missing"}
        except json.JSONDecodeError as e:
            return {
                "ok": False,
                "msg": f"error: the packet for {self.query_id} is not valid JSON "
                       f"(line {e.lineno}, column {e.colno}: {e.msg})",
            }

        if (
            not isinstance(packet, dict)
            or packet.get("query_id") != self.query_id
            or not isinstance(packet.get("items"), list)
        ):
            return {"ok": False, "msg": f"error: the packet for {self.query_id} is malformed"}

        # Under its own key, so file contents can never pose as this method's
        # own ok/msg fields.
        return {"ok": True, "msg": "success", "packet": packet}

    @staticmethod
    def _failed_proposal(msg: str, *, applies: bool, current_item_only: bool, grounded: bool) -> dict:
        """
        A proposal that produced no update. Always carries the three checks and
        an empty diff, because the report needs both for rejected updates too
        (report.schema.json, evidenceUpdate). A check that could not be
        evaluated because an earlier step failed is reported as False.
        """
        return {
            "ok": False,
            "msg": msg,
            "diff": "",
            "checks": {"applies": applies, "current_item_only": current_item_only, "grounded": grounded},
        }

    def propose_evidence(self, quote:str):

        # No item or wrong phase: nothing can be proposed, so no check passes.
        refusal = self._no_item_refusal()
        if refusal:
            return self._failed_proposal(
                refusal["msg"], applies=False, current_item_only=False, grounded=False
            )
        if self.phase == "classify":
            return self._failed_proposal(
                "refused: write_evidence is only allowed in the evidence phase.",
                applies=False, current_item_only=False, grounded=False,
            )

        # Quote checks (brief line 220): a nonempty string of at most 600
        # characters that is a contiguous substring of the target passage.
        # Each failure is a grounding failure; no packet is built, so applies
        # is False. The item would be for the current chunk (the model cannot
        # choose another), so current_item_only is not what failed.
        def quote_refusal(msg: str) -> dict:
            return self._failed_proposal(msg, applies=False, current_item_only=True, grounded=False)

        if not isinstance(quote, str):
            return quote_refusal("refused: quote must be a string")
        if quote.strip() == "":
            return quote_refusal("refused: quote is empty or only whitespace")
        if len(quote) > 600:
            return quote_refusal(f"refused: quote is {len(quote)} characters; the limit is 600")

        # The full canonical record (with source and url), straight from the
        # view. Not self.read: that would add to the observed set, and this
        # method changes no state.
        validation = hw1_tools.validate_id(self.chunk_id, self.prepared_dir)
        if not validation["ok"]:
            return self._failed_proposal(
                "error: " + validation["msg"], applies=False, current_item_only=False, grounded=False
            )
        this_chunk = validation["chunk"]

        if quote not in this_chunk["text"]:
            return quote_refusal(
                "refused: quote is not an exact substring of the target passage; "
                "copy it character for character, including spacing and punctuation"
            )

        curr_item = {"chunk_id":self.chunk_id,
                     "quote":quote,
                     "source":this_chunk["source"],
                     "url":this_chunk["url"]}

        # Build the new packet from the pre-stage snapshot.
        pre = self.pre_stage_packet
        if pre is None:
            return {
                "ok": False,
                "msg": f"error: the packet for {self.query_id} could not be read when the evidence stage began",
                "checks": {"applies": False, "current_item_only": False, "grounded": True},
                "diff": "",
            }

        if any(item.get("chunk_id") == self.chunk_id for item in pre["items"]):
            return {
                "ok": False,
                "msg": f"refused: the packet already has an item for {self.chunk_id}",
                "checks": {"applies": True, "current_item_only": False, "grounded": True},
                "diff": "",
            }

        # Earlier entries are copied unchanged; the only change is one new
        # item at the end.
        new_packet = {"query_id": pre["query_id"], "items": list(pre["items"]) + [curr_item]}

        # Unified diff of the two file texts. Labels follow the example
        # report: the packet path without its leading slash, under a/ and b/.
        label = self.packet_path.as_posix().lstrip("/")
        diff = "".join(difflib.unified_diff(
            serialize_packet(pre).splitlines(keepends=True),
            serialize_packet(new_packet).splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
        ))

        return {
            "ok": True,
            "msg": "success",
            "item": curr_item,
            "packet": new_packet,
            "diff": diff,
            "checks": {"applies": bool(diff), "current_item_only": True, "grounded": True},
        }




    def write_evidence(self, quote: str) -> dict:
        """
        Writes the current chunk's evidence item to its question's packet, if
        propose_evidence accepts it. The model supplies only the quote;
        chunk_id, source, and url come from the trusted current item and the
        canonical view.

        Callers: the MCP server's write_evidence tool, and the CodeAct
        harness's own (parent-side) Backend. A call from generated code inside
        the CodeAct worker also writes, but the harness then finds a change it
        did not make, rejects it, and restores the packet.

        At most one item is written per stage: after one succeeds, further
        calls are refused until the next item's classify stage.
        """
        if self.alert_id is not None and self.evidence_written:
            return self._failed_proposal(
                f"refused: evidence was already written for {self.chunk_id} in this stage",
                applies=False, current_item_only=False, grounded=False,
            )

        # Phase, item, quote, duplicate, and diff checks all live here.
        result = self.propose_evidence(quote)
        if not result["ok"]:
            return result

        try:
            self._write_atomically(self.packet_path, serialize_packet(result["packet"]))
        except OSError as e:
            return self._failed_proposal(
                f"error: the packet for {self.query_id} could not be written ({e.strerror})",
                applies=False, current_item_only=True, grounded=True,
            )

        self.evidence_written = True
        return result

    @staticmethod
    def _write_atomically(path: Path, text: str) -> None:
        """
        Writes text to path so the file is always either the old or the new
        version: write a temporary file in the same folder, then os.replace it
        onto path (a single rename). A crash mid-write leaves the old packet.
        """
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            # Leave no temporary file behind if anything failed before the rename.
            Path(tmp_name).unlink(missing_ok=True)
            raise
