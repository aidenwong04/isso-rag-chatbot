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
import json
from pathlib import Path

PHASES = ("classify", "evidence")
QUERY_SETS = ("dev", "heldout-a", "heldout-b")


class Backend:
    """One item's view of the frozen inputs and its question's packet."""

    def __init__(self, prepared_dir: Path, evidence_dir: Path):
        """
        prepared_dir: the folder holding manifest.json, chunks.json,
            neighbors.json, and <set>/{queries,candidates}.json.
        evidence_dir: the writable folder holding <query_id>.json packets.

        The frozen candidate lists and neighbors are loaded once here, so the
        per-item methods only look things up.
        """
        self.prepared_dir = Path(prepared_dir)
        self.evidence_dir = Path(evidence_dir)

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
        self.queries = {}
        for query_set in QUERY_SETS:
            path = self.prepared_dir / query_set / "queries.json"
            with open(path, encoding="utf-8") as f:
                for query in json.load(f):
                    self.queries[query["id"]] = query

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
        self.allowed_chunk_ids = None
        self.packet_path = None
        self.phase = None
        self.observed = set()

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

        return {"set": res, "ok": True, "msg": "success"}

    # --- Host-only control --------------------------------------------------

    def set_alert(self, alert_id: str, phase: str) -> dict:
        """
        Selects the current item and phase (brief line 180). Never offered to
        the model: in MCP the dispatcher rejects it; in CodeAct, decide where
        it lives so generated code cannot call it to switch items.

        Derives the query, target chunk, allowed read set, and packet path from
        the frozen manifest. phase is "classify" or "evidence".
        """
        if phase not in PHASES:
            return self._refuse_item("refused: unknown phase")
        if phase == "evidence" and alert_id != self.alert_id:
            # trying to continue another alert that wasn't just classified
            return self._refuse_item("refused: attempting to enter evidence phase with a new alert.")

        aset = self.compute_allowed_set(alert_id)
        if not aset["ok"]:
            return self._refuse_item(aset["msg"])
        cand = self.by_alert_id[alert_id]

        if phase == "classify":
            self.alert_id = alert_id
            self.query_id = cand["query_id"]
            self.chunk_id = cand["chunk_id"]
            self.question = self.queries[self.query_id]["text"]
            self.audience = self.queries[self.query_id]["audience"]
            self.allowed_chunk_ids = aset["set"]
            self.packet_path = self.evidence_dir / f"{self.query_id}.json"
            self.phase = "classify"
            self.observed = set()
        else:
            self.phase = "evidence"

        return {
            "ok": True,
            "msg": "success",
            "alert_id": self.alert_id,
            "query_id": self.query_id,
            "chunk_id": self.chunk_id,
            "phase": self.phase,
        }

    # --- Model-facing operations ---------------------------------------------

    def current_item(self) -> dict:
        """Returns the question, audience, selected chunk ID, and permitted candidate IDs."""

        raise NotImplementedError

    def read(self, chunk_id: str) -> dict:
        """Returns canonical text and metadata for chunk_id, refusing IDs outside the allowed set."""
        raise NotImplementedError

    def read_neighbors(self, chunk_id: str) -> dict:
        """Returns the frozen immediate neighbors of chunk_id that are in the allowed set."""
        raise NotImplementedError

    def classify(self, label: int, rationale: str, evidence: list[str]) -> dict:
        """
        Checks a proposed classification against the HW1 contract (brief line 200):
        label is the integer 0 or 1, rationale is nonempty and at most 600
        characters, evidence IDs were delivered during this item and include
        the target chunk. Proposes; the host decides whether to accept.
        """
        raise NotImplementedError

    def read_packet(self) -> dict:
        """Returns the current question's packet. Refused during the classify phase."""
        raise NotImplementedError

    def write_evidence(self, quote: str) -> dict:
        """
        Proposes the current chunk's evidence item for its question's packet.
        Refused outside the evidence phase.

        TODO: an item holds chunk_id, quote, source, and url (brief line 219).
        Which of those should the model supply, and which come from the
        trusted current item and canonical metadata?
        """
        raise NotImplementedError
