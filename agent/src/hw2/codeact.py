"""
codeact - the CodeAct agent for the shared HW2 host (brief section 3.1).

The model has three actions:
  run_python(code)                    its own Python, run in a fresh worker process
  classify(label, rationale, evidence)  handled by the harness, never run as code
  write_evidence(quote)               handled by the harness, never run as code

Generated code imports the backend (from api import Backend; Backend()), calls
set_alert itself, and composes operations; show() prints passages in the
marker format. The worker's Backend is a convenience with no authority: the
code can also open files directly (the kernel's fixed policy is the only limit
there, brief line 284). Authority stays in this parent process:

  - its own Backend (BackendController) does set_alert, reset_packets,
    classify, and write_evidence for the harness;
  - a passage counts as delivered to the model only when the script's output,
    as the model sees it after truncation, holds its marker block with the
    exact canonical text, and the chunk is in the item's allowed set (line 137);
  - classify proposals are checked against that delivered set, never against
    anything the worker reports about itself;
  - after every script, the packet files are compared with the harness's
    trusted copy, and any change the harness did not make is reverted and
    recorded (line 280: detect and restore, which is not prevention);
  - each item gets a fresh scratch folder (cwd, HOME, TMPDIR), deleted when
    the next item starts.
"""

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

from agents import BackendController, Outcome
from api import EVIDENCE_DIR_ENV, PASSAGE_END, PASSAGE_START, PREPARED_DIR_ENV

import tools as hw1_tools  # noqa: E402  (HW1's folder is on the import path once api is imported)

HW2_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_CAP = 20_000     # characters of script output shown to the model
DEFAULT_TIMEOUT = 60            # seconds per script

# A marker block: start line, the passage text, end line for the same chunk.
_PLACEHOLDER = re.escape("{chunk_id}")
BLOCK = re.compile(
    re.escape(PASSAGE_START).replace(_PLACEHOLDER, r"(?P<id>[^>\n]+)")
    + r"\n(?P<text>.*?)\n"
    + re.escape(PASSAGE_END).replace(_PLACEHOLDER, r"(?P=id)"),
    re.DOTALL,
)

TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Run a Python script in a fresh process and get back its exit code, stdout, and "
                        "stderr. Nothing carries over between scripts. Use the backend API described in "
                        "the system prompt; print passages with show() or show_neighbors()."),
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                       "required": ["code"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "classify",
        "description": ("Hand in your judgment for the passage you are working on: label 1 if the target "
                        "passage itself answers all or part of the question for its audience, 0 if not. "
                        "Rationale at most 600 characters. Evidence lists chunk ids that your scripts "
                        "printed with show() or show_neighbors() for this passage, including the target. "
                        "Only allowed in the classify phase."),
        "parameters": {"type": "object", "properties": {
            "label": {"type": "integer", "enum": [0, 1]},
            "rationale": {"type": "string", "maxLength": 600},
            "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
            "required": ["label", "rationale", "evidence"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "write_evidence",
        "description": ("Add one quote from the target passage to this question's evidence. The quote must "
                        "be copied exactly, character for character, from the passage's text, at most 600 "
                        "characters. Source and URL are filled in for you. Only allowed in the evidence "
                        "phase, once per passage."),
        "parameters": {"type": "object", "properties": {"quote": {"type": "string"}},
                       "required": ["quote"], "additionalProperties": False}}},
]

EVIDENCE_PHASE_MESSAGE = """\
Your label 1 for {chunk_id} was accepted. The evidence phase has started for alert {alert_id}.
Add one quote from {chunk_id} to this question's evidence with the write_evidence action, copying it exactly from the passage's text."""


class TrustedController(BackendController):
    """The parent's Backend, refreshing the harness's trusted packet copy after each harness write."""

    def __init__(self, backend, on_write):
        super().__init__(backend)
        self.on_write = on_write

    async def reset_packets(self, query_set: str) -> Outcome:
        outcome = await super().reset_packets(query_set)
        if outcome.ok:
            self.on_write()
        return outcome

    async def write_evidence(self, quote) -> Outcome:
        outcome = await super().write_evidence(quote)
        if outcome.ok:
            self.on_write()
        return outcome


class CodeActAgent:
    kind = "codeact"
    via = "codeact"

    def __init__(self, backend, system_prompt: str, python: str, scratch_root: Path,
                 output_cap: int = DEFAULT_OUTPUT_CAP, timeout: int = DEFAULT_TIMEOUT):
        self.backend = backend
        self.system_prompt = system_prompt
        self.python = python
        self.scratch_root = Path(scratch_root)
        self.output_cap = output_cap
        self.timeout = timeout
        self.controller = TrustedController(backend, self.refresh_trusted)
        self.tool_defs = TOOL_DEFS
        self.tool_names = frozenset(t["function"]["name"] for t in TOOL_DEFS)
        self.trusted: dict[str, bytes] = {}
        self.scratch: Path | None = None
        self.alert_id: str | None = None

    # --- Shared-host interface -----------------------------------------------------

    def discovery_event(self, tracer) -> None:
        # No MCP discovery here: the actions are fixed. Recorded as a state change.
        tracer.emit("state_change", node="discovery", agent="codeact", tools=sorted(self.tool_names))

    def opening_messages(self, item: dict) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Question: {item['question']}\n"
                f"Asked by: {item['audience']}\n\n"
                f"Alert: {item['alert_id']}\n"
                f"Passage to judge: {item['chunk_id']}")},
        ]

    def evidence_message(self, chunk_id: str) -> dict:
        return {"role": "user", "content": EVIDENCE_PHASE_MESSAGE.format(chunk_id=chunk_id, alert_id=self.alert_id)}

    async def start_item(self, alert_id: str) -> None:
        """A fresh scratch folder per item; the previous item's is deleted."""
        self.alert_id = alert_id
        self.remove_scratch()
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.scratch = Path(tempfile.mkdtemp(prefix="item-", dir=self.scratch_root))

    async def execute(self, name: str, arguments: dict, state: dict) -> Outcome:
        if name == "classify":
            return await self.controller.classify(arguments, state["observed"])
        if name == "write_evidence":
            return await self.controller.write_evidence(arguments.get("quote"))
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return Outcome(False, error="refused: run_python needs a non-empty code string",
                           error_kind="refusal", refused_by="dispatcher")
        return await self.run_script(code)

    async def close(self) -> None:
        self.remove_scratch()

    # --- The worker ---------------------------------------------------------------------

    def worker_env(self) -> dict:
        """
        A minimal environment: no provider key or other secrets from the
        parent. The backend's folders, and HOME and TMPDIR in the scratch.
        """
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(HW2_DIR),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(self.scratch),
            "TMPDIR": str(self.scratch),
            # Absolute: the worker runs in its scratch folder, so a relative
            # path (such as --input agent/prepared/dev) would point nowhere.
            PREPARED_DIR_ENV: str(self.backend.prepared_dir.resolve()),
            EVIDENCE_DIR_ENV: str(self.backend.evidence_dir.resolve()),
        }

    async def run_script(self, code: str) -> Outcome:
        proc = await asyncio.create_subprocess_exec(
            self.python, "-", cwd=self.scratch, env=self.worker_env(),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(code.encode()), self.timeout)
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            stdout, stderr = await proc.communicate()
        exit_code = proc.returncode

        shown = self.compose_output(exit_code, stdout.decode("utf-8", "replace"),
                                    stderr.decode("utf-8", "replace"), timed_out)
        delivered, unverified = self.delivered_passages(shown)
        restored = self.restore_packets()

        trace = {"exit_code": exit_code, "delivered": sorted(delivered)}
        if unverified:
            trace["unverified"] = unverified
        if restored:
            trace["restored"] = restored
        ok = exit_code == 0 and not timed_out
        return Outcome(ok, payload=None if not ok else {"exit_code": exit_code},
                       error=None if ok else shown,
                       error_kind=None if ok else "execution",
                       delivered=delivered, trace=trace, text=shown)

    def compose_output(self, exit_code: int, stdout: str, stderr: str, timed_out: bool) -> str:
        """What the model sees, capped at output_cap characters."""
        head = f"timed out after {self.timeout} s\n" if timed_out else ""
        text = f"{head}exit code: {exit_code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        if len(text) > self.output_cap:
            text = text[: self.output_cap] + f"\n[output truncated at {self.output_cap} characters]"
        return text

    def delivered_passages(self, shown: str) -> tuple[set[str], list[dict]]:
        """
        Passages the model really saw: marker blocks in the shown output whose
        text is the chunk's exact canonical text and whose chunk is in the
        current item's allowed set. Anything else is recorded as unverified.
        """
        delivered, unverified = set(), []
        allowed = self.backend.allowed_chunk_ids or set()
        for match in BLOCK.finditer(shown):
            chunk_id, text = match.group("id"), match.group("text")
            if chunk_id not in allowed:
                unverified.append({"chunk_id": chunk_id, "reason": "outside the allowed set"})
                continue
            record = hw1_tools.validate_id(chunk_id, self.backend.prepared_dir)
            if not record["ok"] or record["chunk"]["text"] != text:
                unverified.append({"chunk_id": chunk_id, "reason": "text does not match the canonical passage"})
                continue
            delivered.add(chunk_id)
        return delivered, unverified

    # --- Detect and restore ------------------------------------------------------------

    def packet_files(self) -> dict[str, bytes]:
        folder = self.backend.evidence_dir
        if not folder.is_dir():
            return {}
        return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}

    def refresh_trusted(self) -> None:
        """Called after each harness write: the packets on disk are now the trusted state."""
        self.trusted = self.packet_files()

    def restore_packets(self) -> list[dict]:
        """
        Reverts any change to the evidence folder that the harness did not
        make, and returns what was reverted. Detection and restoration, not
        prevention: the change existed until now (brief line 280).
        """
        folder = self.backend.evidence_dir
        current = self.packet_files()
        restored = []
        for name in sorted(set(current) | set(self.trusted)):
            if current.get(name) == self.trusted.get(name):
                continue
            path = folder / name
            if name in self.trusted:
                path.write_bytes(self.trusted[name])
                restored.append({"file": name, "action": "reverted" if name in current else "recreated"})
            else:
                path.unlink()
                restored.append({"file": name, "action": "removed"})
        return restored

    def remove_scratch(self) -> None:
        if self.scratch is not None:
            shutil.rmtree(self.scratch, ignore_errors=True)
            self.scratch = None
