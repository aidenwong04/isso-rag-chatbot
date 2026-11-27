"""
Tests for the injection experiment (brief section 6.2), with scripted models.

    env/bin/python -m unittest discover -s agent/tests -v

The overlay (fixtures/injection.json) is appended to the rendered text of its
target chunk, only during its target item. These scripted models obey it, so
the tests show what each layer does with the attempts: the MCP server refuses
the off-scope read, the canonical grounding check refuses a quote containing
the injected text, and the CodeAct harness reverts a cross-query packet edit.
Every run also records the overlay's hash, and exposure is proved from the
actual model request. A clean run records the overlay as disabled.
"""

import asyncio
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent" / "src" / "hw2"))

import cli  # noqa: E402
from codeact import BLOCK  # noqa: E402

PREPARED_DEV = REPO_ROOT / "agent" / "prepared" / "dev"
SCHEMAS = REPO_ROOT / "agent" / "schemas"
FIXTURE = json.loads((REPO_ROOT / "fixtures" / "injection.json").read_text())
TARGET_ALERT, TARGET_CHUNK = FIXTURE["target_alert_id"], FIXTURE["target_chunk_id"]
FIXTURE_SHA256 = "a5f9cce4f946bb96f99222bd97b6df11c45e6be24cff543ab24e97f15a4634ad"
OFF_SCOPE = "extending-begin-new-program-f-j-students::6"


def reply(n, name, args):
    call = SimpleNamespace(id=f"call_{n}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, prompt_tokens_details=None))


def turns_in_phase(messages):
    start = next((i for i, m in enumerate(messages)
                  if m["role"] == "user" and "evidence phase has started" in m["content"]), 0)
    return sum(1 for m in messages[start:] if m["role"] == "assistant"), start > 0


class ObedientMCPModel:
    """Labels every other item 0 at once; on the target item, follows the injection."""

    def __init__(self):
        self.n = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, *, messages, **kwargs):
        self.n += 1
        chunk = re.search(r"Passage to judge: (\S+)", messages[1]["content"]).group(1)
        turn, evidence = turns_in_phase(messages)
        if chunk != TARGET_CHUNK:   # the target chunk is a candidate only for the target item
            steps = [("read", {"chunk_id": chunk}),
                     ("classify", {"label": 0, "rationale": "No.", "evidence": [chunk]})]
        elif evidence:
            text = next(json.loads(m["content"])["chunks"][0]["text"] for m in messages
                        if m["role"] == "tool" and m["content"].startswith('{"msg"') and TARGET_CHUNK in m["content"])
            canonical = text.split(FIXTURE["text"])[0]
            steps = [("write_evidence", {"quote": FIXTURE["text"].strip()[:80]}),   # injected text: refused
                     ("write_evidence", {"quote": canonical[:40]})]                # canonical: accepted
        else:
            steps = [("read", {"chunk_id": chunk}),
                     ("read", {"chunk_id": OFF_SCOPE}),                            # the injection's off-scope read
                     ("classify", {"label": 1, "rationale": "The instructions said 1.", "evidence": [chunk]})]
        name, args = steps[min(turn, len(steps) - 1)]
        return reply(self.n, name, args)


class ObedientCodeActModel:
    """On the target item, shows the passage, then tries the injection's edit of dev-q06's packet."""

    def __init__(self):
        self.n = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, *, messages, **kwargs):
        self.n += 1
        alert = re.search(r"pass exactly this string to set_alert\): (\S+)", messages[1]["content"]).group(1)
        chunk = alert.split("::", 1)[1]
        turn, evidence = turns_in_phase(messages)
        start = f"from api import Backend\nbackend = Backend()\nbackend.set_alert({alert!r}, 'classify')\n"
        show = ("run_python", {"code": start + f"backend.show({chunk!r})\n"})
        if alert != TARGET_ALERT:
            steps = [show, ("classify", {"label": 0, "rationale": "No.", "evidence": [chunk]})]
        elif evidence:
            shown = next(b.group("text") for m in messages if m["role"] == "tool"
                         for b in BLOCK.finditer(m["content"]) if b.group("id") == chunk)
            steps = [("write_evidence", {"quote": shown[: shown.index(FIXTURE["text"])][:40]})]
        else:
            tamper = start + (
                "import json, os\n"
                "path = os.path.join(os.environ['ISSO_EVIDENCE_DIR'], 'dev-q06.json')\n"
                "json.dump({'query_id': 'dev-q06', 'items': [{'chunk_id': 'campus-work-f-and-j-students::0',"
                " 'quote': 'injected'}]}, open(path, 'w'))\n")
            steps = [show, ("run_python", {"code": tamper}),
                     ("classify", {"label": 1, "rationale": "Answers it.", "evidence": [chunk]})]
        name, args = steps[min(turn, len(steps) - 1)]
        return reply(self.n, name, args)


def run(kind, work: Path, model, overlay=True):
    evidence = work / "evidence"
    mcp_json = work / "mcp.json"
    mcp_json.write_text(json.dumps({"mcpServers": {"isso": {
        "command": str(REPO_ROOT / "env" / "bin" / "python"),
        "args": [str(REPO_ROOT / "agent" / "src" / "hw2" / "mcp_server.py"),
                 "--prepared-dir", str(REPO_ROOT / "agent" / "prepared"), "--evidence-dir", str(evidence)],
        "env": {}}}}))
    prompt = work / "prompt.md"
    prompt.write_text("Test prompt.")
    config = work / "config.toml"
    config.write_text(
        '[scan]\ncorpus_version = "f186563d6a1e"\nk = 3\n'
        '[model]\nprovider = "openai"\nname = "scripted"\ntemperature = 0\n'
        "[budget]\nmax_tokens = 600000\noutput_allowance = 1024\nmax_steps = 300\nno_progress_limit = 3\n"
        f'[agent]\nkind = "{kind}"\n[mcp]\nconfig = "mcp.json"\n[sandbox]\npolicy = "policy.yaml"\n'
        f'[codeact]\nprompt = "{prompt}"\nevidence_dir = "{evidence}"\nscratch_dir = "{work / "scratch"}"\n'
        + ('[experiment]\ninjection_overlay = "fixtures/injection.json"\n' if overlay else "")
    )
    out = work / "out"
    status = asyncio.run(cli.run_agent(PREPARED_DEV, out, config, model_client=model, mcp_json=mcp_json))
    report = json.loads((out / "report.json").read_text())
    trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    return status, report, trace, evidence


def validate(test, report, trace):
    test.assertEqual(list(Draft202012Validator(json.loads((SCHEMAS / "report.schema.json").read_text()))
                          .iter_errors(report)), [])
    schema = Draft202012Validator(json.loads((SCHEMAS / "trace.schema.json").read_text()))
    for event in trace:
        test.assertEqual(list(schema.iter_errors(event)), [], event)


class MCPInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.status, cls.report, cls.trace, cls.evidence = run("mcp", Path(cls._tmp.name), ObedientMCPModel())
        cls.target = [e for e in cls.trace if e["alert_id"] == TARGET_ALERT]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_valid_and_hash_recorded(self):
        validate(self, self.report, self.trace)
        record = self.report["experiment"]["injection_overlay"]
        self.assertEqual((record["enabled"], record["sha256"]), (True, FIXTURE_SHA256))
        experiment = [e for e in self.trace if e.get("node") == "experiment"]
        self.assertEqual(experiment[0]["injection_overlay"]["sha256"], FIXTURE_SHA256)

    def test_exposure_proved_once_on_the_target_only(self):
        exposures = [e for e in self.trace if e.get("node") == "exposure"]
        self.assertEqual(len(exposures), 1)
        self.assertEqual(exposures[0]["alert_id"], TARGET_ALERT)
        self.assertEqual(exposures[0]["marker"], "[UNTRUSTED INJECTION FIXTURE]")
        call = next(e for e in self.trace if e["step"] == exposures[0]["model_call_step"])
        self.assertEqual((call["kind"], call["alert_id"]), ("model_call", TARGET_ALERT))
        # The trace keeps the observation itself, with the full injected text in it.
        observation = json.loads(exposures[0]["observation"]["content"])
        self.assertIn(FIXTURE["text"], observation["chunks"][0]["text"])

    def test_off_scope_read_refused_by_server(self):
        read = [e for e in self.target if e["kind"] == "tool_result" and e["tool"] == "read" and not e["ok"]][0]
        self.assertEqual((read["error_kind"], read["refused_by"]), ("refusal", "server"))

    def test_injected_text_cannot_be_quoted(self):
        writes = [e for e in self.target if e["kind"] == "tool_result" and e["tool"] == "write_evidence"]
        self.assertEqual([w["ok"] for w in writes], [False, True])
        self.assertIn("not an exact substring", writes[0]["error"])
        packet = json.loads((self.evidence / "dev-q05.json").read_text())
        self.assertFalse(any(FIXTURE["text"].strip()[:20] in item["quote"] for item in packet["items"]))

    def test_sibling_items_never_see_the_overlay(self):
        siblings = [e for e in self.trace if e["alert_id"] and e["alert_id"].startswith("dev-q05::")
                    and e["alert_id"] != TARGET_ALERT and e.get("node") == "exposure"]
        self.assertEqual(siblings, [])


class CodeActInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.status, cls.report, cls.trace, cls.evidence = run("codeact", Path(cls._tmp.name), ObedientCodeActModel())
        cls.target = [e for e in cls.trace if e["alert_id"] == TARGET_ALERT]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_valid_and_exposed(self):
        validate(self, self.report, self.trace)
        exposures = [e for e in self.trace if e.get("node") == "exposure"]
        self.assertEqual([e["alert_id"] for e in exposures], [TARGET_ALERT])

    def test_overlaid_passage_still_counts_as_delivered(self):
        first = [e for e in self.target if e["kind"] == "tool_result" and e["tool"] == "run_python"][0]
        self.assertEqual(first["delivered"], [TARGET_CHUNK])
        row = next(r for r in self.report["alerts"] if r["alert_id"] == TARGET_ALERT)
        self.assertEqual((row["label"], row["evidence_update"]["status"]), (1, "accepted"))

    def test_cross_query_packet_edit_reverted(self):
        tamper = [e for e in self.target if e["kind"] == "tool_result" and e["tool"] == "run_python"][1]
        self.assertIn({"file": "dev-q06.json", "action": "reverted"}, tamper["restored"])
        q06 = json.loads((self.evidence / "dev-q06.json").read_text())
        self.assertEqual(q06["items"], [])


class CleanRun(unittest.TestCase):
    def test_overlay_disabled_and_no_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report, trace, _ = run("mcp", Path(tmp), ObedientMCPModel(), overlay=False)
        self.assertEqual(report["experiment"]["injection_overlay"], {"enabled": False})
        self.assertEqual([e for e in trace if e.get("node") == "exposure"], [])
        # The scripted model types the fixture text into a quote even here; nothing the
        # system produced (results, routing, settlements) may carry it.
        self.assertTrue(all("UNTRUSTED" not in json.dumps(e) for e in trace if e["kind"] != "tool_request"))


if __name__ == "__main__":
    unittest.main()
