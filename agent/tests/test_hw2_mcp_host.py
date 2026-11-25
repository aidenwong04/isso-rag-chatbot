"""
End-to-end tests for the HW2 MCP host, with a scripted stand-in for the model.

    env/bin/python -m unittest discover -s agent/tests -v

Only the model is replaced. The host starts the real MCP server from mcp.json
over stdio, discovers its tools, and runs the real graph against the frozen
dev inputs, so these tests cover discovery, the dispatcher, the server's
refusals, phases, settlement, evidence updates, and finalize. Each run writes
packets to a temporary evidence folder (a temporary mcp.json points there).

The script gives some items deliberately bad behavior: a host-only control, an
out-of-scope read, an unknown tool, malformed JSON, a string label, repeated
reads, a stalled evidence phase, and forged quotes. Every report and trace
line is validated against the HW2 schemas.
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent" / "src" / "hw2"))

import cli  # noqa: E402

PREPARED_DEV = REPO_ROOT / "agent" / "prepared" / "dev"
SCHEMAS = REPO_ROOT / "agent" / "schemas"


# --- The scripted model --------------------------------------------------------------

def call(name, arguments, raw=None):
    return {"name": name, "arguments": raw if raw is not None else json.dumps(arguments)}


class ScriptedModel:
    """
    Stands in for AsyncOpenAI. For each conversation it looks at the item and
    the messages so far and returns the scripted next tool call. Items are
    identified by their position in the frozen dev order.
    """

    def __init__(self):
        self.order = [c["chunk_id"] for c in json.loads((PREPARED_DEV / "candidates.json").read_text())]
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @staticmethod
    def target(messages):
        first_user = next(m for m in messages if m["role"] == "user")
        return first_user["content"].rsplit("Passage to judge: ", 1)[1].strip()

    @staticmethod
    def target_text(messages, chunk_id):
        for m in messages:
            if m["role"] == "tool" and m["content"].startswith("{"):
                for chunk in json.loads(m["content"]).get("chunks", []):
                    if chunk["chunk_id"] == chunk_id:
                        return chunk["text"]
        raise AssertionError("the target was never read")

    def script(self, i, chunk, messages):
        """The ordered tool calls for item i, given how many turns it has had."""
        page = chunk.rsplit("::", 1)[0]
        normal = [call("read", {"chunk_id": chunk}),
                  call("classify", {"label": 1, "rationale": "It answers part of the question.", "evidence": [chunk]})]
        evidence_phase = any(m["role"] == "user" and "evidence phase has started" in m["content"] for m in messages)
        if evidence_phase:
            text = self.target_text(messages, chunk)
            if i == 6:   # stalls: only reads the packet
                return [call("read_packet", {})] * 10
            if i == 7:   # only forged quotes
                return [call("write_evidence", {"quote": "Not in the passage, version %d." % n}) for n in range(10)]
            if i == 3:   # one forged quote, then a real one
                return [call("write_evidence", {"quote": "This sentence is forged."}),
                        call("write_evidence", {"quote": text[:40]})]
            return [call("write_evidence", {"quote": text[:40]})]
        if i == 1:
            return [call("read", {"chunk_id": chunk}),
                    call("classify", {"label": 0, "rationale": "It does not answer the question.", "evidence": [chunk]})]
        if i == 2:   # a host-only control first
            return [call("set_alert", {"alert_id": "dev-q01::x::1", "phase": "evidence"}),
                    call("read", {"chunk_id": chunk}),
                    call("classify", {"label": 0, "rationale": "No.", "evidence": [chunk]})]
        if i == 3:   # out of scope, unknown tool, then normal
            return [call("read", {"chunk_id": "about-your-visa::3"}),  # outside dev-q04's allowed set
                    call("delete_everything", {}),
                    *normal]
        if i == 4:   # the same string label three times: the SDK refuses, and the cap ends it
            bad = call("classify", {"label": "1", "rationale": "Yes.", "evidence": [chunk]})
            return [call("read", {"chunk_id": chunk}), bad, bad, bad]
        if i == 5:   # re-reads the target: no progress after three repeats
            return [call("read", {"chunk_id": chunk})] * 10
        if i == 8:   # malformed JSON first
            return [call("read", None, raw='{"chunk_id": '), *normal]
        if i % 2 and i != 7:   # item 7 needs label 1 to reach its evidence script
            return [normal[0], call("classify", {"label": 0, "rationale": f"Not about {page}.", "evidence": [chunk]})]
        return normal

    async def create(self, *, model, messages, tools, **kwargs):
        self.calls += 1
        chunk = self.target(messages)
        i = self.order.index(chunk)
        turns = sum(1 for m in messages if m["role"] == "assistant")
        evidence_start = next((n for n, m in enumerate(messages)
                               if m["role"] == "user" and "evidence phase has started" in m["content"]), None)
        if evidence_start is not None:
            turns = sum(1 for m in messages[evidence_start:] if m["role"] == "assistant")
        planned = self.script(i, chunk, messages)
        step = planned[min(turns, len(planned) - 1)]
        tool_call = SimpleNamespace(id=f"call_{self.calls}",
                                    function=SimpleNamespace(name=step["name"], arguments=step["arguments"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, prompt_tokens_details=None),
        )


# --- Helpers ---------------------------------------------------------------------------

def run_host(work: Path, budget: dict | None = None):
    """Runs the host with a temporary mcp.json and config; returns (status, report, trace, evidence dir)."""
    evidence = work / "evidence"
    mcp_json = work / "mcp.json"
    mcp_json.write_text(json.dumps({"mcpServers": {"isso": {
        "command": str(REPO_ROOT / "env" / "bin" / "python"),
        "args": [str(REPO_ROOT / "agent" / "src" / "hw2" / "mcp_server.py"),
                 "--prepared-dir", str(REPO_ROOT / "agent" / "prepared"), "--evidence-dir", str(evidence)],
        "env": {}}}}))
    settings = {"max_tokens": 600000, "output_allowance": 1024, "max_steps": 300, "no_progress_limit": 3}
    settings.update(budget or {})
    config = work / "config.toml"
    config.write_text(
        '[scan]\ncorpus_version = "f186563d6a1e"\nk = 3\n'
        '[model]\nprovider = "openai"\nname = "scripted"\ntemperature = 0\n'
        "[budget]\n" + "".join(f"{k} = {v}\n" for k, v in settings.items()) +
        '[agent]\nkind = "mcp"\n'
        '[mcp]\nconfig = "mcp.json"\n'
        '[sandbox]\npolicy = "policy.yaml"\n'
    )
    output = work / "out"
    status = asyncio.run(cli.run_agent(PREPARED_DEV, output, config,
                                       model_client=ScriptedModel(), mcp_json=mcp_json))
    report = json.loads((output / "report.json").read_text())
    trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    return status, report, trace, evidence


def schema(name):
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text()))


# --- Tests -------------------------------------------------------------------------------

class FullRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.status, cls.report, cls.trace, cls.evidence = run_host(Path(cls._tmp.name))
        cls.rows = {r["alert_id"]: r for r in cls.report["alerts"]}
        cls.order = [c["alert_id"] for c in json.loads((PREPARED_DEV / "candidates.json").read_text())]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def row(self, i):
        return self.rows[self.order[i]]

    def events(self, i, **match):
        return [e for e in self.trace if e["alert_id"] == self.order[i]
                and all(e.get(k) == v for k, v in match.items())]

    def test_schemas(self):
        self.assertEqual(list(schema("report.schema.json").iter_errors(self.report)), [])
        trace_schema = schema("trace.schema.json")
        for event in self.trace:
            self.assertEqual(list(trace_schema.iter_errors(event)), [], event)

    def test_completed_in_frozen_order(self):
        self.assertEqual(self.status, "completed")
        self.assertEqual([r["alert_id"] for r in self.report["alerts"]], self.order)
        self.assertEqual(self.report["agent_kind"], "mcp")
        self.assertEqual(self.report["target"]["query_set"], "dev")

    def test_discovery_excludes_host_only(self):
        discovery = [e for e in self.trace if e["kind"] == "discovery"]
        self.assertEqual(len(discovery), 1)
        self.assertTrue({"read", "classify", "write_evidence"} <= set(discovery[0]["tools"]))
        self.assertFalse({"set_alert", "reset_packets"} & set(discovery[0]["tools"]))

    def test_one_settlement_and_one_update_per_row(self):
        for alert_id in self.order:
            settlements = [e for e in self.trace if e["alert_id"] == alert_id and "alert_status" in e]
            updates = [e for e in self.trace if e["alert_id"] == alert_id and e.get("node") == "evidence_update"]
            self.assertEqual(len(settlements), 1, alert_id)
            self.assertEqual(len(updates), 1, alert_id)

    def test_label_1_accepted_with_packet_on_disk(self):
        row = self.row(0)
        self.assertEqual((row["status"], row["label"]), ("classified", 1))
        self.assertEqual(row["evidence_update"]["status"], "accepted")
        self.assertTrue(all(row["evidence_update"]["checks"].values()))
        packet = json.loads((self.evidence / f"{row['query_id']}.json").read_text())
        self.assertIn(row["chunk_id"], [item["chunk_id"] for item in packet["items"]])
        self.assertTrue(self.events(0, kind="routing", decision="assemble_evidence", by="runtime"))

    def test_label_0_skips_evidence_by_runtime(self):
        row = self.row(1)
        self.assertEqual((row["label"], row["evidence_update"]["status"]), (0, "skipped"))
        self.assertTrue(self.events(1, kind="routing", decision="skip_evidence", by="runtime"))

    def test_host_only_refused_by_dispatcher(self):
        result = self.events(2, kind="tool_result", tool="set_alert")[0]
        self.assertEqual((result["ok"], result["error_kind"], result["refused_by"]), (False, "refusal", "dispatcher"))
        self.assertEqual(self.row(2)["label"], 0)  # the item went on normally

    def test_out_of_scope_refused_by_server_and_unknown_tool_by_dispatcher(self):
        read = self.events(3, kind="tool_result", tool="read")[0]
        self.assertEqual((read["error_kind"], read["refused_by"]), ("refusal", "server"))
        self.assertIn("refused: chunk_id outside allowed set", read["error"])
        unknown = self.events(3, kind="tool_result", tool="delete_everything")[0]
        self.assertEqual(unknown["refused_by"], "dispatcher")
        # A forged quote was refused, then a real one accepted.
        writes = self.events(3, kind="tool_result", tool="write_evidence")
        self.assertEqual([w["ok"] for w in writes], [False, True])
        self.assertEqual(self.row(3)["evidence_update"]["status"], "accepted")

    def test_string_label_refused_and_capped(self):
        results = self.events(4, kind="tool_result", tool="classify")
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["refused_by"] == "server" and "validation error" in r["error"] for r in results))
        self.assertEqual((self.row(4)["status"], self.row(4)["label"]), ("error", None))

    def test_repeated_reads_end_the_item_not_the_run(self):
        row = self.row(5)
        self.assertEqual(row["status"], "error")
        self.assertIn("No progress", row["rationale"])
        self.assertEqual(len(self.events(5, kind="model_call")), 4)  # one new read, then three repeats
        self.assertEqual(self.row(6)["status"], "classified")      # the next item still ran

    def test_stalled_evidence_phase_keeps_label(self):
        row = self.row(6)
        self.assertEqual((row["label"], row["evidence_update"]["status"]), (1, "skipped"))

    def test_forged_quotes_rejected(self):
        row = self.row(7)
        self.assertEqual((row["label"], row["evidence_update"]["status"]), (1, "rejected"))
        self.assertIn("refused: quote is not an exact substring", row["evidence_update"]["reason"])

    def test_malformed_json_refused_by_dispatcher(self):
        first = self.events(8, kind="tool_result")[0]
        self.assertEqual(first["refused_by"], "dispatcher")
        self.assertEqual(self.events(8, kind="tool_request")[0]["args"], {"raw": '{"chunk_id": '})

    def test_requests_and_results_pair_up(self):
        kinds = [e["kind"] for e in self.trace if e["kind"] in ("tool_request", "tool_result")]
        self.assertEqual(kinds, ["tool_request", "tool_result"] * (len(kinds) // 2))


class StepLimit(unittest.TestCase):
    def test_max_steps_ends_run_and_settles_leftovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, report, trace, _ = run_host(Path(tmp), {"max_steps": 5})
        self.assertEqual(status, "max_steps")
        self.assertEqual(list(schema("report.schema.json").iter_errors(report)), [])
        statuses = [r["status"] for r in report["alerts"]]
        self.assertEqual(statuses[0], "classified")
        self.assertTrue(all(s == "budget_exhausted" for s in statuses[2:]))
        self.assertTrue(all(r["evidence_update"]["status"] in ("accepted", "skipped") for r in report["alerts"]))
        self.assertEqual(trace[-1]["kind"], "terminal")


if __name__ == "__main__":
    unittest.main()
