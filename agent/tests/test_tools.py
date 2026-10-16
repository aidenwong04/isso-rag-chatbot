"""
Tests for the agent's tools.

    env/bin/python -m unittest discover -s agent/tests -v

The tools are run against a small hand-made corpus instead of the pinned one,
so the cases the real corpus does not contain (an archived chunk, a page with
one chunk, a page with no stated audience next to one with it) can be tested,
and no test depends on what chunks.json holds today. The corpus goes through
the same workspace copy a real run makes (workspace.py), so the tests also
cover what the copy withholds.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import tools
import workspace


def record(chunk_id, audience="students", archived=False):
    page = chunk_id.split("::")[0]
    return {
        "id": chunk_id,
        "source": page,
        "url": f"https://example.test/{page}",
        "heading": f"heading of {chunk_id}",
        "text": f"text of {chunk_id}",
        "breadcrumb": ["Home", page],
        "audience": audience,
        "archived": archived,
        "embedding": [0.1, 0.2],
    }


FAKE_RECORDS = [
    record("page::0"),
    record("page::1"),
    record("page::2"),
    record("solo::0"),
    record("open::0", audience=None),
    record("gone::0", archived=True),
]


class ToolsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name)
        workspace.write_copy(FAKE_RECORDS, self.ws)

    def tearDown(self):
        tools._lookups.pop(self.ws, None)
        self._tmp.cleanup()


class WorkspaceCopyTest(ToolsTestCase):
    def copied(self):
        return json.loads((self.ws / workspace.CORPUS_FILE).read_text())

    def test_leaves_out_archived_pages(self):
        self.assertNotIn("gone::0", {r["id"] for r in self.copied()})
        self.assertEqual(len(self.copied()), len(FAKE_RECORDS) - 1)

    def test_keeps_only_the_copied_fields(self):
        for r in self.copied():
            self.assertEqual(tuple(r), workspace.COPIED_FIELDS)

    def test_does_not_change_the_records_it_copies(self):
        self.assertIn("embedding", FAKE_RECORDS[0])


class ValidateIdTest(ToolsTestCase):
    def test_accepts_an_existing_id_and_returns_its_record(self):
        result = tools.validate_id("page::1", self.ws)
        self.assertTrue(result["ok"])
        self.assertEqual(result["chunk"]["id"], "page::1")

    def test_refuses_malformed_ids(self):
        for chunk_id in [7, ["a"], None, "page", "a::b::c", "::5", "page::", "page::x", "page::-1"]:
            with self.subTest(chunk_id=chunk_id):
                result = tools.validate_id(chunk_id, self.ws)
                self.assertFalse(result["ok"])
                self.assertIn("Malformed", result["msg"])

    def test_refuses_a_nonexistent_id(self):
        result = tools.validate_id("page::999", self.ws)
        self.assertFalse(result["ok"])
        self.assertIn("Nonexistent", result["msg"])

    def test_refuses_an_archived_id_as_nonexistent(self):
        # Archived pages are not in the workspace copy at all.
        result = tools.validate_id("gone::0", self.ws)
        self.assertFalse(result["ok"])
        self.assertIn("Nonexistent", result["msg"])


class ReadTest(ToolsTestCase):
    def test_returns_the_one_chunk_with_the_contract_fields(self):
        result = tools.read("page::1", self.ws)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(
            set(result["chunks"][0]),
            {"chunk_id", "text", "heading", "breadcrumb", "audience"},
        )
        self.assertEqual(result["chunks"][0]["chunk_id"], "page::1")

    def test_a_page_with_no_audience_reads_as_not_stated(self):
        self.assertEqual(tools.read("open::0", self.ws)["chunks"][0]["audience"], "not stated")

    def test_changing_a_result_does_not_change_the_corpus(self):
        tools.read("page::1", self.ws)["chunks"][0]["breadcrumb"].append("changed")
        self.assertEqual(tools.read("page::1", self.ws)["chunks"][0]["breadcrumb"], ["Home", "page"])

    def test_refuses_bad_ids_without_raising(self):
        for chunk_id in [7, None, "page", "page::999", "gone::0"]:
            with self.subTest(chunk_id=chunk_id):
                result = tools.read(chunk_id, self.ws)
                self.assertFalse(result["ok"])
                self.assertTrue(result["msg"])


class ReadNeighborsTest(ToolsTestCase):
    def neighbor_ids(self, chunk_id):
        result = tools.read_neighbors(chunk_id, self.ws)
        self.assertTrue(result["ok"])
        return [chunk["chunk_id"] for chunk in result["chunks"]]

    def test_middle_chunk_returns_previous_then_next(self):
        self.assertEqual(self.neighbor_ids("page::1"), ["page::0", "page::2"])

    def test_first_chunk_returns_only_the_next(self):
        self.assertEqual(self.neighbor_ids("page::0"), ["page::1"])

    def test_last_chunk_returns_only_the_previous(self):
        self.assertEqual(self.neighbor_ids("page::2"), ["page::1"])

    def test_single_chunk_page_returns_an_empty_list(self):
        self.assertEqual(self.neighbor_ids("solo::0"), [])

    def test_refuses_bad_ids_without_raising(self):
        for chunk_id in [7, None, "page", "page::999", "gone::0"]:
            with self.subTest(chunk_id=chunk_id):
                self.assertFalse(tools.read_neighbors(chunk_id, self.ws)["ok"])


class ClassifyTest(ToolsTestCase):
    SHOWN = {"page::0", "page::1"}

    def classify(self, label=1, rationale="because", evidence=("page::0",)):
        if isinstance(evidence, tuple):
            evidence = list(evidence)
        return tools.classify(label, rationale, evidence, self.SHOWN, self.ws)

    def test_accepts_a_valid_judgment(self):
        for label in [0, 1]:
            with self.subTest(label=label):
                self.assertTrue(self.classify(label=label)["ok"])
        self.assertTrue(self.classify(evidence=["page::0", "page::1"])["ok"])
        self.assertTrue(self.classify(rationale="x" * 600)["ok"])

    def test_refuses_a_label_that_is_not_the_integer_0_or_1(self):
        for label in [2, -1, "1", 1.0, None, True, False]:
            with self.subTest(label=label):
                self.assertFalse(self.classify(label=label)["ok"])

    def test_refuses_a_bad_rationale(self):
        for rationale in [None, 123, "", "   ", "x" * 601]:
            with self.subTest(rationale=rationale):
                self.assertFalse(self.classify(rationale=rationale)["ok"])

    def test_refuses_bad_evidence(self):
        cases = {
            "empty": [],
            "none": None,
            "bare string": "page::0",
            "not ids": [7],
            "nested": [["page::0"]],
            "malformed id": ["page"],
            "not shown": ["page::2"],
            "one shown, one not": ["page::0", "page::2"],
        }
        for name, evidence in cases.items():
            with self.subTest(name):
                self.assertFalse(self.classify(evidence=evidence)["ok"])


if __name__ == "__main__":
    unittest.main()
