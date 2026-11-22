"""
inputs:
Source corpus - data/embeddings/chunks.json
Query files ×3
HW1 full-run reports ×3

before running experiments, I need to prepare and freeze the corpus.
"Freeze and document the bytes and SHA-256 hashes of your prepared inputs before experiments."

outputs:
prepared/chunks.json - non-archvied records - (keep only id, source, url, heading, text, breadcrumb, audience)
prepared/{set}/candidates.json ×3 One {alert_id, query_id, chunk_id} per report row, in the original row order. No labels, rationales, scores or evidence.
prepared/neighbors.json - For each chunk, its immediate previous and next chunk IDs on the same page.
Query files - Copied unchanged (the brief says "retain the original query files").
prepared/manifest.json - SHA-256 hashes, plus a description of how each file is serialized, for every input and output above.

"""

import os
import json
from dotenv import load_dotenv
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# This file is agent/src/hw2/prepare.py, so the repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_DIR = REPO_ROOT / "agent"

SOURCE_CORPUS = REPO_ROOT / "data" / "embeddings" / "chunks.json"
EXPECTED_CORPUS_SHA256 = "f186563d6a1e702703386a25e5047e44eac3d666a7fe382d516c86dd47e5bf8e"

# Query set name -> (query file, HW1 full-run report the candidates come from).
QUERY_SETS = {
    "dev": (
        AGENT_DIR / "queries" / "dev" / "queries.json",
        AGENT_DIR / "runs" / "dev-full" / "report.json",
    ),
    "heldout-a": (
        AGENT_DIR / "queries" / "heldout-a" / "queries.json",
        AGENT_DIR / "runs" / "heldout-a-full" / "report.json",
    ),
    "heldout-b": (
        AGENT_DIR / "queries" / "heldout-b" / "queries.json",
        AGENT_DIR / "runs" / "heldout-b-full" / "report.json",
    ),
}

OUTPUT_DIR = AGENT_DIR / "prepared"


# --- Hashing and serialization ----------------------------------------------

def sha256_bytes(data: bytes) -> str:
    """Returns the hex SHA-256 of data."""
    raise NotImplementedError


def sha256_file(path: Path) -> str:
    """Returns the hex SHA-256 of the file's exact bytes."""
    raise NotImplementedError


def serialize(obj, *, indent: int | None, trailing_newline: bool) -> bytes:
    """
    Turns obj into the exact bytes that get written and hashed.

    TODO: every output's bytes must be identical across runs. The manifest
    records how each file was serialized, so whatever choices you make here
    have to match what you write there.
    """
    raise NotImplementedError


def write_bytes(path: Path, data: bytes) -> None:
    """Writes data to path, creating parent directories."""
    raise NotImplementedError


# --- Step 1: verify the source -----------------------------------------------

def load_verified_corpus(path: Path, expected_sha256: str) -> list[dict]:
    """
    Checks the source corpus against the pinned hash, then returns its records.

    TODO: decide what happens on a mismatch.
    """
    raise NotImplementedError


# --- Step 2: canonical corpus view ------------------------------------------

def build_canonical_view(records: list[dict]) -> list[dict]:
    """
    Returns the agent-facing corpus: non-archived records only, each cut down
    to id, source, url, heading, text, breadcrumb, audience. Source order kept.
    """
    raise NotImplementedError


# --- Step 3: frozen candidate lists -----------------------------------------

def freeze_candidates(report_path: Path) -> list[dict]:
    """
    Returns one {alert_id, query_id, chunk_id} per row of a HW1 full-run
    report, in the report's row order.

    query_id comes from test_id; chunk_id is path + "::" + str(line).
    Nothing else from the row (label, rationale, evidence, confidence) is kept.
    """
    raise NotImplementedError


# --- Step 4: frozen neighbor relationships ----------------------------------

def build_neighbors(view: list[dict]) -> list[dict]:
    """
    Returns, for every chunk in the canonical view, its immediate previous and
    next chunk on the same page: [{"chunk_id": ..., "neighbors": [prev, next]}].

    Previous before next; a page-edge chunk omits the missing side.
    """
    raise NotImplementedError


# --- Step 5: manifest --------------------------------------------------------

def build_manifest(entries: dict) -> dict:
    """
    Returns the manifest: source hashes, prepared-file hashes, record and pair
    counts, and how each prepared file was serialized.

    TODO: decide what `entries` carries; examples/prepared-manifest.example.json
    in the hw2-isso repo shows one shape.
    """
    raise NotImplementedError


def verify_prepared(manifest_path: Path) -> None:
    """
    Re-hashes every prepared file and fails if any differs from the manifest.
    The harness calls this before each run.
    """
    raise NotImplementedError


# --- Entry point -------------------------------------------------------------

def main() -> None:
    """
    Runs steps 1-5 and writes everything under OUTPUT_DIR.

    TODO:
      1. load_verified_corpus
      2. build_canonical_view, write prepared/chunks.json
      3. for each query set: freeze_candidates, write prepared/<set>/candidates.json,
         copy the query file unchanged
      4. build_neighbors, write prepared/neighbors.json
      5. build_manifest, write prepared/manifest.json
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
