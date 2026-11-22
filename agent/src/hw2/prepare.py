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

import json
from pathlib import Path
import hashlib

# --- Paths -------------------------------------------------------------------
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
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Returns the hex SHA-256 of the file's exact bytes."""
    return sha256_bytes(path.read_bytes())


def serialize(obj, *, indent: int | None, trailing_newline: bool) -> bytes:
    """
    Turns obj into the exact bytes that get written and hashed.

    json.dumps keeps list order and dict insertion order, so the same input
    always gives the same bytes. ensure_ascii=False keeps non-ASCII text as
    UTF-8 instead of \\u escapes. describe_serialization() states these same
    choices in the manifest.
    """
    text = json.dumps(obj, ensure_ascii=False, indent=indent)
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def describe_serialization(*, indent: int | None, trailing_newline: bool) -> str:
    """The manifest's description of what serialize() did with these arguments."""
    layout = "compact" if indent is None else f"indent={indent}"
    ending = "plus one LF" if trailing_newline else "no trailing newline"
    return f"UTF-8 json.dumps(obj, ensure_ascii=False, {layout}), {ending}"


def write_bytes(path: Path, data: bytes) -> None:
    """Writes data to path, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# --- Step 1: verify the source -----------------------------------------------

def load_verified_corpus(path: Path, expected_sha256: str) -> list[dict]:
    """
    Checks the source corpus against the pinned hash, then returns its records.

    TODO: decide what happens on a mismatch.
    """
    # Read once, so the bytes that are parsed are the bytes that were hashed.
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise ValueError(
            f"{path} does not match the pinned hash: expected {expected_sha256}, got {actual}"
        )

    return json.loads(raw)


# --- Step 2: canonical corpus view ------------------------------------------

def build_canonical_view(records: list[dict]) -> list[dict]:
    """
    Returns the agent-facing corpus: non-archived records only, each cut down
    to id, source, url, heading, text, breadcrumb, audience. Source order kept.
    """
    res = []
    for chunk in records:
        if not chunk["archived"]:
            cur_dict = {"id":chunk["id"],
                        "source":chunk["source"],
                        "url":chunk["url"],
                        "heading":chunk["heading"],
                        "text":chunk["text"],
                        "breadcrumb":chunk["breadcrumb"],
                        "audience":chunk["audience"]
                        }
            res.append(cur_dict)

    return res

# --- Step 3: frozen candidate lists -----------------------------------------

def freeze_candidates(report_path: Path) -> list[dict]:
    """
    Returns one {alert_id, query_id, chunk_id} per row of a HW1 full-run
    report, in the report's row order.

    query_id comes from test_id; chunk_id is path + "::" + str(line).
    Nothing else from the row (label, rationale, evidence, confidence) is kept.
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    alerts = data["alerts"] # list of dicts
    res = []

    for alert in alerts:
        cur_dict = {
            "alert_id":alert["alert_id"],
            "query_id":alert["test_id"],
            "chunk_id":f"{alert['path']}::{alert['line']}"
        }
        res.append(cur_dict)
        
    return res


# --- Step 4: frozen neighbor relationships ----------------------------------

def build_neighbors(view: list[dict]) -> list[dict]:
    """
    Returns, for every chunk in the canonical view, its immediate previous and
    next chunk on the same page: [{"chunk_id": ..., "neighbors": [prev, next]}].

    Previous before next; a page-edge chunk omits the missing side.
    """
    def page_and_index(record: dict) -> tuple[str, int]:
        # Sort key: page first, then the chunk index as an integer, so ::10 follows ::9.
        page, index = record["id"].rsplit("::", 1)
        return page, int(index)

    # A new list; view itself keeps source order for prepared/chunks.json.
    ordered = sorted(view, key=page_and_index)

    res = []
    for idx in range(len(ordered)):
        chunk_id = ordered[idx]["id"]
        page, _ = page_and_index(ordered[idx])
        neighbors = []
        # idx > 0, not a try: ordered[-1] would wrap around to the last chunk.
        if idx > 0:
            prev_pg, _ = page_and_index(ordered[idx - 1])
            if prev_pg == page:
                neighbors.append(ordered[idx - 1]["id"])

        if idx + 1 < len(ordered):
            next_pg, _ = page_and_index(ordered[idx + 1])
            if next_pg == page:
                neighbors.append(ordered[idx + 1]["id"])

        res.append({"chunk_id": chunk_id, "neighbors": neighbors})

    return res


# --- Step 5: manifest --------------------------------------------------------

MANIFEST_NAME = "manifest.json"

# Serialization per prepared file. The view matches the example manifest's
# compact form; candidates and neighbors are indented, ending in one LF.
VIEW_FORMAT = {"indent": None, "trailing_newline": False}
LIST_FORMAT = {"indent": 2, "trailing_newline": True}


def source_entry(path: Path) -> dict:
    """A manifest record for an input: its repo-relative path and hash."""
    return {
        "source_path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def prepare_file(relative_path: str, data: bytes, description: str) -> dict:
    """
    Writes data to OUTPUT_DIR/relative_path and returns its manifest record.

    The hash is taken from the same bytes that were written. verify_prepared()
    finds every record by its "prepared_path" key, so each prepared file must
    go through here.
    """
    write_bytes(OUTPUT_DIR / relative_path, data)
    return {
        "prepared_path": relative_path,
        "sha256": sha256_bytes(data),
        "serialization": description,
    }


def build_manifest(corpus: dict, query_sets: dict, neighbors: dict) -> dict:
    """
    Returns the manifest: source hashes, prepared-file hashes, record and pair
    counts, and how each prepared file was serialized.

    corpus, query_sets, and neighbors are the sections main() assembled; this
    adds the run-wide fields and fixes the key order.
    """
    return {
        "corpus_version": EXPECTED_CORPUS_SHA256[:12],
        "k": 3,
        "corpus": corpus,
        "query_sets": query_sets,
        "neighbors": neighbors,
    }


def _prepared_records(node):
    """Yields every manifest record that names a prepared file, at any depth."""
    if isinstance(node, dict):
        if "prepared_path" in node:
            yield node
        for value in node.values():
            yield from _prepared_records(value)
    elif isinstance(node, list):
        for value in node:
            yield from _prepared_records(value)


def verify_prepared(manifest_path: Path) -> None:
    """
    Re-hashes every prepared file and fails if any differs from the manifest.
    The harness calls this before each run.

    Prepared paths are relative to the manifest's folder, so this works on the
    local prepared/ folder and on a staged copy inside the sandbox image.
    Source files are not rechecked: they are not staged into the sandbox.
    """
    manifest = json.loads(manifest_path.read_bytes())
    root = manifest_path.parent
    records = list(_prepared_records(manifest))
    if not records:
        raise ValueError(f"{manifest_path} lists no prepared files")

    problems = []
    for record in records:
        path = root / record["prepared_path"]
        if not path.is_file():
            problems.append(f"{record['prepared_path']}: missing")
            continue
        actual = sha256_file(path)
        if actual != record["sha256"]:
            problems.append(
                f"{record['prepared_path']}: expected {record['sha256']}, got {actual}"
            )
    if problems:
        raise ValueError("prepared inputs do not match the manifest:\n  " + "\n  ".join(problems))


# --- Entry point -------------------------------------------------------------

def main() -> None:
    """Runs steps 1-5, writes everything under OUTPUT_DIR, then verifies it."""
    # Step 1: verify the source.
    records = load_verified_corpus(SOURCE_CORPUS, EXPECTED_CORPUS_SHA256)

    # Step 2: canonical corpus view.
    view = build_canonical_view(records)
    corpus = {
        "source": source_entry(SOURCE_CORPUS),
        "source_record_count": len(records),
        "view": prepare_file(
            "chunks.json",
            serialize(view, **VIEW_FORMAT),
            describe_serialization(**VIEW_FORMAT)
            + "; source record order; records with archived == false only",
        ),
        "view_record_count": len(view),
        "view_fields": list(view[0]) if view else [],
    }

    # Step 3: frozen candidate lists, and the query files copied unchanged.
    query_sets = {}
    for name, (queries_path, report_path) in QUERY_SETS.items():
        queries_bytes = queries_path.read_bytes()
        candidates = freeze_candidates(report_path)
        query_sets[name] = {
            "query_count": len(json.loads(queries_bytes)),
            "pair_count": len(candidates),
            "queries_source": source_entry(queries_path),
            "queries": prepare_file(
                f"{name}/queries.json", queries_bytes, "byte-for-byte copy of the source"
            ),
            "candidate_source": source_entry(report_path),
            "candidates": prepare_file(
                f"{name}/candidates.json",
                serialize(candidates, **LIST_FORMAT),
                describe_serialization(**LIST_FORMAT)
                + "; source report row order; alert_id, query_id, chunk_id only",
            ),
        }

    # Step 4: frozen neighbor relationships.
    neighbor_records = build_neighbors(view)
    neighbors = {
        "record_count": len(neighbor_records),
        "file": prepare_file(
            "neighbors.json",
            serialize(neighbor_records, **LIST_FORMAT),
            describe_serialization(**LIST_FORMAT)
            + "; each record holds chunk_id then neighbors (previous before next)",
        ),
    }

    # Step 5: manifest, then check everything just written against it.
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    manifest = build_manifest(corpus, query_sets, neighbors)
    write_bytes(manifest_path, serialize(manifest, **LIST_FORMAT))
    verify_prepared(manifest_path)

    print(f"prepared inputs written to {OUTPUT_DIR}")
    print(f"manifest sha256: {sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
