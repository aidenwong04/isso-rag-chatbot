"""
workspace - the disposable copy of the corpus the model's tools read (A7).

The CLI writes the copy into a temporary directory before the run and the
directory is deleted when the run ends, crashed or not. The original
chunks.json is never written to, so nothing has to be restored.

The copy holds only what the model may see: no archived page, and no
embedding. Withholding them is then not up to the shared chunk-id check alone;
a chunk that is not in the copy cannot be shown by any tool.

Discovery is not a model tool, so it keeps reading the original through
retrieve(), which needs the embeddings (section 7: discovery scans the whole
scope). The copy is built from the same in-memory records load_index() read
once and hashed into the corpus version, so discovery and the tools cannot see
two different corpora.
"""

import json
from pathlib import Path

CORPUS_FILE = "chunks.json"

# Per chunk, what goes into the copy. embedding is withheld; archived is
# dropped because every chunk left in the copy would have it false.
COPIED_FIELDS = ("id", "source", "url", "heading", "text", "breadcrumb", "audience")


def write_copy(records, workspace: Path) -> Path:
    """Writes the non-archived records, cut down to COPIED_FIELDS, into workspace. Returns the file."""
    copy = [
        {field: record[field] for field in COPIED_FIELDS}
        for record in records
        if not record["archived"]
    ]
    path = workspace / CORPUS_FILE
    with open(path, "w") as f:
        json.dump(copy, f, ensure_ascii=False)
    return path


def load_copy(workspace: Path) -> list[dict]:
    with open(workspace / CORPUS_FILE) as f:
        return json.load(f)
