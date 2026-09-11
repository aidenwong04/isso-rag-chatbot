"""Embed every chunk in data/processed/*.json with Gemini and write a single
local index file: data/embeddings/chunks.json.

Each record: {id, source, url, heading, text, breadcrumb, audience, embedding}.
Only heading and text are embedded; the rest is metadata for filtering and citation.

Requires GEMINI_API_KEY (get one at https://aistudio.google.com/apikey),
either exported in the shell or set in a .env file in the project root.

Usage: python src/embed.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from config import (
    DATA_DIR,
    DOCUMENT_TASK_TYPE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)

BATCH_SIZE = 20
MAX_RETRIES = 5

PROCESSED_DIR = DATA_DIR / "processed"
URL_MAP_PATH = DATA_DIR / "filename-to-url.json"
OUT_PATH = DATA_DIR / "embeddings" / "chunks.json"


def load_records():
    url_map = json.loads(URL_MAP_PATH.read_text(encoding="utf-8"))
    records = []

    for chunk_path in sorted(PROCESSED_DIR.glob("*.json")):
        chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        # A miss here used to yield url=None, which reaches the model as the
        # literal "URL: None" in the prompt while the system prompt is still
        # telling it to cite a source - so a typo in the map degraded into a
        # citation problem at answer time. Refuse to index instead.
        key = f"{chunk_path.stem}.html"
        if key not in url_map:
            raise KeyError(
                f"{key} has no entry in {URL_MAP_PATH.name} - "
                f"every chunk must carry a citable source URL"
            )
        url = url_map[key]

        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{chunk_path.stem}::{i}",
                    "source": chunk_path.stem,
                    "url": url,
                    "heading": chunk["heading"],
                    "text": chunk["text"],
                    # Carried for filtering and citation, never embedded -
                    # see the texts built in main(), which use heading and
                    # text only. Putting metadata in the embedded string
                    # would dilute the vector without helping retrieval.
                    "breadcrumb": chunk.get("breadcrumb", []),
                    "audience": chunk.get("audience"),
                }
            )

    return records


def embed_batch(client, texts):
    # Both halves of the asymmetric scheme, and the dimensionality, come
    # from config.py so the index and the query side cannot drift apart.
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=DOCUMENT_TASK_TYPE,
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            )
            return [e.values for e in response.embeddings]
        except errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = 2**attempt
                print(f"  rate limited, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise


def main():
    load_dotenv(DATA_DIR.parent / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set (shell env or .env file)")

    client = genai.Client()
    records = load_records()
    print(f"embedding {len(records)} chunks...")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        # heading is prepended so the embedding also captures the section
        # breadcrumb, not just the body text
        texts = [f"{r['heading']}\n\n{r['text']}" for r in batch]
        embeddings = embed_batch(client, texts)

        for record, embedding in zip(batch, embeddings):
            record["embedding"] = embedding

        print(f"  {start + len(batch)}/{len(records)}")
        print("Waiting 10 seconds before the next batch to avoid rate limit...")
        time.sleep(10)  # avoid rate limit on next batch

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} embedded chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
