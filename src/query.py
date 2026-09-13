"""Retrieve chunks for a question and answer from them.

Importable rather than script-shaped, because three different callers need
the same retrieval: an eval harness computing Recall@k, a web app answering
a user, and a corpus comparison re-running the same queries against different
indexes. Every one of those needs to call retrieval without a terminal
attached.

    from query import answer, retrieve

    result = answer("what do i need for my visa interview")
    result.text                 # the answer
    result.chunks               # what it was built from, with scores
    result.to_log_record()      # pipeline fields for the caller to log

answer() does not write anything. The caller decides whether and where to
log, because only the caller knows what else belongs on the row (the web app
adds end-to-end latency) and whether the query is real traffic at all (an
eval run is not).

The index loads once per process and is reused, so a server does not re-parse
182 x 768 floats on every request. Call load_index() once at startup to pay
that cost before the first request rather than during it.

Imports here are flat (`from config import ...`), so a caller needs src/ on
the path - simplest is to put the app in src/ alongside this and run it from
there.

CLI (prints, never logs):  python src/query.py
Also accepts the question as arguments:  python src/query.py how do i report my arrival
"""

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    QUERY_TASK_TYPE,
)

EMBEDDING_PATH = DATA_DIR / "embeddings" / "chunks.json"

SYSTEM_PROMPT = """You are a helpful assistant answering questions from international students about Columbia's International Students and Scholars Office (ISSO), using only the ISSO source material provided below.

How to write:
- Use ONLY the source material below. Do not use outside knowledge and do not guess.
- Write to the student directly, as "you", in plain language. Lead with the answer: no preamble, no apologising, no restating the question.
- The student cannot see the source material and does not know it exists. Write as if you simply know this, never describing where the information came from.

When the sources answer the question:
- Answer it, then end with a "More info:" line.

When the sources do not answer the question:
- Open with "I don't have information about ..." naming what was asked.
- If something genuinely related is available, add one sentence in the form "ISSO does have guidance on ..." and end with a "More info:" line pointing to it. Describe what ISSO covers, never what the material in front of you contains.
- Only do that when the related topic would actually help with what was asked. If the nearest thing is a stretch, do not reach for it - say nothing further is available here and suggest emailing ISSO.
- Never present related material as though it answers the question that was asked.

The "More info:" line:
- Format: `More info: <Heading> (<URL>)`
- The Heading and the URL must be copied exactly from ONE numbered source above, and must be that same source's own Heading and URL.
- Never build a citation out of a page name, link text or document title mentioned inside a source's Text. Those have no URL here, and pairing one with a different source's URL sends the student to the wrong page.
- Cite only sources you actually used. If you used none, write no "More info:" line at all.

Never invent a URL, heading, or fact that is not in the sources. This is immigration guidance, and an answer that sounds right but is wrong can cause a student real harm."""

CHUNK_TEMPLATE = """[{n}] Source: {source}
Heading: {heading}
URL: {url}
Text: {text}"""

# Phrases the system prompt steers the model toward when the chunks do not
# answer the question. Matching them is a triage hint for spotting the
# abstention rate at a glance - it is NOT a label. Whether an abstention was
# correct is a judgment about the corpus, and belongs in hand-labelled eval
# data, not in a substring check.
ABSTENTION_HINTS = (
    "do not contain",
    "does not contain",
    "don't contain",
    "doesn't contain",
    "not enough information",
    "no information",
    "i don't have information",
    "i do not have information",
    "i don't have any information",
    "i do not have any information",
    "i don't have any material"
)

_index_cache = None
_client_cache = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    source: str
    url: str
    heading: str
    text: str
    audience: str | None = None

    def to_log_record(self):
        # text is deliberately omitted: it is reproducible from chunk_id plus
        # the corpus commit, and repeating it on every row would bloat the log
        # for no recoverable information.
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "audience": self.audience,
        }


@dataclass
class AnswerResult:
    query: str
    text: str
    chunks: list = field(default_factory=list)
    session_id: str = ""
    retrieval_ms: int | None = None
    generation_ms: int | None = None
    error: str | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_thought_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def looks_like_abstention(self):
        lowered = self.text.lower()
        return any(hint in lowered for hint in ABSTENTION_HINTS)

    def to_log_record(self):
        """The pipeline's half of a log row.

        No timestamp: the caller stamps the row with when the request
        arrived, which answer() cannot see.

        The whole retrieved set is recorded with scores, not just the top
        hit. Recall@k cannot be computed from a log that only kept the winner.
        """
        return {
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.text,
            "looks_like_abstention": self.looks_like_abstention,
            "retrieved": [chunk.to_log_record() for chunk in self.chunks],
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "generation_model": GENERATION_MODEL,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_thought_tokens": self.total_thought_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_tokens,
            "corpus_commit": corpus_commit(),
            "corpus_chunks": len(load_index()[0]),
            "error": self.error,
        }


def corpus_commit():
    """Short git commit of the working tree, or None outside a repo.

    A logged interaction that cannot be tied back to the corpus that produced
    it is not usable as evidence later.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=DATA_DIR.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def get_client():
    global _client_cache
    if _client_cache is None:
        load_dotenv(DATA_DIR.parent / ".env")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set (shell env or .env file)")
        _client_cache = genai.Client(api_key=api_key)
    return _client_cache


def load_index():
    """Return (records, embedding matrix), loaded once per process."""
    global _index_cache
    if _index_cache is None:
        records = json.loads(EMBEDDING_PATH.read_text(encoding="utf-8"))
        matrix = np.array([record["embedding"] for record in records])
        _index_cache = (records, matrix)
    return _index_cache


def embed_query(text):
    response = get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type=QUERY_TASK_TYPE,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return np.array(response.embeddings[0].values)


def retrieve(query, k=DEFAULT_TOP_K, audience=None):
    """Top-k chunks by cosine similarity, highest first.

    `audience` filters candidates *before* ranking, so the result is the best
    of the right chunks rather than a hope that the right ones outrank the
    wrong ones. Chunks whose page states no audience are always kept: the
    absence of a stated audience is not evidence of a mismatch. Nothing calls
    this with an audience yet - which corpus a version serves is a per-version
    decision, and this is only the hook.
    """
    records, matrix = load_index()
    query_vector = embed_query(query)

    scores = np.dot(query_vector, matrix.T) / (
        np.linalg.norm(query_vector) * np.linalg.norm(matrix, axis=1)
    )

    candidates = range(len(records))
    if audience is not None:
        candidates = [
            i
            for i in candidates
            if records[i].get("audience") in (audience, None)
        ]
        if not candidates:
            return []

    ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
    return [
        RetrievedChunk(
            chunk_id=records[i]["id"],
            score=float(scores[i]),
            source=records[i]["source"],
            url=records[i]["url"],
            heading=records[i]["heading"],
            text=records[i]["text"],
            audience=records[i].get("audience"),
        )
        for i in ranked
    ]


def build_prompt(query, chunks):
    """Assemble the context block for however many chunks were retrieved.

    Built by joining per-chunk blocks rather than formatting a fixed
    three-slot template, so k is a parameter rather than a constant. Recall@10
    is not measurable against a prompt that can only hold three.
    """
    blocks = [
        CHUNK_TEMPLATE.format(
            n=n,
            source=chunk.source,
            heading=chunk.heading,
            url=chunk.url,
            text=chunk.text,
        )
        for n, chunk in enumerate(chunks, start=1)
    ]
    return "Context chunks:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"


def answer(query, k=DEFAULT_TOP_K, audience=None, session_id=None):
    """Retrieve and generate. Returns an AnswerResult and logs nothing.

    Never raises for an upstream failure: the error is kept on the result
    so the caller's log row records it.
    """
    result = AnswerResult(
        query=query,
        text="",
        session_id=session_id or uuid.uuid4().hex[:12],
    )

    # Both stages call Gemini, so both are guarded the same way. Embedding the
    # question is an upstream call too: leaving it outside the guard turned an
    # outage (or a request Gemini rejects, like an empty string) into an
    # uncaught exception with no log row, exactly when the row matters most.
    started = time.perf_counter()
    try:
        result.chunks = retrieve(query, k=k, audience=audience)
    except Exception as error:  # noqa: BLE001 - surface, log, do not crash
        record_failure(result, "retrieval", error)
    result.retrieval_ms = int((time.perf_counter() - started) * 1000)

    if result.error is not None:
        pass
    elif not result.chunks:
        result.text = "I don't have any material to answer that from."
    else:
        started = time.perf_counter()
        try:
            interaction = get_client().interactions.create(
                model=GENERATION_MODEL,
                input=SYSTEM_PROMPT + build_prompt(query, result.chunks),
            )
            result.text = interaction.output_text
            if interaction.usage is not None:
                result.total_input_tokens = interaction.usage.total_input_tokens
                result.total_output_tokens = interaction.usage.total_output_tokens
                result.total_thought_tokens = interaction.usage.total_thought_tokens
                result.total_cached_tokens = interaction.usage.total_cached_tokens
                result.total_tokens = interaction.usage.total_tokens
        except Exception as error:  # noqa: BLE001 - surface, log, do not crash
            record_failure(result, "generation", error)
        result.generation_ms = int((time.perf_counter() - started) * 1000)

    return result


def record_failure(result, stage, error):
    """Mark an AnswerResult as failed at `stage`, keeping the cause for the log.

    A user-facing process should not die because one upstream call failed.
    The stage prefix says which call it was, since the student sees the same
    message either way.
    """
    result.error = f"{stage}: {type(error).__name__}: {error}"
    result.text = "Something went wrong reaching the model. Please try again."


def main():
    query = " ".join(sys.argv[1:]) or input("Enter your prompt: ")
    result = answer(query)

    summary = ", ".join(
        f"{chunk.heading} ({chunk.score:.2f})" for chunk in result.chunks
    )
    print(f"\nTop {len(result.chunks)} chunks for '{query}': {summary}\n")
    print(result.text)

    # None means the stage never ran, which is different from a stage that
    # took 0ms, so say so rather than printing a number.
    timings = [
        f"{ms}ms {stage}" if ms is not None else f"{stage} skipped"
        for stage, ms in (
            ("retrieval", result.retrieval_ms),
            ("generation", result.generation_ms),
        )
    ]
    print(f"\n[{', '.join(timings)}]")


if __name__ == "__main__":
    main()
