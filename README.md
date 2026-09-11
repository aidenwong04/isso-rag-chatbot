# ISSO RAG Chatbot

A RAG (retrieval-augmented generation) chatbot for Columbia's International Students and Scholars Office (ISSO).

Learning project focused on RAG techniques - chunking, embeddings, retrieval, and LLM integration.
If it works out, may reach out to ISSO about actual use.

## Structure

- `data/` - raw and processed ISSO source content, embeddings, and monitor state
- `docs/` - design and planning documents
- `src/` - RAG pipeline code (ingestion, embedding, retrieval, chat)
- `notebooks/` - exploration and experimentation

## Pipeline

```
config.py      model ids and dimensions, shared by embed and query
one-page.py    data/raw/*.html       -> data/processed/*.json  (parse and chunk)
embed.py       data/processed/*.json -> data/embeddings/chunks.json
query.py       a question            -> retrieved chunks + an answer
monitor.py     the live site         -> drift report vs data/state/baseline.json
```

## Status

Working end to end over a 10-page corpus (182 chunks), embedded with
`gemini-embedding-001` at 768 dimensions, retrieved by brute-force cosine.

`monitor.py` detects when the corpus has drifted from the live site, in four
tiers: sitemap `lastmod`, then a `#main-block` hash, then novelty in the set of
Drupal `paragraph--type--*` component markers, then block-level extraction
coverage. Read-only by default; `--apply` refreshes `data/raw/` and the
baseline. Re-embedding is never automatic.

`query.py` is importable: `retrieve(query, k, audience)` and `answer(query)`
return structured results, and every answer appends a row to
`data/logs/queries.jsonl` recording the full retrieved set with scores, both
model ids, latency and the corpus commit. Pass `log=False` from an eval
harness so benchmark runs stay out of the record of what real users asked.
Query logs are gitignored.

Next: a small web front end over `answer()`, then an eval set built from real
queries. See `docs/evaluation-plan.md`.
