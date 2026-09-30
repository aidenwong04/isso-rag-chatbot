# ISSO RAG Chatbot

A RAG (retrieval-augmented generation) chatbot for Columbia's International Students and Scholars Office (ISSO).

Learning project focused on RAG techniques - chunking, embeddings, retrieval, and LLM integration.
If it works out, may reach out to ISSO about actual use.

## Structure

- `data/` - raw and processed ISSO source content, embeddings, and monitor state
- `docs/` - design and planning documents
- `src/` - RAG pipeline code (ingestion, embedding, retrieval, chat)
- `notebooks/` - exploration and experimentation
- `web/` - static front end for `/chat`, deployed to Vercel
- `agent/` - coursework agents built on the corpus; not part of the deployed API

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
return structured results, and each result carries a log record with the
model ids, latency, token spend and the corpus commit. `answer()` itself logs
nothing: `POST /chat` in `main.py` prints one JSON row per answered request to
stdout when `LOGGING_ENABLED` is on, so eval runs and CLI use stay out of the
record of what real users asked. Refused requests (bad input, rate limited)
are always logged, without the message text. On Cloud Run, stdout lands in
Cloud Logging.

`Dockerfile` builds the `/chat` API for Cloud Run, with `chunks.json` baked
into the image so code and corpus deploy and roll back together.

Next: a small web front end over `answer()`, then an eval set built from real
queries. See `docs/evaluation-plan.md`.

## Deploys

The API deploys to Cloud Run from `.github/workflows/deploy-api.yml` on every push to `main` that touches `src/`, `Dockerfile`, `requirements.txt` or `data/embeddings/chunks.json`.
Each new revision starts with no traffic, is smoke-tested at its own URL (health check, a rejected empty message, one real question), and takes traffic only if all three pass.
GitHub authenticates through Workload Identity Federation, so no service account key exists.

The front end in `web/` is deployed separately with `npx vercel deploy --prod` from that folder.
The API only accepts browser requests from the origins in the service's `ALLOWED_ORIGINS` env var.
