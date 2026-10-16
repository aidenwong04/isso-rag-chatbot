# HW1 agent: bounded relevance classification over the ISSO corpus

This is the HW1 submission for COMS E6998-019 (Fall 2026).
The mechanisms the handout asks for are the same; only the domain changes.

| Handout | Here |
|---|---|
| A pinned target repository (`--input`) | A query set: a folder holding `queries.json` |
| Discovery: Bandit over the `[scan]` scope | Retrieval: the top `k` chunks of the pinned corpus for each query |
| An alert | One (query, chunk) pair; its `alert_id` is `<query id>::<chunk id>` |
| Label 1 / 0 | The chunk is / is not relevant to the query, for the asker (`agent/DESIGN.md`, "Relevance") |
| `.git/`, the answer key, safe twins | Archived pages, embeddings, the answer key, other alerts' conversations (A7) |

The design report notes are in `agent/DESIGN.md`, and the record of how the dev labels were used to refine the agent is in `agent/REFINEMENT.md`.

## Setup from a fresh clone

Tested with Python 3.13.5 on macOS.
Bandit is not used: discovery is retrieval over the corpus, which is committed in the repository (`data/embeddings/chunks.json`).

The command script expects a Python 3.13 environment named `env/` at the repository root.
Create it with whichever tool provides Python 3.13 on your machine:

```bash
git clone https://github.com/aidenwong04/isso-rag-chatbot.git
cd isso-rag-chatbot
python3.13 -m venv env        # or: conda create -p ./env python=3.13
env/bin/pip install -r agent/requirements.txt
```

`agent/requirements.txt` includes the root `requirements.txt`; every direct dependency in both is pinned to an exact version.

The agent reads two environment variables, from the shell or from a `.env` file at the repository root:

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | The classification model (`[model] name` in the config) |
| `GEMINI_API_KEY` | Embedding each query at discovery, with the same embedding model the corpus was built with |

## Running the agent

```bash
./isso-agent --input <query set folder> --output <out dir> --config <config.toml>
```

`./isso-agent` is a shell script at the repository root that runs `agent/src/cli.py` with the `env/` virtualenv.
It writes `report.json` and `trace.jsonl` into `<out dir>`, creating the folder if needed.
A second run into the same folder replaces both files.

The three targets:

| Target | `--input` | Role | Alerts (k = 3) |
|---|---|---|---|
| Dev | `agent/queries/dev` | Development set; its answer key is `agent/labels/dev.worksheet.json` | 30 |
| Held out A | `agent/queries/heldout-a` | Held out, never tuned against | 24 |
| Held out B | `agent/queries/heldout-b` | Held out, never tuned against | 24 |

For example, the six runs (three targets, two configs):

```bash
./isso-agent --input agent/queries/dev --output runs/dev-full --config configs/full.toml
./isso-agent --input agent/queries/dev --output runs/dev-low --config configs/low.toml
./isso-agent --input agent/queries/heldout-a --output runs/heldout-a-full --config configs/full.toml
./isso-agent --input agent/queries/heldout-a --output runs/heldout-a-low --config configs/low.toml
./isso-agent --input agent/queries/heldout-b --output runs/heldout-b-full --config configs/full.toml
./isso-agent --input agent/queries/heldout-b --output runs/heldout-b-low --config configs/low.toml
```

The command prints the run's terminal status and exits 0 for every planned ending (`completed`, `budget_exhausted`, `max_steps`, `no_progress`).
A bad config, a missing `queries.json` or a corpus that does not match the config stops the agent before the run starts, with a message and exit status 1, and writes no files.
A crash during a run still writes both files, with the terminal status `crashed`, and then exits non-zero with the traceback.

## Configs

Two configs are committed:

- `configs/full.toml`: the full budget, enough for a complete run of any of the three targets.
- `configs/low.toml`: `max_tokens = 20000`, which runs out partway through every target, so some rows end `budget_exhausted` (B7).

Both configs share one `[scan]` block, because all three targets are query sets over the same pinned corpus:

| Key | Meaning |
|---|---|
| `[scan] corpus_version` | The first 12 hex digits of the SHA-256 of `data/embeddings/chunks.json`. The agent refuses to start when the corpus on disk differs |
| `[scan] k` | Chunks retrieved per query, so alerts per query |
| `[model] provider`, `name` | `"openai"` and the exact model snapshot |
| `[model] temperature` | Sent on every model call; 0, so two runs of one prompt can be compared |
| `[budget] max_tokens` | The token budget for the whole run: input + output as the provider reports them, checked before every model call |
| `[budget] output_allowance` | The maximum output tokens per model call, sent to the provider as `max_completion_tokens` |
| `[budget] max_steps` | A failsafe on the number of model calls in the run |
| `[budget] no_progress_limit` | Consecutive steps that show no new chunk before the run ends `no_progress` |

## Scoring a dev run

```bash
env/bin/python agent/src/score.py runs/dev-full/report.json agent/labels/dev.worksheet.json
```

It prints the counts of label 1, label 0 and null, then precision, recall and accuracy over the classified rows, and every miss with its rationale.
The scorer is a separate script; the agent never opens the answer key.

## Tests

```bash
env/bin/python -m unittest discover -s agent/tests -v
```

## Recorded runs

Every run cited in the report is committed unedited under `runs/`, with the config it used.

| Folder | Target | Config | Terminal status | used_tokens | Label 1 / 0 / null |
|---|---|---|---|---|---|
| `runs/dev-full-v1` | dev | full (before v2; no temperature set) | completed | 82686 | 20 / 10 / 0 |
| `runs/dev-full-v2` | dev | full | completed | 74319 | 18 / 12 / 0 |
| `runs/dev-full` | dev | full | completed | 74298 | 18 / 12 / 0 |
| `runs/dev-low` | dev | low | budget_exhausted | 20961 | 4 / 4 / 22 |
| `runs/heldout-a-full` | held out A | full | completed | 59935 | 13 / 11 / 0 |
| `runs/heldout-a-low` | held out A | low | budget_exhausted | 21070 | 5 / 3 / 16 |
| `runs/heldout-b-full` | held out B | full | completed | 60384 | 16 / 8 / 0 |
| `runs/heldout-b-low` | held out B | low | budget_exhausted | 21063 | 7 / 1 / 16 |

`runs/dev-full-v1` ran an earlier prompt and is the baseline in `agent/REFINEMENT.md`.
`runs/dev-full` is a rerun of the v2 code and gives the same 30 labels as `runs/dev-full-v2`.

## Layout

```
isso-agent              the command (shell script)
configs/                full.toml, low.toml
agent/DESIGN.md         design notes: behaviours, graph, state, tools, termination, withholding
agent/REFINEMENT.md     how the dev labels were used to refine the agent
agent/queries/          the three targets (one folder each, holding queries.json)
agent/labels/           the dev answer key, outside every target folder
agent/src/
  cli.py                reads and checks the config, builds the workspace, tracer and run context
  run.py                runs the graph once; finalize always runs, after a crash too
  graph.py              the LangGraph state machine
  nodes.py              discovery, next_alert, guard, model, dispatcher, and the routers
  tools.py              read, read_neighbors, classify, and the shared chunk-id check
  workspace.py          the disposable corpus copy the tools read
  tracer.py             trace.jsonl, flushed after every event
  report.py             report rows and finalize
  state.py              the graph state and the run context
  prompt.py             the system prompt and each alert's opening message
  discovery.py          retrieval: query set -> alerts
  score.py              compares a dev report with the answer key
agent/tests/            unit tests for the tools and the workspace copy
runs/                   the recorded runs
```
