# HW2 agents: ISSO evidence selection with MCP and CodeAct

This is the HW2 submission for COMS E6998-019 (Fall 2026), the ISSO variant of the brief (`briefs/hw2-isso.tex` in the course's `hw2-isso` repository).
It extends the HW1 relevance agent (commit `f14b7f6`, tag `hw1-submission`) into two agents that share one backend and one host loop:

- the **MCP agent**, whose model calls tools on a stdio MCP server (`agent/src/hw2/mcp_server.py`);
- the **CodeAct agent**, whose model writes Python that calls the backend directly (`agent/src/hw2/codeact.py`).

For each frozen question-passage pair, an agent labels the passage 1 (relevant) or 0, and after an accepted label 1 adds one quote from that passage to the question's evidence packet.
Both agents run inside an OpenShell sandbox.

The design report is [`agent/REPORT.md`](REPORT.md).
The HW1 instructions are still in [`agent/README.md`](README.md).

## Versions

| Component | Version |
|---|---|
| Python (local) | 3.13.5 |
| Python (image, `python:3.13-slim`) | 3.13.18 |
| `mcp` | 2.3.0 |
| `langgraph` | 1.2.15 |
| `openai` | 3.31.0 |
| OpenShell | 0.1.4 |
| Colima | 0.10.3 (Docker 29.7.2), on Apple Silicon |
| Model | `gpt-4.1-mini-2025-04-14`, temperature 0 |

Every Python package, dependencies included, is pinned in `agent/requirements.lock`; the image installs exactly that file.

The only environment variable is `OPENAI_API_KEY`.
Locally it is read from the shell or from `.env` at the repository root.
In the sandbox it is never present: the OpenShell provider holds it outside and injects it into requests to `api.openai.com` (see [Provider](#provider)).
Preparation and the agents make no embedding calls, so HW1's `GEMINI_API_KEY` is not needed.

## 1. Fresh clone and dependencies

```bash
git clone https://github.com/aidenwong04/isso-rag-chatbot.git
cd isso-rag-chatbot
git checkout hw2
python3.13 -m venv env
env/bin/pip install --no-deps -r agent/requirements.lock
```

The launcher `agent/isso-agent-hw2` expects this `env/` folder at the repository root.
Run every command in this README from the repository root.

## 2. Frozen inputs

The frozen inputs are committed in `agent/prepared/`, and the agents never regenerate them.

| File | Content |
|---|---|
| `chunks.json` | The canonical view: every chunk's id, text, heading, breadcrumb, audience, source and URL, without embeddings |
| `neighbors.json` | Each chunk's immediate previous and next chunk on the same page |
| `<set>/queries.json` | The questions of `dev`, `heldout-a`, `heldout-b` |
| `<set>/candidates.json` | The 3 frozen candidate passages per question (`k = 3`), with their `alert_id` (`<question id>::<page>::<index>`) |
| `manifest.json` | The SHA-256 of every file above and of the source corpus |

The source corpus is HW1's `data/embeddings/chunks.json`, pinned at SHA-256 `f186563d6a1e702703386a25e5047e44eac3d666a7fe382d516c86dd47e5bf8e` (`[scan] corpus_version = "f186563d6a1e"` in the configs).
The frozen manifest's own SHA-256 is `bd4a06db336003bda25d42ac92707db3ccb1709f4bbe3ebc6e99daea030c5bfd`.

To reconstruct the view from the HW1 source and check it against the committed one:

```bash
env/bin/python agent/src/hw2/prepare.py
git status --short agent/prepared
```

`prepare.py` checks the corpus pin, rewrites `agent/prepared/`, verifies it against the new manifest, and prints the manifest hash, which must be `bd4a06db…` above.
An empty `git status` means the rebuilt files are byte-identical to the committed ones.
It makes no model or embedding calls: questions are copied byte for byte from `agent/queries/<set>/queries.json`, and candidates are the rows of the submitted HW1 full runs (`agent/runs/<set>-full/report.json`), never retrieved again.

The backend verifies `agent/prepared/` against `manifest.json` every time it starts, and refuses every operation if any hash differs.

### Staging in the image

The brief's section 2.2 stages separate corpus, queries and manifest roots.
Here the prepared files stay together in one folder, so the server and backend take one `--prepared-dir`.

| Path in the image | Content | Policy |
|---|---|---|
| `/app/` | Code, venv, schemas, configs, `mcp.json`, `policy.yaml`, the injection overlay | read-only |
| `/target/prepared/` | `agent/prepared/` | read-only |
| `/target/labels/` | The HW1 dev answer key | no grant |
| `/target/private/chunks.json` | The original corpus with embeddings | no grant |
| `/sandbox/work/` | `evidence/` (packets), `run/` (report and trace), `scratch/` (CodeAct) | read-write |

The withheld files exist in the image and are readable by every user, so a denied read inside the sandbox comes from the Landlock policy, not from file permissions or a missing file.

## 3. Backend, prompts, and how CodeAct runs code

The backend is the `Backend` class in [`agent/src/hw2/api.py`](src/hw2/api.py).
The MCP server wraps it, and CodeAct scripts import it.

| File | Role |
|---|---|
| `agent/src/hw2/prompt.py` | The MCP agent's system prompt and per-item messages |
| `agent/src/hw2/codeact_prompt.md` | The CodeAct agent's system prompt, generated with Claude Code from `api.py` and reviewed |
| `agent/src/hw2/codeact_generation_prompt.md` | The exact prompt used to generate `codeact_prompt.md` (rubric C5) |

The CodeAct model has three actions: `run_python(code)`, `classify(label, rationale, evidence)`, and `write_evidence(quote)`.
`classify` and `write_evidence` are handled by the harness, never run as code.
Each `run_python` runs in a fresh process of the agent's own Python (`/app/.venv/bin/python` in the image), in a per-item scratch folder under `/sandbox/work/scratch` that is deleted between items, with a minimal environment (no key), a 60 second timeout, and the first 20,000 characters of output shown to the model.
A script starts with `from api import Backend`, `backend = Backend()`, and `backend.set_alert(<alert id>, <phase>)`.
`backend.show(id)` and `backend.show_neighbors(id)` print passages between `<<<chunk ID>>>` and `<<<end ID>>>` markers.
A passage counts as delivered, and so may be cited, only if its exact text appears inside markers in output the model saw and its id is in the item's allowed read set.
After every script the harness compares the packet files with its trusted copy and restores any change it did not make itself.

## 4. Tool contracts and trusted controls

Each item's allowed read set is its question's 3 frozen candidates plus their frozen immediate neighbors.
It never grows during an item.

| Tool | Arguments | Result | Who may call it |
|---|---|---|---|
| `set_alert` | `alert_id: str`, `phase: "classify" \| "evidence"` | `msg`, `alert_id`, `query_id`, `chunk_id`, `phase` | host only |
| `reset_packets` | `query_set: str` | `msg`, `query_set`, `query_ids` | host only |
| `current_item` | none | `question`, `audience`, `chunk_id`, `candidate_ids` | model |
| `read` | `chunk_id: str` | `msg`, `chunks` (one chunk) | model (read) |
| `read_neighbors` | `chunk_id: str` | `msg`, `chunks` (allowed neighbors; `msg` says how many were left out) | model (read) |
| `classify` | `label: int` (0 or 1, strict), `rationale: str` (at most 600 characters), `evidence: list[str]` | `msg` | model, classify phase |
| `read_packet` | none | `packet` (`query_id`, `items`) | model, evidence phase (read) |
| `write_evidence` | `quote: str` (exact text of the target passage, at most 600 characters) | `msg`, `item`, `diff`, `checks` (`applies`, `current_item_only`, `grounded`) | model, evidence phase (write) |

The read operations are `read`, `read_neighbors` and `read_packet`; the only write operation is `write_evidence`.
`write_evidence` builds the item itself (`chunk_id`, `source` and `url` come from the current item), writes at most one item per passage, and writes atomically to `<evidence dir>/<question id>.json`.
Every result schema is a `TypedDict` the server publishes through `tools/list`, and the SDK validates arguments against the input schemas.

**Refusals.** The backend refuses with `{"ok": false, "msg": "refused: ..."}`, and the server raises that message as a `ToolError`.
`mcp` 2.3.0 adds its own prefix, so the text the client receives is `Error executing tool <name>: refused: ...`.
The host strips that prefix and traces the result as `error_kind: "refusal"`, `refused_by: "server"`.
Argument-validation failures from the SDK also count as server refusals; any other error is `error_kind: "execution"`.

**Trusted controls.** `set_alert` and `reset_packets` are left out of the tools offered to the model, and the dispatcher refuses a model request that names either one (`refused_by: "dispatcher"`), as it refuses unknown tools.
The host selects each item and phase, and only an accepted label 1 starts the evidence phase.

## 5. Image and server

Install OpenShell 0.1.4 and Colima as the course handout describes for Apple Silicon.
This setup used `~/.config/openshell/gateway.toml` with the Docker compute driver at `~/.colima/default/docker.sock`.

```bash
colima start --cpu 4 --memory 6
docker build -f sandbox/Dockerfile -t isso-hw2:1 .
```

`sandbox/Dockerfile.dockerignore` allow-lists the build context, so `.env` and other secrets never enter it.

### Provider

```bash
openshell provider profile import -f sandbox/openai.yaml --global
openshell provider create --name openai --type openai --from-existing
```

Run the second command in a shell where `OPENAI_API_KEY` is set to the key itself, without quotes.
`sandbox/openai.yaml` is OpenShell's OpenAI profile with `binaries` limited to the image's Python.

### Server discovery

The host starts the server from [`mcp.json`](../mcp.json), named by `[mcp] config` in the config:

```json
"command": "/app/.venv/bin/python",
"args": ["/app/agent/src/hw2/mcp_server.py", "--prepared-dir", "/target/prepared", "--evidence-dir", "/sandbox/work/evidence"]
```

It runs over stdio as a child of the agent, inside the same sandbox.
The host lists the tools with `tools/list`, records the list in the trace, and offers the model every tool except the two host-only controls.
`mcp.local.json` has the same server with local paths, for runs outside the sandbox.

## 6. Running the agents

The agent command is the same for both agents; `[agent] kind` in the config selects one:

```bash
agent/isso-agent-hw2 --input <prepared set> --output <out dir> --config <config.toml>
```

| Config | Agent | Overlay |
|---|---|---|
| `configs/mcp.toml` | MCP | off |
| `configs/codeact.toml` | CodeAct | off |
| `configs/mcp.injection.toml` | MCP | `fixtures/injection.json` |
| `configs/codeact.injection.toml` | CodeAct | `fixtures/injection.json` |

The injection configs are identical to the clean ones plus `[experiment] injection_overlay`.
`[scan]`, `[model]` and `[budget]` are identical in all four.
Configs are validated against `agent/schemas/config.schema.json` before the run starts.

### In the sandbox

`sandbox/run.sh` does one run in a fresh sandbox: it creates the sandbox with `policy.yaml` and the `openai` provider, runs the agent, downloads the output, and deletes the sandbox.

```bash
sandbox/run.sh mcp dev mcp-dev
sandbox/run.sh mcp heldout-a mcp-heldout-a
sandbox/run.sh mcp heldout-b mcp-heldout-b
sandbox/run.sh codeact dev codeact-dev
sandbox/run.sh mcp dev mcp-injection-1 mcp.injection.toml
sandbox/run.sh codeact dev codeact-injection-1 codeact.injection.toml
```

The same run by hand, which is what `run.sh` does:

```bash
openshell sandbox create --name hw2-run --from isso-hw2:1 --policy policy.yaml --provider openai --detach -- sleep infinity
openshell sandbox exec -n hw2-run --no-login-shell --no-tty -- /app/agent/isso-agent-hw2 --input /target/prepared/dev --output /sandbox/work/run --config /app/configs/mcp.toml
openshell sandbox download hw2-run /sandbox/work/run agent/runs/sandbox/hw2-run
openshell sandbox download hw2-run /sandbox/work/evidence agent/runs/sandbox/hw2-run/evidence
openshell sandbox delete hw2-run
```

**Packet reset.** Every run starts by calling `reset_packets(<set>)`, which writes an empty packet for each of the set's questions, so a run never inherits packets from an earlier run, even in a reused sandbox.

**Output retrieval.** The agent writes `report.json` and `trace.jsonl` to `--output` (`/sandbox/work/run`) and the packets to `/sandbox/work/evidence`.
`openshell sandbox download` copies both out; `run.sh` puts them in `agent/runs/sandbox/<run name>/` and `agent/runs/sandbox/<run name>/evidence/`.
The MCP host itself never opens packet files: only the server reads and writes them.

### Locally, outside the sandbox

For development only; local runs have no kernel policy.

```bash
agent/isso-agent-hw2 --input agent/prepared/dev --output agent/runs/hw2-mcp-dev --config configs/mcp.toml --mcp-json mcp.local.json
agent/isso-agent-hw2 --input agent/prepared/dev --output agent/runs/hw2-codeact-dev --config configs/codeact.local.toml
```

## 7. Recorded runs

Every run cited in the report is committed unedited, with its report, trace, and final packets.

| Folder | Agent | Set | Where |
|---|---|---|---|
| `agent/runs/sandbox/mcp-dev` | MCP | dev | sandbox |
| `agent/runs/sandbox/mcp-heldout-a` | MCP | heldout-a | sandbox |
| `agent/runs/sandbox/mcp-heldout-b` | MCP | heldout-b | sandbox |
| `agent/runs/sandbox/codeact-dev` | CodeAct | dev | sandbox |
| `agent/runs/sandbox/mcp-injection-{1..5}` | MCP | dev, overlay on | sandbox |
| `agent/runs/sandbox/codeact-injection-{1..5}` | CodeAct | dev, overlay on | sandbox |
| `agent/runs/hw2-mcp-dev`, `agent/runs/hw2-codeact-dev-v1`, `agent/runs/hw2-codeact-dev` | both | dev | local, during development |

The injected runs record the overlay's path and SHA-256 in the trace and in the report's `experiment` field.

## 8. Tests

```bash
env/bin/python -m unittest discover -s agent/tests -v
```

## Layout

```
agent/isso-agent-hw2          the command for both agents
agent/src/hw2/
  prepare.py                  builds and verifies agent/prepared/
  api.py                      the Backend both agents use
  mcp_server.py               the MCP server over the Backend
  agents.py                   the MCP agent and the shared controller interface
  codeact.py                  the CodeAct agent: worker processes, delivered set, packet restore
  graph.py, nodes.py          the shared LangGraph host loop
  state.py, report.py, run.py run state, report rows, the run itself
  cli.py                      config checks and agent selection
  prompt.py                   the MCP prompt
  codeact_prompt.md           the CodeAct prompt
  codeact_generation_prompt.md  the prompt that generated it
agent/prepared/               the frozen inputs
agent/schemas/                config, report, and trace schemas
configs/                      the four configs
mcp.json, mcp.local.json      server definitions (sandbox, local)
policy.yaml                   the OpenShell policy
sandbox/                      Dockerfile, provider profile, run.sh
fixtures/injection.json       the injection overlay
```
