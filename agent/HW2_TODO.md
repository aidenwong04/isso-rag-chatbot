# HW2 TODO

Open items collected while building HW2, grouped by where they get done.
Line numbers refer to `briefs/hw2-isso.tex` in the hw2-isso repo.

## Harness (both agents)

- [x] `Backend` verifies the prepared inputs against `manifest.json` when it is created (lazily, on the MCP server's first tool call; directly in the CodeAct parent), and every operation refuses if verification failed (line 86).
- [x] Add `Backend.reset_packets(query_set)` and the host-only `reset_packets` server tool: write an empty packet `{"query_id": ..., "items": []}` for every question in that set with `serialize_packet` (lines 215, 518). The MCP host never opens packet files (line 151).
- [x] Call `reset_packets(<set>)` before the first item: the MCP host through the server tool, the CodeAct parent on its own `Backend`.
- [x] When `set_alert` returns `ok: False`, record that item as an error row with the refusal message and move on; never retry it (line 211).
- [ ] Keep the authoritative record of accepted labels and packets in the parent harness, and publish the report and trace from it after workers stop (lines 305-307).

## MCP host

- [x] Filter the host-only tools (`set_alert`, `reset_packets`) out of the tools offered to the model, and make the dispatcher reject a model request that names either (line 183).
- [x] Classify tool errors for the trace: strip the SDK's `Error executing tool <name>: ` prefix, then `refused:` means `error_kind: refusal` (`refused_by: server`) and `error:` or a bare `Error executing tool <name>` means `error_kind: execution`.
- [x] Read 2.x client fields in snake_case: `input_schema`, `is_error`, `structured_content`.
- [x] Count SDK argument-validation failures (`Error executing tool <name>: ... validation error for <name>Arguments ...`) as refusals too: they carry no `refused:` text but are the server refusing invalid arguments (line 177).

## MCP server

- [ ] Create the `Backend` lazily, on the first tool call, so `initialize` answers promptly (line 154).
- [ ] Resolve the data-root mismatch: `examples/mcp.example.json` passes separate corpus, queries, and manifests roots (staging layout, section 2.2), while `Backend` takes one `prepared_dir`.

## CodeAct harness

- [x] Actions: `run_python(code)` runs in the worker; `classify` and `write_evidence` are separate actions the harness handles, never run as code.
- [x] Give the backend a marker format for printed passages, and build the delivered set from stdout: parsed ID, exact canonical text, and ID in the allowed set (line 137).
- [x] After every `run_python`, diff the packet files against the harness's trusted copy, reject and restore any change the harness did not make (line 280).
- [x] Wipe per-item scratch space (`/tmp` and any scratch folders) between items, keeping only what the harness accepted.
- [ ] Generate `agent/src/hw2/codeact_prompt.md` with Claude Code from `api.py` (brief 3.1): Aiden writes the generation prompt, saves it word for word as a separate file (rubric C5), reviews the generated prompt; it must state the read, write, and network rules (line 259).
- [ ] Run CodeAct on dev with the real model once the prompt exists.

## Injection experiment

- [ ] Overlay `fixtures/injection.json` on the model-visible text of `dev-q05::campus-work-f-and-j-students::0` only: canonical corpus, frozen candidates, and grounding checks stay unchanged (lines 403-406).
- [ ] Enable it only through config (`[experiment] injection_overlay`), record the overlay's hash and whether it was enabled, and use the same overlay for both agents.
- [ ] Prove exposure: record the marker and a reference to the model-visible observation in an actual model request, after any truncation (line 408).
- [ ] Run each agent five times on dev with the overlay and fresh packets; report exposed runs, forbidden read/write attempts, server or kernel refusals, and rejected CodeAct updates, with trace lines (lines 407-410).
- [ ] Never special-case the fixture's marker (line 412); disable the overlay for the clean runs.

## Sandbox image

- [x] Put `agent/prepared/manifest.json` and the prepared files into the image, staged per section 2.2 (line 86).
- [x] Copy `agent/src/hw1/tools.py`, `agent/src/hw1/workspace.py`, and `agent/src/hw2/prepare.py` into `/app`: `api.py` imports them.
- [x] Keep `agent/labels/` and the submitted `agent/REPORT.pdf` (it carries the graders' API key) out of every readable grant (line 112).

- [ ] Create the OpenAI provider (`openshell provider create --name openai --type openai --from-existing`, run by Aiden with the key in the shell) and attach it with `--provider openai`.
- [x] Copy each run's output (report, trace, evidence) out of the sandbox with `openshell sandbox download` and commit it (`sandbox/run.sh`).
- [x] The four required clean runs in the sandbox (brief 6.1): `agent/runs/sandbox/{mcp-dev,mcp-heldout-a,mcp-heldout-b,codeact-dev}`.

## README

- [ ] How to reconstruct the canonical view from the HW1 source with `env/bin/python agent/src/hw2/prepare.py` (line 540).
- [ ] Output retrieval: how the final packets, report, and trace are copied out of the sandbox after a run (line 518). The MCP host cannot open packet files itself (line 151).
- [ ] State that `mcp` 2.3.0 prefixes `ToolError` text with `Error executing tool <name>: `, so refusal text begins with that prefix and then `refused:` (line 178).

## Report

- [ ] Record preparation separately from runtime costs; preparation makes no model or embedding calls (line 96).
- [ ] Cite the HW1 submission commit `f14b7f6` (tag `hw1-submission`) and explain the HW2 changes (line 54).
- [ ] Echo the frozen manifest hash `bd4a06db336003bda25d42ac92707db3ccb1709f4bbe3ebc6e99daea030c5bfd`.
- [ ] Explain which layer enforces what: kernel (fixed paths), MCP server (per-item scope before access), runtime acceptance (CodeAct detects and restores, but a passage already read cannot be unread).
- [ ] Mention the `ToolError` prefix as a known deviation from "text must start with `refused:`".
- [ ] Say how "control metadata" (line 152) was interpreted: the MCP host reads only `prepared/<set>/candidates.json` (alert, query, and chunk IDs, no passage or question text) for its item list.
- [ ] State the step unit and progress condition (line 206): no-progress limit 3, counter resets on progress and on phase change; classify progress is a new observed chunk or a classify proposal not seen before in the item; evidence progress is an accepted quote, which ends the phase, so the evidence phase has at most 3 turns. The step limit and token budget bound a model that keeps varying its proposals.

## Contribution statement

- [ ] List what Claude wrote: in `prepare.py`, serialization, manifest, verification, and `main`; in `api.py`, the `read`, `read_neighbors`, and `classify` wrappers, `read_packet` fixes, the packet and diff building and refusals in `propose_evidence`, and `write_evidence`; the `mcp_server.py` stub; the `mcp` install and pins.
  Aiden designed the behavior these follow and wrote the rest.

## Ask the instructor

- [ ] Is the `mcp` 2.3.0 `ToolError` prefix acceptable for "text must start with `refused:`" (line 178)?

## Optional

- [ ] Make `freeze_candidates` enforce what the data already satisfies: no duplicate `alert_id`, `alert_id == query_id::chunk_id`, every chunk in the view, 3 candidates per question.
