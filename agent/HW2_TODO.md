# HW2 TODO

Open items from brief section 2.1 that depend on later work.
Line numbers refer to `briefs/hw2-isso.tex` in the hw2-isso repo.

- [ ] Call `verify_prepared(agent/prepared/manifest.json)` at harness startup and refuse to run if it fails (line 86).
- [ ] Put `agent/prepared/manifest.json` and the prepared files into the sandbox image, staged per section 2.2 (line 86).
- [ ] README: how to reconstruct the canonical view from the HW1 source with `env/bin/python agent/src/hw2/prepare.py` (line 540).
- [ ] README/report: record preparation separately from runtime costs; preparation makes no model or embedding calls (line 96).
- [ ] Report: cite the HW1 submission commit `f14b7f6` (tag `hw1-submission`) and explain the HW2 changes (line 54).
- [ ] Report: echo the frozen manifest hash `bd4a06db336003bda25d42ac92707db3ccb1709f4bbe3ebc6e99daea030c5bfd`.

## Optional

- [ ] Make `freeze_candidates` enforce what the data already satisfies: no duplicate `alert_id`, `alert_id == query_id::chunk_id`, every chunk in the view, 3 candidates per question.
