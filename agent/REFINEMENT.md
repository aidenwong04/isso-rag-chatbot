# Refinement log (dev set)

How the dev labels were used to refine the agent.
This is the raw material for C2 in REPORT.md.
Each entry names the run, the code it ran, its scores, and what the misses showed.

Scores come from `agent/src/score.py <report.json> agent/labels/dev.worksheet.json`.
Precision and recall are over classified rows only.

## v1 baseline: `runs/dev-full/`

- Code: commit `fdccb6b` (the opening message gives only the chunk id; the shown set starts empty).
- Config: `configs/full.toml` (gpt-4.1-mini-2025-04-14, k = 3, max_tokens 600000, temperature not set, so the API default of 1).
- Terminal status `completed`; used_tokens 82686 (input 78900, output 3786, cache_read 16128).
- Tool calls: 38 `read`, 30 `classify`, 0 `read_neighbors`.

| | |
|---|---|
| Label 1 / 0 / null | 20 / 10 / 0 (30 of 30 classified) |
| tp / fp / fn / tn | 15 / 5 / 0 / 10 |
| Precision | 0.750 |
| Recall | 1.000 |
| Accuracy | 0.833 |

All five misses are false positives: the agent never missed a relevant chunk, it was too generous.

| Alert | Agent | Key |
|---|---|---|
| `dev-q05::campus-work-f-and-j-students::0` | 1 | 0 |
| `dev-q04::extending-begin-new-program-f-j-students::6` | 1 | 0 |
| `dev-q02::f-1-cpt-curricular-practical-training::14` | 1 | 0 |
| `dev-q03::incoming-student-programs::9` | 1 | 0 |
| `dev-q08::reporting-your-arrival-students::1` | 1 | 0 |

### Finding 1: the agent judged the page, not the chunk (`dev-q05::campus-work-f-and-j-students::0`)

The chunk is a three-line outline of the page ("Overview for both F and J students", "J-1 students only: Mandatory Authorization", "After you find an on-campus job").
The rationale describes hour limits, affiliated schools and mandatory authorization, none of which is in the chunk.

This is not a hallucination.
The trace (`runs/dev-full/trace.jsonl`, steps 182 to 244) shows the model read the passage, then guessed the `page::index` id pattern and read chunks 1 to 8 of the same page one at a time with `read` (not `read_neighbors`), 10 model calls in all.
It then classified 1 with evidence `[::1, ::6, ::7]`: the passage itself is not in its own evidence.
Every read grew the shown set, so the no-progress detector never fired.

The system worked as designed; the model read a rule in a way that was not intended.
The prompt says to judge the chunk in its page context and to read the chunks around it when its meaning depends on them, and the model took "context" to mean that the page's content counts.
The `classify` contract checks that evidence is in the shown set, but not that it includes the passage being judged.

Open design questions this raises:

1. What is page context for in the relevance definition?
   To interpret the chunk's own words (who "you" is, what "this form" refers to), or also to supply answer content the chunk does not contain?
   The hand label (0) says the first; the prompt needs one sentence that cannot be read the second way.
2. Should `classify` require the alert's own chunk in `evidence`?
   For every label, or only for label 1?
   (A 0 for an audience mismatch may rest on the page header rather than on the chunk.)
3. Should `tool_result` events record the returned chunk ids (not the text)?
   This miss was diagnosable only because the `tool_request` args happened to carry the ids.

### Finding 2: the same chunk got different labels in two runs

`dev-q05::campus-work-f-and-j-students::0` was labeled 0 in an earlier low-budget run on the same day ("only provides a brief outline of topics") and 1 in this run, with the same prompt and chunk.
Temperature is not set, so the API default of 1 applies and the model samples.

Consequences for comparing prompt versions on 30 alerts:

- One flipped label moves precision by several points, so a small difference between two runs may be noise.
- Before trusting a change, either fix the sampling (set temperature in the config and the model call) or run each version more than once and compare the spread.

### Finding 3: right audience, wrong situation (`dev-q04::extending-begin-new-program-f-j-students::6`)

The question asks what to do with an I-20 when transferring schools.
The chunk ("Review, sign, and date your new I-20/DS-2019") gives the start date of a new I-20 and travel instructions for it.
It is from the page on extending or beginning a new program at Columbia, not on transferring to another school.
The key labels it 0: it is about the new I-20 after a change of program, not the transfer process.

The model read only the passage (trace steps 157 to 163) and cited it, so the rationale matches the chunk; unlike Finding 1, nothing came from elsewhere.
It took "new I-20" and "new program" to be the transfer scenario.
The asker's audience matches (an F-1 student either way); their situation does not.

Decision: no rule about this scenario in the prompt.
That would tune the agent to one dev alert, which the held-out sets exist to catch.
Instead, a general working habit for v2: before labeling 1, check that the page the chunk comes from is about the situation in the question (from its page id, heading and breadcrumb, or by reading around the chunk).

This pulls the other way from Finding 1, where reading the rest of the page led to a wrong 1.
The two fit together only if context is used to understand which situation the chunk is about, never as answer content in place of the chunk's own text.
The v2 prompt has to say both.

## v2 changes

- `classify` refuses evidence that does not include the passage being judged (Finding 1).
  Other chunks may support a label but not replace the passage.
- `[model] temperature = 0` in both configs, sent on every model call (Finding 2), so runs of one prompt can be compared.
- Relevance is judged as the chatbot's generator sees a chunk: its text, heading and page.
  `src/query.py` sends exactly source, heading, URL and text per chunk, with no neighbours.
  Context may explain what the chunk means, never add to what it says.
  Before labeling 1, the model checks that the chunk's page is about the asker's situation (Finding 3).
  The prompt and the relevance definition in DESIGN.md say this.
- `read_neighbors` stays, for interpreting a chunk's own words (a referent, a list cut off from its lead-in), not for answer content.
- Considered and not built: a `page_outline(chunk_id)` tool returning the page id, URL, breadcrumb, audience and section headings, with no chunk text.
  It would show what a page is about without supplying answer content, and would replace `read_neighbors` under the generator's-view definition.
  Left out for time.

### Still to check

- For the other four misses: which relevance rule should have made each one 0 ("related topic but does not answer", "pointer", or another), and whether the model read beyond the passage for them too.
- The rationales keep saying "addresses" and "relates to"; which word in the rules is being read too loosely.

## v2 run: `runs/dev-full-v2/`

- Code: commit `dbf647f` (the v2 changes above).
- Config: `configs/full.toml`, now with `temperature = 0`.
- Terminal status `completed`; used_tokens 74319 (input 70818, output 3501, cache_read 2176), against 82686 for v1.
- Tool calls: 30 `read`, 30 `classify`, 0 `read_neighbors`, no refusals; every alert took exactly two model calls (read the passage, then classify).

| | v1 | v2 |
|---|---|---|
| Label 1 / 0 / null | 20 / 10 / 0 | 18 / 12 / 0 |
| tp / fp / fn / tn | 15 / 5 / 0 / 10 | 15 / 3 / 0 / 12 |
| Precision | 0.750 | 0.833 |
| Recall | 1.000 | 1.000 |
| Accuracy | 0.833 | 0.900 |

Fixed by v2:

- `dev-q05::campus-work-f-and-j-students::0` (Finding 1) is now 0: "only provides an overview of the sections on the page".
  The model read only the passage; it no longer reads the rest of the page.
- `dev-q02::f-1-cpt-curricular-practical-training::14` is now 0: the chunk says guidance may change but "does not provide any specific" rule.

Still missed (all false positives):

- `dev-q04::extending-begin-new-program-f-j-students::6` (Finding 3).
  The situation check did not take: the rationale says beginning a new program "includes transferring schools".
  The page id names the situation, but the model did not treat it as different from the asker's.
  As decided, no scenario-specific rule was added for it.
- `dev-q03::incoming-student-programs::9`: the chunk says incoming students can take part in the events on the calendar; the question asks whether there are upcoming social events.
  The key note calls it a gray area and a pointer to where the information is.
- `dev-q08::reporting-your-arrival-students::1`: the chunk ("Who needs to report") lists who must report their arrival; the question asks why and how.
  The model read "who must report" as answering "why".

Caveat on the comparison: v1 ran at the API's default temperature and v2 at 0, so part of the difference may be sampling rather than the prompt.
The two misses v2 fixed are explained by the changes themselves (the model stopped reading the page, and rationales now point to what the chunk lacks), which is stronger evidence than the score alone.

The two gray-area misses (q03 and q08) are judgment calls the labels made before any agent run.
Relabeling them now, after seeing the agent disagree, would tune the key to the agent, so the labels stay as they are.
