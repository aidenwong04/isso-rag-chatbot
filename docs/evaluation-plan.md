# Corpus strategy and evaluation plan

Working document.
Written 2026-09-10, after the freshness/monitoring work landed.

The question this plans an answer to: **which corpus strategy should this bot use, and how would we know?**

Three candidate strategies are laid out as versions so they can be compared with numbers instead of argued about.
None of them can be compared until an eval set exists, so the eval set is the first piece of work, not the last.

## Where things stand

Corpus: 10 pages, 182 chunks, embedded with `gemini-embedding-001` at 768 dimensions.

Retrieval: brute-force cosine over a numpy array in `src/query.py`, top-3 into the prompt.
No ANN index, and none needed - the corpus would have to grow roughly 100x before that changes.

Guardrails: the system prompt instructs the model to say so when the chunks do not contain the answer.
A similarity-score threshold was considered and deliberately rejected, because in-corpus and out-of-corpus score ranges overlap (one out-of-corpus query scored 0.76, above several in-corpus queries at 0.62-0.66).

Freshness: `src/monitor.py` runs a four-tier drift cascade against the live site.
Verified working - it caught a real dead link on `resources-students` that chunk-level comparison is structurally blind to.

Not yet built: any way to measure whether retrieval is actually good.

## The three versions

Each version is a corpus/retrieval strategy, not a code release.
They share the same parser, embedding model, and prompt so that differences between them are attributable.

### v1 - ingest the whole sitemap

Take all ~657 sitemap URLs, filter to HTML pages, ingest everything.

**Hypothesis:** maximum coverage.
Nothing is missing because nothing was excluded.

**Expected failure, and the reason this version exists as a control rather than a candidate:**
Roughly 6% of the sitemap is newly-admitted-student guidance.
About 339 URLs are news, events and annual reports, which are dated content baked into a static index.
About 59 URLs are scholar and faculty track content - J-1 research scholars, H-1B, permanent residence.

That last group is the dangerous one.
F-1 student rules and J-1 scholar rules differ on work authorization, travel, status maintenance and dependents.
A student asking "can I work off campus" could retrieve a well-written, authoritative, correctly-cited chunk that answers the question for a different visa category.

The existing guardrail does not catch this.
It was built to detect *absence* of an answer.
This is presence of an answer aimed at the wrong audience, which reads to the model as a successful retrieval.

v1 is worth building precisely to measure how often that happens.
If the answer is "never", the whole selection argument is wrong and worth abandoning.

### v1.1 - corpus built by link crawling

Seed from the current 10 pages.
Follow internal links, rank uncovered targets by inbound link count, approve by hand, ingest, repeat one hop at a time.

**Hypothesis:** the link structure of the site encodes ISSO's own editorial judgment about what belongs in the newly-admitted-student flow, so crawling outward from trusted seeds stays inside the right audience boundary without anyone having to define that boundary explicitly.

**Evidence this is plausible:** the current 10 pages contain 253 outbound links, 51 distinct targets on `isso.columbia.edu`, of which 44 are not yet in the corpus.
The top-ranked uncovered targets by inbound count are all plainly in scope:

```
10  /content/schools-columbia-campuses
 4  /content/compass-user-guide
 3  /content/guidelines-maintaining-status-students
 3  /content/estimated-expenses
 3  /content/documents-needed-travel-student
```

**Hypothesis tested 2026-09-10, and it is leaky.**
All 32 uncovered internal HTML targets were fetched and classified by breadcrumb.
17 are explicitly "For Students", but `resources-scholars` crossed on one inbound link, and `resources-dependents-spousechildren` and `families` are dependent-track.
Roughly 9% of one hop landed outside the student boundary.
Dependent content deserves as much caution as scholar content, since F-2 and J-2 rules differ sharply from F-1 - most importantly, dependents generally cannot work.

Conclusion: crawling narrows the corpus toward the right audience but does not enforce the boundary.
It needs a filter, not trust.

**What the measurement has to answer:** did crawling lose any answers that v1 has?
That is a recall question, and it is the whole point of the comparison.

**Known cost:** selection is manual.
The ranking is automated, the judgment is not, deliberately - corpus membership is a scope decision with real consequences, and a similarity threshold was already shown not to separate cleanly.

### v1.2 - link hopping at inference

On top of v1.1.
Chunks carry their outbound links as metadata.
At query time, retrieve normally, then follow links to pull in neighbouring chunks as additional context.

**Hypothesis:** some questions can only be answered by a chunk that similarity cannot surface, because the query's wording matches page A while the answer lives on page B, and only A's link text knows B is relevant.

**Reason to be skeptical, and to build this last:**
Most link-following in this corpus is not a retrieval problem at all.
It is a coverage gap - 44 of 51 internal link targets are pages on the same site that could simply be ingested.
Once ingested, ordinary cosine retrieval finds them and there is nothing to hop to.
v1.1 is expected to dissolve most of the motivation for v1.2.

**When it would earn its place:** a specific, observable failure signature in the query log.
The bot says the chunks do not contain the answer, the answer was in fact in the corpus, and top-k never retrieved it.
If that signature does not appear, v1.2 has nothing to fix.

**Cost that must be measured alongside recall:** an extra round trip per query.
Latency and token cost per query go in the results table next to the recall numbers, or the tradeoff is invisible.

## Metrics

Retrieval and generation are measured separately.
A bad answer can come from retrieval missing the chunk or from the generator fumbling a chunk it had, and a single end-to-end score cannot tell those apart.

### The minimum set, four numbers per version

**Recall@3 and Recall@10** - of the chunks that genuinely answer the query, how many appear in the top k.
This is the ceiling on everything downstream: if the right chunk is not in the top-k, no amount of prompt work can produce a correct answer.
It is also the metric that actually discriminates between the three versions, since all three are arguments about what is reachable.

**Abstention accuracy** - on a held-out set of out-of-corpus queries, how often does the bot correctly decline.
The decision to rely on the LLM guardrail instead of a score threshold is currently an untested judgment call.
This is the number that tests it.

**Audience correctness** - on a deliberately adversarial slice of student queries whose topic also exists in scholar content, how often does the answer come from the right track.
No off-the-shelf framework has this.
It is the metric that decides whether v1 is safe, and it is the one with real-world consequences.

**Faithfulness** - is every claim in the answer supported by the retrieved chunks.
LLM-as-judge is the standard approach.

### Worth adding once the basics work

- **Precision@k** - distractors in context measurably degrade generation, so a version that raises recall by flooding context is not obviously better.
- **Citation accuracy** - does the cited URL support the claim, and does it still resolve.
  Not academic here: the corpus shipped 20 chunks with `url: None`, and `resources-students` carried a link that 404s.
- **MRR** - useful when a query has exactly one right chunk.

### Recorded with every run, or the result is not a data point

- embedding model id and output dimensionality
- corpus page count and chunk count
- eval set hash
- k
- git commit

The embedding model matters more than it looks.
`gemini-embedding-001` and `output_dimensionality=768` are currently pinned in two separate files.
If that changes between versions, the comparison measures the embedding model rather than the corpus strategy, and nothing will raise an error.
Pin it in one place before the first run.

## Bootstrapping the eval set

This is the piece everything else waits on, and the piece that cannot be automated away.

`data/log.json` is a lab notebook, not an eval set.
It records the top chunk that was returned, but no claim about which chunk *should* have been returned, so recall cannot be computed from it.

A label is a statement about what correct retrieval looks like.
Only someone who knows the corpus can make one, which means these labels are hand-written.

### The sequencing problem

Real user queries are strictly better than anything written at a desk.
Students do not type "What are the document requirements for F-1 visa application".
They type "what do i need to bring to my visa interview".

But users are needed to get real queries, an eval set is needed to know the bot is safe enough to put in front of users, and the specific risk identified above is a student receiving confidently-served scholar rules for a student question.
That is not a risk to discover through a real person making a real immigration decision.

### The way out

1. **Hand-label roughly 50 queries now.**
   Enough to establish that a version does nothing dangerous.
   Not enough to be representative, and it should not be mistaken for representative.
   Write them the way a student would type them, not by paraphrasing chunk headings - headings-as-queries inflates every recall number produced.

2. **Ship to a small, informed audience.**
   People who know it is an experiment.
   Log every query, the retrieved chunk ids and scores, the answer, and latency.

3. **Grow the eval set from real traffic.**
   Users supply the query distribution.
   Labels stay hand-written.

4. **Only then consider showing it to ISSO.**

### On synthetic queries

Prompting an LLM to generate questions from each chunk makes ground truth free - the answer is the chunk it was generated from.
The trap is that synthetic queries share vocabulary with their source chunk, so recall comes out optimistically high.
Use them for regression detection between versions.
Do not believe the absolute value.

### On user feedback

Collecting queries and collecting labels have very different costs.
Queries are free and need no participation - logging produces them.
Labels require someone who knows the corpus.

A thumbs up/down control is therefore best understood as a *sampling strategy*, not a source of labels.
It tells you which queries are worth spending labeling time on.

Two things to expect:

- Voluntary feedback rates are well under 1%, and negative feedback is over-represented because people click when annoyed.
  Twelve downvotes out of three thousand queries is not a 0.4% error rate, it is a sample of the angry tail.
- The strongest signal is implicit, not explicit.
  An immediate rephrase-and-retry on the same topic is a failure report that costs the user nothing, and it comes from logging alone.

### Privacy, to decide before logging real users

Query logs from international students are sensitive.
Someone asking what happens if they fall out of status is disclosing something they may not want durably attached to them.

Retention period, anonymization and access need to be decided deliberately and early rather than bolted on.
This is also the first question ISSO's own staff will ask.

## Shipping to users (decided 2026-09-10)

The corpus going to users is the **current student-track corpus, expanded by link crawl** - not plan-doc v1.
v1 means the whole sitemap, which is the contaminated configuration.
The v1/v1.1/v1.2 comparison is an offline experiment that needs an eval set, not users.
What users provide is the query distribution, and a clean corpus supplies that just as well as a dirty one.

**Distribution: a Reddit post plus people Aiden meets in person.**
That changes the audience from an informed cohort to strangers who may act on the answers, with two consequences.

First, the corpus is scoped to *newly admitted* students, while Reddit will bring current students asking about OPT, CPT, travel signatures, status violations and job offers.
A large share of real queries will therefore be out of scope, which makes **abstention quality the most important behaviour of the system**, ahead of answer quality.

Second, three things must be settled before the post goes up:

- **Email ISSO before posting.** The goal is eventually to pitch them.
  Them learning third-hand that a student runs a public unofficial ISSO-advice bot is a worse first contact than a short note beforehand.
- **The privacy decision is due now, not later.** Retention, whether raw query text is stored long-term, and who can read the log.
- **Frame it as unmistakably unofficial**, in the post and on the page, and keep the API key server-side behind a per-session rate limit.

### Expansion shortlist

From one hop of link crawling, ranked by inbound links and filtered by breadcrumb audience:

- **17 unambiguous** - breadcrumb says "For Students". Top by inbound: `guidelines-maintaining-status-students`, `documents-needed-travel-student`, `resources-new-students`, `about-your-visa`.
- **9 need a judgment call** - no stated audience. Cross-cutting utilities that plainly belong (`schools-columbia-campuses` at 10 inbound, `compass-user-guide`, `estimated-expenses`, `scams`) sit alongside `transferring-your-sevis-record-columbia-j-1-student`, which is J-1 *student* rather than scholar - in scope, but on the F-1/J-1 axis the breadcrumb cannot see.
- **3 excluded as news/events**, and 3 excluded by audience (1 scholars, 2 dependents).

Raw data: `data/state/link-audience-2026-09-10.json`.

## System prompt (rewritten 2026-09-10)

Rewritten for a student audience rather than a developer testing it.
Three cases are now named explicitly: answered, no answer with something genuinely related, and no answer with nothing related.

Changes that came from observed failures rather than taste:

- **Citations were being fabricated by stitching.** The model took a page name mentioned inside a chunk's body and paired it with that chunk's URL, producing `More info: Guidelines to Maintaining Status (.../applying-your-transfer-i-20)` - a real heading beside the wrong page.
  The rule now requires the Heading and URL to be copied from the same numbered source, and forbids building a citation out of link text appearing inside a source.
  Verified consistent across all test cases afterwards.
- **Prohibitions did not work; positive phrasing did.** "Never say chunks/context" left "The provided source material does not contain..." intact.
  Replacing it with an opening to use - "I don't have information about ..." - and with "ISSO does have guidance on ..." for the related-topic sentence removed the leakage.
- **Citation is conditional on having used a source**, so a pure non-answer no longer ends with a dangling `More info:` line.

Known residual, and a good eval case rather than a prompt fix: encouraging redirects encourages finding adjacency, so a question with nothing genuinely near it ("am I a resident alien?", a tax question) still sometimes redirects to a loosely related page.

Also surfaced: chunk headings are now user-visible in citations, so uninformative ones like `RELATED INFORMATION` degrade the experience directly.
That turns the chunk-boundary question below into a UX problem, not just a retrieval one.

## Open decisions

- **Where the audience boundary lives.**
  Selecting at ingest bakes it into whatever was crawled.
  The alternative is ingesting broadly and tagging each page with its audience, then filtering candidates by tag before ranking - which makes the boundary an explicit, adjustable field rather than an implicit one.

  **Implemented 2026-09-10.**
  `one-page.py` extracts the breadcrumb and derives an audience, both written onto every chunk and carried through `embed.py` into `chunks.json` as `breadcrumb` and `audience`.
  Neither is part of the embedded string.
  Nothing reads `audience` yet - filtering on it is a per-version decision, not a parser decision.

  **The audience is stated in the page's breadcrumb**, e.g.
  `Home > Employment > For Students > F-1 CPT` versus
  `Home > Getting Started > For Scholars (Professors/Researchers) > Resources for Scholars`.
  ISSO authors it as part of the site navigation, so it does not have to be inferred from URL patterns (shown insufficient) or from embeddings (score ranges shown not to separate).
  Capture it as page-level metadata at parse time and carry it onto every chunk.

  Two limits, both real:
  11 of 32 crawled targets carry no audience segment, and they do not form one group - some are cross-cutting utilities that belong in a student corpus (`schools-columbia-campuses`, `compass-user-guide`, `estimated-expenses`, `scams`), others are news and events that belong in no static index (`news-highlights`, `programs-events-cloned`).
  And the breadcrumb separates student from scholar from dependent, but not F-1 from J-1, which also differ in their rules.
  So it is one axis of audience, not the whole of it.

- **PDFs.**
  `FundingDocumentsChecklist.pdf`, `SponsorCertificationForm.pdf`, `F1andJ1ComparisonChart.pdf` and `SEVP_I-515.pdf` are linked as real guidance and `one-page.py` cannot read any of them.
  Separate ingest path, or excluded and cited as links?

- **The events feed.**
  `paragraph--type--cu-events-feed` is JS-rendered with a JSON `data-url`.
  Dated content in a static index goes stale by construction.
  Fetch the feed, or exclude events from a standing-guidance corpus?

- **Chunk boundaries versus ISSO's own.**
  Several intra-corpus links point at card-group anchors such as `#!#cu_card_group-18049`.
  Those name units the page's authors consider addressable.
  Whether they line up with the current heading-based chunk boundaries is a testable question about whether the chunking matches how the content is actually structured.

- **Whether the bot ever follows a link at answer time, or only ever cites.**
  Argument for cite-only: a student asking about I-94 corrections should be sent to CBP, not read a summary of CBP.

## Settled, so it does not get relitigated

- **No ANN index.** Brute-force numpy cosine is correct at this scale and survives a 100x corpus.
- **No similarity threshold as a relevance filter.** Score ranges overlap; the LLM guardrail covers the case a threshold would try to catch.
  Revisit via a cross-encoder or reranker only if the guardrail starts being fooled by close-but-wrong matches.
- **Scraping is permitted.** `robots.txt` allows `/content/` with no `Crawl-delay`.
  The earlier 403 was a User-Agent block, not policy.
- **Coverage is measured at the block level, not by character count.**
  A character ratio double-counts the heading breadcrumb repeated into every chunk and can exceed 100%.
- **Comparators are the hard part of every check.**
  Three separate false-positive bugs during the monitor work all came from comparing two renderings of the same text: browser pretty-printing, trailing whitespace inside text nodes, and Cloudflare re-keying its email cipher on every response.
  Any new check that compares text needs both sides normalized identically.
