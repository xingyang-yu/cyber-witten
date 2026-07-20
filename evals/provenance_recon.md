# Provenance recon: can the citation split be done locally? (2026-07-20, night)

Recon for the question "should we integrate inspire-cite into the system": is
the grounded / ungrounded / fabricated classification a **local** check, and how
much does an INSPIRE layer actually have to do? Read the code, then checked the
claim against tonight's 15 audited IDs. Nothing here is committed yet.

## TL;DR

1. **Grounded vs off-passage is already local and already shipping** (validator +
   guardrail, explicitly "no torch, no faiss, no network"). The question "can the
   grounded check be local" is answered: it *is* the current mechanism.
2. **Citations are structured IDs, not prose.** The free-text-citation worry from
   last night is moot: the system prompt mandates `[arxiv_id]` / `[inspire:recid]`
   and `evals/validator.py` already extracts them by regex. No entity
   disambiguation needed.
3. **The real-vs-fabricated split of off-passage cites is mostly NOT local** — and
   I was wrong last night guessing otherwise. On tonight's 15 flagged IDs the
   local in-corpus filter resolves only 1; **14 need INSPIRE**. So an INSPIRE
   layer is genuinely load-bearing, but (a) the client already exists in-repo,
   (b) a date pre-filter and a cache make it cheap in practice.
4. **Net cost is low.** Offline metric = commit `citation_audit.md` as code.
   Online firewall = widen the guardrail's binary check to 3-way + stop
   `_linkify` from dressing fabricated IDs as real links. Both wire together
   pieces that already exist.

## What already exists (file-cited)

- **Per-chunk provenance.** `data/index/lookup.jsonl` chunks carry
  `arxiv_id, title, year, primary_category` (+ chunk_idx). `ask.py:149` and
  `app_local.py:95` both compute `retrieved_ids = [p["arxiv_id"] for _, p in passages]`,
  so "what was shown this turn" is always known locally.
- **Structured-citation extractor.** `evals/validator.py:extract_citations`
  parses the three ID grammars (modern arXiv, legacy arXiv, `inspire:recid`),
  normalizes versions, splits comma-lists, and ignores non-ID brackets
  (`[cite]`, `[outside corpus]`). Pure regex, no network.
- **Local grounded check.** `validator.validate_citations` computes
  `grounding_violation = cited − retrieved ≠ ∅`. Its docstring: "Pure text-in /
  metrics-out: no torch, no faiss, no network."
- **Live guardrail.** `scripts/guardrail.py:generate_grounded` enforces the same
  check at serve time with a corrective regenerate loop; used by `ask.py --guardrail`,
  `evals/run_eval.py`, and `app_local.py`.
- **INSPIRE client, already in-repo.** `scripts/05_delta_add.py:45`,
  `06_pre_arxiv.py:95`, `07_ingest_manual_pdfs.py` all query
  `inspirehep.net/api/literature` with a User-Agent/session. Tonight's audit used
  `?q=arxiv:<id>` for existence. The verification layer reuses this, not greenfield.
- **The audit prototype.** `evals/citation_audit.md` is the manual version of the
  offline metric, taxonomy already worked out.

## The gap

The local check lumps **every** off-passage cite into one `invalid_citations`
bucket the code internally still calls "fabricated" (`guardrail.py` `problem="fabricated"`,
the `_FABRICATED` correction prompt, the validator docstring's "fabricated *or
misremembered*"). Tonight's relabel to "ungrounded" was **display-only**
(`ask.py:166`, `app_local.py:108`); the internals still say fabricated. Splitting
that bucket into real-off-passage vs nonexistent is exactly what INSPIRE adds.

## How local can the split actually get? (tonight's 15 IDs)

| tier | check | cost | tonight |
|---|---|---|---|
| grounded | `cited ∈ retrieved_ids` | local | (the valid cites) |
| impossible-date fabrication | date-plausibility rule | local | catches e.g. `hep-th/8910145` (dated 1989) |
| in-corpus, off-passage | `cited ∈ 293-corpus` | local | **only 1 / 8** real IDs (`hep-th/9407087`, the real Seiberg-Witten paper) |
| out-of-corpus real vs fake | INSPIRE `?q=arxiv:<id>` | **network** | **14 / 15** flagged IDs land here |

The correction to last night's guess: the corpus is Witten's *authored* papers, so
when the model cites off-passage it usually reaches for real papers his work
*references* (others' papers), which are not in the 293-set. So the local
in-corpus filter is a correct zero-cost first pass but does **not** remove most
INSPIRE calls. What keeps INSPIRE cheap is instead:
- a **local date-plausibility pre-filter** (the audit's `1900+yy if yy>=91 else
  2000+yy` rule) that kills impossible-date fabrications with no network, and
- a **persistent cache**: the set of distinct off-passage IDs the system emits is
  small and repeats, so each novel ID costs one INSPIRE call once, ever.

There is also a more interesting reading for the paper: 7/8 of the "real
off-passage" cites are correct papers from the true literature that the model
recalled from parameters (right paper, wrong grounding), which is a different and
milder failure than inventing an ID.

## Integration plan

### Offline metric (do first, cheap)
Commit `citation_audit.md` as a script over any results file: take
`invalid_citations` (already produced by the validator) → local date + in-corpus
filters → INSPIRE-resolve the out-of-corpus tail (reusing the ingestion client) →
emit per-condition rates of grounded / ungrounded-in-corpus / ungrounded-out-of-corpus-real
/ fabricated. This is the paper's citation metric, and it makes tonight's relabel
reproducible instead of a one-off.

### Online firewall (the interesting half)
1. Widen `validate_citations` / the guardrail from binary (grounded vs violation)
   to the 3-way verdict, with the local tiers first and one cached INSPIRE call
   for a novel out-of-corpus ID.
2. Fix `app_local.py:_linkify` — it currently linkifies **every** bracketed ID,
   so a fabricated `[hep-th/8910145]` renders as an official-looking arXiv link
   that 404s. Replace with a per-cite badge: green grounded / yellow "real, not in
   retrieved set" / red "no INSPIRE record" (withhold the link, warn).
3. Make the regenerate loop honest: **strip + regenerate** on fabricated, but
   **allow + badge** on ungrounded-real, instead of today's "regenerate on any
   off-passage cite." This turns the guardrail from a grounding-enforcer into a
   provenance-labeler, which is the more defensible behavior and a better demo.
4. Architecture: pre-resolve the 293-corpus at load, keep a JSON cache of
   out-of-corpus verdicts; runtime is a local lookup except on a never-seen ID.

## The one genuinely hard bucket

INSPIRE existence cannot catch **real-but-irrelevant** cites (the audit's
`1803.04574` "quantum reservoir computing" resolves fine but is topically bogus).
That needs an abstract-vs-claim relevance judgment (inspire-cite's abstract fetch
+ an LLM call) — the only model-dependent, latency-adding piece. Recommendation:
the offline metric fetches abstracts for the out-of-corpus-real tail and reports
relevance as a sub-category; the online firewall skips it in v1 (badge "real, off
passage" and stop there).

## Bottom line

- **Can the grounded check be local? Yes — it already is, and citations are
  structured IDs so there is no parsing barrier.**
- INSPIRE is load-bearing only for the real-vs-fabricated split of the
  out-of-corpus tail (14/15 tonight), but its client already lives in the
  ingestion scripts and a date-filter + cache keep it cheap.
- Both layers are assembly, not new infrastructure. Offline metric first (it is
  `citation_audit.md` as code and feeds the paper); online firewall second (3-way
  guardrail + honest `_linkify`), as the paper's mitigation section.
