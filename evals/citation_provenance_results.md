# Citation provenance: the audit, now automated (2026-07-20)

`evals/citation_provenance.py` turns the one-off hand audit in `citation_audit.md`
into a reproducible, per-condition metric. It classifies every off-passage
("invalid") citation through a cheap-to-expensive ladder, only the last tier
touching the network (cached to `evals/cache/inspire_existence.json`):

    in-corpus (local) -> impossible-date (local) -> INSPIRE existence (cached net)

Verdicts: `ungrounded_in_corpus`, `ungrounded_out_real`, `fabricated_bad_date`,
`fabricated_no_record`, `unknown` (offline / net error). Tests in
`evals/test_citation_provenance.py` (7 checks, no network, no index needed).

## Validation

Reproduces the hand audit exactly: over the 15 distinct flagged IDs it returns
**8 real / 7 fabricated** (`test_reproduces_hand_audit_8_real_7_fabricated`, and
live over the result files). The two local tiers alone settle 2 of 15
(`hep-th/9407087` in-corpus; `hep-th/8910145` impossible 1989 date); INSPIRE
settles the other 13.

## Per condition (main 4-arm run, `full_A..D`, 115 answers)

| condition | answers | ans w/ off-passage | in-corpus | out-real | fabricated |
|---|---:|---:|---:|---:|---:|
| qwen2.5:7b / closed-book | 23 | 4 | 1 | 4 | 3 |
| qwen2.5:7b / rag         | 23 | 2 | 0 | 0 | 2 |
| qwen2.5:7b / rag+guard   | 23 | 0 | 0 | 0 | 0 |
| cyber-witten-7b / rag       | 23 | 4 | 0 | 3 | 2 |
| cyber-witten-7b / rag+guard | 23 | 0 | 0 | 0 | 0 |

(occurrences of off-passage citations; "fabricated" = bad-date + no-record.)

### Findings

1. **The guardrail drives off-passage citations to zero** in both models
   (both `rag+guard` rows: 0 violations, all citations grounded). This is the
   mitigation working, and it is the number to report when the online layer lands.
2. **Fabrication concentrates without retrieval.** The only impossible-date
   fabrication and the largest fabricated share sit in base closed-book; adding
   retrieval (base `rag`) already removes the bad-date case.
3. **Distillation shifts the off-passage failure from invention to recall.** Base
   `rag` off-passage cites are 100% fabricated (0 real / 2 fab); distilled `rag`
   are majority real papers cited off-passage (3 real / 2 fab). When the distilled
   model leaves the passages it tends to reach for real literature, not invent IDs
   -- the citation-side echo of the "distillation recalls the true literature"
   result in `counterfactual_results.md`.
4. **Some "fabrications" are digit-slips of a real corpus paper, not hallucinated
   from nothing.** The detector flags `2206.10790` (base) and `2306.10780`
   (distilled, in the counterfactual run) as single-digit slips of `2206.10780`
   -- "An Algebra of Observables for de Sitter Space", which is in the corpus and
   in the README demo. The same real paper, fat-fingered two different ways.

## Counterfactual run (`cf_base` + `cf_distilled`, 80 answers)

Almost fully grounded: the only off-passage citation anywhere is the digit-slip
`2306.10780` (distilled `rag`, 2 occurrences); both guardrail arms are clean.
Consistent with the refusal-heavy, low-citation-count answers that experiment
produced. Artifact: `evals/results/provenance_counterfactual.json`.

## Next

This is the **offline** half of the inspire-cite integration (see
`evals/provenance_recon.md`). The online half reuses this classifier at serve
time: widen the guardrail from binary to the 3-way verdict and stop
`app_local.py:_linkify` from rendering fabricated IDs as clickable arXiv links.
