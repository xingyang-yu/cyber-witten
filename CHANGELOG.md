# Changelog

Engineering release notes for Cyber-Witten. For the narrative version — the
findings, the plot twists, and what broke along the way — see the
[blog dev log](https://xingyangyu.com/blog/cyber-witten.html#devlog).

Versions are milestones, not shipped releases: this is a personal research tool.

## v0.6 — Corpus Archaeology (2026-07-20)

### Added
- Shareable corpus **manifest** (`scripts/build_manifest.py` → `data/manifest/witten_corpus.jsonl`):
  identifiers + metadata + an obtainability `tier` (arxiv / oa_inspire / paywalled /
  metadata_only), and **no copyrighted full text**.
- Self-serve **coverage/gap** tool (`scripts/corpus_coverage.py`): diffs the manifest
  against the local corpus and writes a publisher-grouped download list. Others
  reproduce the corpus from arXiv (free) plus their own journal access; the tool
  never fetches paywalled content itself.

### Fixed
- **Author disambiguation**: query INSPIRE by the BAI `Edward.Witten.1` (recid 983328)
  instead of a fuzzy name, which had smuggled the astronomer C.E.C. Witten's galaxy
  papers into the corpus.
- **Ingestion robustness** (`scripts/05_delta_add.py`): `parse_paper` now falls back
  pylatexenc → regex de-TeX (body, then whole file) → PDF (source, then arXiv),
  recovering single `.tex.gz`, LaTeX-2.09 / plain-TeX that pylatexenc empties, and
  PDF-only sources. Recovered 5 previously-dropped articles; arxiv-tier coverage complete.

### Known
- 88 paywalled pre-1991 papers still need a bring-your-own subscription to ingest.

## v0.5 — Provenance Engine (2026-07-20)

### Added
- Offline **citation-provenance metric** (`evals/citation_provenance.py` + `evals/test_citation_provenance.py`):
  classifies every off-passage citation as grounded / ungrounded-in-corpus /
  ungrounded-off-corpus-real / fabricated, via local date + in-corpus filters then a
  cached INSPIRE existence check (`evals/cache/inspire_existence.json`).
- **Experiment 1** — retrieval-aware counterfactual set (`evals/run_counterfactual.py`,
  `scripts/build_counterfactual.py`) and **Experiment 2** — authority ladder
  (`evals/run_authority_ladder.py`).

## v0.4 — Controlled Reversal (2026-07-19)

### Added
- Counterfactual refusal experiment: identical retrieved passages paired with a
  supported vs a false-premise twin, isolating premise support from retrieval.
  (Finding — retrieval collapses refusal; distillation's effect is non-monotonic —
  is in the [blog](https://xingyangyu.com/blog/cyber-witten.html#devlog).)

## v0.3 — Truth in Labeling (2026-07-19)

### Changed
- Relabeled the "fabricated citations" metric to **ungrounded** across README, blog,
  and the model card; "fabricated" now names only the verified-nonexistent subset.
  See `evals/citation_audit.md`.

## v0.2 — The Apprentice

### Added
- Distilled **`cyber-witten-7b`** (Q4_K_M GGUF + LoRA adapter) from the RAG teacher;
  it cites unprompted. Published to HuggingFace with a RAG-only warning on the card.

## v0.1 — First Light

### Added
- RAG over Witten's corpus with strict cite-or-fail prompting, a citation
  validator + guardrail, and a pre-generation refusal gate.
