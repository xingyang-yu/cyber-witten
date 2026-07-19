# Citation audit: "ungrounded" is not the same as "fabricated"

**Prompted by a reader's doubt (2026-07-19):** the eval's `grounding_violation`
metric flags any answer that cites a paper ID **not present in the retrieved
passages**. The validator docstring already calls this "a fabricated *or
misremembered* citation" — but the blog/README prose collapsed it to "fake /
fabricated." Are those flagged citations actually fabricated, or real papers
cited off-passage?

## Method

Extracted every `invalid_citations` entry across all result files
(`full_run_merged`, `final_showdown`, `full_A..D`), then classified each ID:

1. **Date sanity** — old-style `archive/YYMMNNN` valid 1991-08 .. 2007-03;
   new-style `YYMM.NNNNN` valid from 2007-04. (First pass had a bug: `yy < 91`
   wrongly flagged 2000s IDs like `math-ph/0212366` as pre-arXiv. Fixed:
   `year = 1900+yy if yy>=91 else 2000+yy`. The audit tool made the *same class*
   of false-fabrication error it was auditing for — logged here on purpose.)
2. **Existence** — queried INSPIRE-HEP (`/api/literature?q=arxiv:<id>`) for every
   date-plausible ID.

## Result: 15 distinct flagged IDs, ~50/50

| verdict | count | IDs |
|---|---|---|
| **Real paper** (verified on INSPIRE), not retrieved | **8** | hep-th/9407087 (the *actual* Seiberg-Witten paper), hep-th/9109055, hep-th/9804195, hep-th/9807022, hep-th/9606101, hep-th/9812208, 1803.04574, 2507.06945 |
| **Fake / nonexistent** (impossible date or no INSPIRE record) | **7** | hep-th/8910145 (dated 1989, before arXiv), 1605.08291, 1601.03987, hep-th/9205140, 1803.04576, math-ph/0212366, 2206.10790 (likely a digit-slip of the real 2206.10780) |

By occurrences it is ~16 / 16.

## What this does and does not change

- **The metric and its numbers are valid.** `grounding_violation` = "cited a
  paper not in the retrieved passages." In a RAG system that is a real failure
  regardless of whether the paper exists: the model asserts a source it was
  never shown. The *rate* stands.
- **The word "fabricated / fake" was overstated.** About half of the flagged
  citations are real papers cited off-passage (misremembered provenance), not
  invented IDs. The accurate umbrella term is **ungrounded** (or off-passage).
- **The two flagship fabrication examples hold.** `hep-th/8910145` (impossible
  pre-arXiv date) and `1605.08291` (no INSPIRE record) are genuine fabrications,
  fine to keep as illustrations of the *fabricated subset*.
- **A third, in-between category exists:** real papers that are also topically
  irrelevant (e.g. 1803.04574 "quantum reservoir computing", 2507.06945 a seesaw
  modular-forms paper) — the ID resolves, but the citation is still bogus.

## Recommended taxonomy for the writeup

Report the headline number as **ungrounded citations** (cited a paper not in the
passages), then break it down:

- **fabricated** — ID does not resolve to a real paper (impossible date / no record);
- **off-passage-real** — real paper, but not among the retrieved passages;
  - *relevant* (misremembered which retrieved paper) vs *irrelevant* (random real ID).

This is a "measure your measurement" result: the label was wrong even though the
number was right, and the audit tool reproduced the same bug it was hunting.
