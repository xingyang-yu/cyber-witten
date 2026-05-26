# Cyber-Witten

**A retrieval-augmented question-answering system grounded in Edward Witten's papers.**

12,998 chunks across 289 papers spanning 1976–2026 — his arXiv corpus plus 44 pre-arXiv journal papers recovered via an INSPIRE → Semantic Scholar → Unpaywall fallback chain. Pre-1991 coverage is partial (see [Known limitations](#known-limitations)). Local BGE-large embeddings, exact FAISS search, Claude Sonnet for grounded synthesis.

---

## Why

Witten has published continuously for fifty years across QFT, string theory, topology, and mathematical physics. A general-purpose LLM has read a lot *about* him; this system reads only *him*, and is required to cite the specific paper for every claim. The goal is a tool that answers technical physics questions with the same constraint a careful reader would impose: nothing said without a passage to back it up.

The project was built as a portfolio piece to demonstrate end-to-end RAG engineering: ingestion under messy real-world data conditions (LaTeX, scanned PDFs, missing IDs), domain-aware retrieval, and strict grounded generation.

---

## Demo

Three real queries against the local index. Each table shows the top-5 retrievals with cosine similarity and the source paper. No filtering, no cherry-picking — straight from `scripts/healthcheck.py --embed` style retrieval against the live FAISS index.

**Q: What is the relation between Chern-Simons theory and the Jones polynomial?**

| sim | arxiv_id | year | title |
|----:|---|---:|---|
| 0.827 | [1001.2933](https://arxiv.org/abs/1001.2933) | 2010 | Analytic Continuation Of Chern-Simons Theory |
| 0.817 | [1106.4789](https://arxiv.org/abs/1106.4789) | 2011 | Knot Invariants from Four-Dimensional Gauge Theory |
| 0.797 | [1106.4789](https://arxiv.org/abs/1106.4789) | 2011 | Knot Invariants from Four-Dimensional Gauge Theory |
| 0.797 | [1401.6996](https://arxiv.org/abs/1401.6996) | 2014 | Two Lectures On The Jones Polynomial And Khovanov Homology |
| 0.795 | [1101.3216](https://arxiv.org/abs/1101.3216) | 2011 | Fivebranes and Knots |

**Q: How does mirror symmetry relate Calabi-Yau manifolds?**

| sim | arxiv_id | year | title |
|----:|---|---:|---|
| 0.819 | [hep-th/9112056](https://arxiv.org/abs/hep-th/9112056) | 1991 | Mirror Manifolds And Topological Field Theory |
| 0.785 | [hep-th/9306122](https://arxiv.org/abs/hep-th/9306122) | 1993 | Quantum Background Independence In String Theory |
| 0.777 | [hep-th/9112056](https://arxiv.org/abs/hep-th/9112056) | 1991 | Mirror Manifolds And Topological Field Theory |
| 0.774 | [0710.5939](https://arxiv.org/abs/0710.5939) | 2007 | Geometric Endoscopy and Mirror Symmetry |
| 0.774 | [hep-th/9112056](https://arxiv.org/abs/hep-th/9112056) | 1991 | Mirror Manifolds And Topological Field Theory |

**Q: What is the entropy of a black hole in JT gravity?**

| sim | arxiv_id | year | title |
|----:|---|---:|---|
| 0.800 | [2006.03494](https://arxiv.org/abs/2006.03494) | 2020 | Deformations of JT Gravity and Phase Transitions |
| 0.775 | [2206.10780](https://arxiv.org/abs/2206.10780) | 2022 | An Algebra of Observables for de Sitter Space |
| 0.765 | [2412.15549](https://arxiv.org/abs/2412.15549) | 2024 | Algebras and states in super-JT gravity |
| 0.761 | [2301.07257](https://arxiv.org/abs/2301.07257) | 2023 | Algebras and States in JT Gravity |
| 0.761 | [2412.15549](https://arxiv.org/abs/2412.15549) | 2024 | Algebras and states in super-JT gravity |

The retriever picks the canonical paper for each topic (1991 mirror-manifolds, 2020 JT-gravity-deformations) and stays within Witten's own work across a 35-year span. The generation step (top-K → Claude Sonnet) requires `ANTHROPIC_API_KEY`; retrieval alone runs in <200ms per query on a laptop once the model is warm.

---

## Architecture

```
              ┌──────────────────────────────────────────────────────┐
              │                    Ingestion                         │
              │                                                      │
   arXiv API ─┼─▶ 01_pull_metadata ─▶ papers.jsonl                   │
              │                                                      │
   arXiv src ─┼─▶ 02_download_sources ─▶ data/sources/*.tar          │
              │                                                      │
              │   03_parse_and_chunk (pylatexenc → text)             │
              │           │                                          │
              │           ▼                                          │
              │       chunks.jsonl ──────────────┐                   │
              │                                  │                   │
   INSPIRE ──▶│ 06_pre_arxiv ──┐                 │                   │
   S2 / DOI   │                ├─ PDF → text ────┤                   │
   Unpaywall ▶│                │  (pdfplumber +  │                   │
              │ 07_manual_pdfs ┘   tesseract OCR)│                   │
              └──────────────────────────────────┼───────────────────┘
                                                 │
              ┌──────────────────────────────────┴───────────────────┐
              │                  Indexing                            │
              │                                                      │
              │   04_build_index:                                    │
              │     BAAI/bge-large-en-v1.5 (transformers-direct)     │
              │     CLS pooling + L2 normalize                       │
              │     FAISS IndexFlatIP, dim=1024                      │
              │                                                      │
              │     bge.faiss ─── parallel ─── lookup.jsonl          │
              └────────────────────┬─────────────────────────────────┘
                                   │
              ┌────────────────────┴─────────────────────────────────┐
              │                  Query (ask.py)                      │
              │                                                      │
              │   question ─▶ BGE encode (with query prefix) ─▶ top-K│
              │                                                      │
              │   passages + strict system prompt ─▶ Claude Sonnet   │
              │   ──▶ grounded answer with inline [arxiv_id] cites   │
              └──────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

**Sanity check the corpus** (fast, no model load):

```bash
.venv/bin/python scripts/healthcheck.py
```

**Smoke test retrieval** (loads the 1.3GB BGE model on first run):

```bash
.venv/bin/python scripts/healthcheck.py --embed
```

**Ask a question**:

```bash
.venv/bin/python ask.py "What is the relation between Chern-Simons theory and the Jones polynomial?"
.venv/bin/python ask.py "..." --show-passages    # also print retrieved passages
.venv/bin/python ask.py "..." -k 12              # retrieve more
```

The BGE model is auto-resolved from `~/.cache/huggingface` if present, otherwise downloaded. Override with `BGE_MODEL_PATH=/path/to/snapshot`.

---

## Corpus

| | |
|---|---|
| Total chunks | 12,998 |
| Total papers | 289 |
| via arXiv | 245 papers / 11,684 chunks |
| via INSPIRE+DOI fallback | 44 papers / 1,314 chunks |
| Year range | 1976 – 2026 |
| Embedding dim | 1024 (BGE-large-en-v1.5) |
| Index | FAISS `IndexFlatIP` (exact, L2-normalized → cosine) |
| Primary categories | hep-th (90%), math.AG, cond-mat.str-el, math.DG, cond-mat.mes-hall |

Chunks are paragraph-bounded, 512-token-truncated, with `chunk_id = {arxiv_id}_{idx:03d}`. Pre-arXiv papers use synthetic IDs of the form `inspire:{recid}` and are excluded from the public redistribution bundle (see [Data provenance](#data-provenance--license)).

---

## Design decisions

### Why local BGE-large over a hosted embeddings API

Voyage and OpenAI embeddings would have been one line of code and finished embedding in minutes. I chose local BGE-large-en-v1.5 because:

- **Cost predictability.** Re-embedding the full corpus is free; I can iterate on chunk strategy without per-token anxiety.
- **No data egress.** Witten's papers are public, but the principle generalizes: nothing in the indexing pipeline phones home.
- **Demonstrates the workflow.** Any serious RAG engineer needs to be comfortable running real embedding models; using a hosted API hides that surface.
- **BGE-large benchmarks competitively with Voyage-2 / text-embedding-3-large on MTEB retrieval** at the cost of one local GPU/MPS minute per thousand chunks.

The retrieval-time cost on Apple Silicon is ~150ms per query end-to-end after the model is warm.

### Why RAG over fine-tuning / continued pretraining

For a corpus this small (~50MB of text), fine-tuning is the wrong tool:

- **Hallucination control.** RAG with strict citation prompting can guarantee that every claim points to a specific passage. A fine-tuned model can still confidently fabricate.
- **Update latency.** Adding a new Witten paper means appending to a JSONL and running `05_delta_add.py` (~30 seconds). Fine-tuning would mean a new training run.
- **Transparency.** With `--show-passages` you can see exactly what the model was given. With fine-tuning, the "knowledge" is opaque weights.
- **Scope honesty.** I want this system to *fail loudly* when the corpus doesn't cover a question, not to confabulate from latent training-data familiarity.

Fine-tuning *would* help with stylistic mimicry — generating prose that sounds like Witten — but that's not the goal. The goal is technically correct answers attributable to specific papers.

### The INSPIRE multi-source strategy

The arXiv only goes back to 1991. Witten published heavily from 1976–1990 — including foundational work like *Dynamical Breaking of Supersymmetry* (1981) and the Jones polynomial paper (1989). A pure-arXiv corpus would be missing the historical bedrock.

The pipeline in `scripts/06_pre_arxiv.py` walks a four-step fallback chain for every pre-arXiv paper in his INSPIRE bibliography:

1. **INSPIRE `documents` field** — direct hosted PDF when the journal allows it.
2. **Semantic Scholar `openAccessPdf`** — picks up a lot of self-hosted preprints.
3. **Unpaywall by DOI** — best legitimate open-access copy.
4. **Manual dropbox** (`07_ingest_manual_pdfs.py`) — for the long tail (Springer / APS / Elsevier PDFs the user obtains separately), with filename → INSPIRE recid resolution.

Parsing falls back from `pdfplumber` (digital PDFs) to `pytesseract` OCR (scans). The result: 44 pre-arXiv papers recovered, including all of the canonical 1980s work. The trade-off is that the INSPIRE-sourced papers carry redistribution risk, so the public export filters them out.

### Why FAISS `IndexFlatIP` (no ANN, no quantization)

At 13k vectors / 1024 dims, exact inner-product search costs ~10ms per query and 53MB on disk. An ANN index (HNSW, IVF-PQ) would save nothing meaningful and would introduce recall failures. Premature optimization is more costly than a flat scan that fits in L2.

### Why `transformers` directly instead of `sentence-transformers`

`sentence-transformers` imports scikit-learn and a stack of evaluator modules at module load — adding ~2 seconds of cold start that the application doesn't use. `scripts/bge_embed.py` reimplements just the BGE-specific pieces (CLS pooling + L2 normalize + query-side prompt prefix) against `transformers.AutoModel` directly. The result is the same vectors, half the import time, and one fewer dependency tree to audit.

### Strict no-fallback prompting

The system prompt in `ask.py` is the load-bearing piece of the generation step:

> *Answer using ONLY the passages provided. ... If the passages don't contain enough to answer, say so explicitly. Do NOT fall back to general knowledge or invent details.*

Combined with required inline citations (`[hep-th/9112056]`, `[1106.4789]`, `[inspire:264818]`), this makes hallucination visible: a claim without a bracketed ID is, by construction, a violation. Speculative bridging beyond the corpus must be marked `[outside corpus]`.

---

## Pipeline (scripts in execution order)

| Script | Purpose |
|---|---|
| `01_pull_metadata.py` | arXiv API → `data/metadata/papers.jsonl` |
| `02_download_sources.py` | arXiv source tarballs → `data/sources/` |
| `03_parse_and_chunk.py` | LaTeX → text via pylatexenc → `data/chunks/chunks.jsonl` |
| `04_build_index.py` | Embed all chunks → `data/index/{bge.faiss, lookup.jsonl}` |
| `05_delta_add.py` | Append specific missing arXiv IDs without re-embedding the world |
| `06_pre_arxiv.py` | INSPIRE + S2 + Unpaywall fallback for pre-1991 papers |
| `07_ingest_manual_pdfs.py` | Resolve manually dropped PDFs (filename → INSPIRE) |
| `bge_embed.py` | BGE helpers used by 04/05/06/07 and `ask.py` |
| `healthcheck.py` | Fast corpus + retrieval sanity check |
| `export_public.py` | Drop `inspire:*` rows → arXiv-only redistribution bundle |

---

## Operations

**Add new arXiv papers (incremental)**:

```bash
echo "2605.15180" >> /tmp/missing_arxiv_ids.txt
.venv/bin/python scripts/05_delta_add.py
```

**Add a pre-arXiv paper manually**:

```bash
# Drop PDFs into data/pdfs/ named by INSPIRE recid (e.g. 193975.pdf)
.venv/bin/python scripts/07_ingest_manual_pdfs.py
```

**Produce a redistribution-safe bundle**:

```bash
.venv/bin/python scripts/export_public.py
# → data/public_export/{bge.faiss, lookup.jsonl, chunks.jsonl, manifest.json}
```

---

## Known limitations

- **Equations are lossy.** LaTeX → text via `pylatexenc` preserves most math semantics but flattens display equations. Queries about specific equation forms (e.g. "what is the exact action in equation 2.7 of...") retrieve the right passage but the rendered text isn't pretty.
- **No diagrams.** Figures and TikZ are dropped. A question about a specific Feynman diagram can be answered descriptively but never visually.
- **No reranker.** Top-K is whatever BGE cosine returns. A cross-encoder reranker (e.g. BGE-reranker-v2) would likely improve precision on ambiguous queries; not implemented to keep latency under 1s.
- **No query rewriting.** A question like "what did he say about that black hole thing in '83?" relies on the BGE encoder to disambiguate. A small LLM rewriting step would help; deferred.
- **Pre-arXiv recall is incomplete.** 44 of ~70 pre-1991 papers were recovered. The rest are paywalled in ways the fallback chain couldn't break, or are conference proceedings without DOIs.
- **OCR quality on early scans is mixed.** Pre-1985 papers sometimes have noisy passages where tesseract misread Greek letters and mathematical symbols. These show up in retrieval but degrade readability.
- **Citation-grounding is enforced by prompting, not by post-hoc validation.** A truly safe system would parse the model output, verify each `[id]` resolves to a passage that was actually retrieved, and refuse the response otherwise. Deferred.

---

## Future work

- **Pre-1991 OCR cleanup pipeline.** Equation-aware OCR (Mathpix, Nougat) on the scan-quality outliers to lift retrieval quality on the 1976–1985 papers.
- **Citation validator.** Parse model output, regex out `[id]` tokens, verify membership in the retrieved set, refuse non-grounded responses.
- **Cross-encoder reranker.** BGE-reranker-v2 over top-50 → top-K=8, evaluate retrieval@k on a hand-curated query set.
- **HuggingFace Space (retrieval-only).** Public Gradio demo of the retriever using the `export_public.py` bundle; no LLM key required.
- **Fine-tuned BGE.** Self-supervised contrastive fine-tune on (paper-title, abstract) pairs from the corpus to specialize the embedder for physics vocabulary. Open question whether the gain over the off-the-shelf checkpoint justifies the engineering.
- **Distillation.** Train a small open-weight model on (question, retrieved-passages, grounded-answer) triples generated by Claude, for a fully-local end-to-end system. Possibly out of scope; included for discussion.

---

## Repo layout

```
.
├── README.md            this file
├── ask.py               query entrypoint: retrieve → Claude → grounded answer
├── requirements.txt     pinned runtime deps
├── .env.example         ANTHROPIC_API_KEY + optional BGE_MODEL_PATH
├── scripts/
│   ├── 01_pull_metadata.py
│   ├── 02_download_sources.py
│   ├── 03_parse_and_chunk.py
│   ├── 04_build_index.py
│   ├── 05_delta_add.py
│   ├── 06_pre_arxiv.py
│   ├── 07_ingest_manual_pdfs.py
│   ├── bge_embed.py
│   ├── healthcheck.py
│   └── export_public.py
└── data/                gitignored — large artifacts shipped via HuggingFace
    ├── metadata/papers.jsonl
    ├── sources/         arXiv source tarballs
    ├── pdfs/            INSPIRE / manual PDFs
    ├── chunks/chunks.jsonl
    ├── index/{bge.faiss, lookup.jsonl}
    └── public_export/   produced by scripts/export_public.py
```

---

## Data provenance & license

- **Code**: MIT. See header comments in each script.
- **arXiv content**: each preprint is the property of its authors and is redistributable under the license the author selected when posting to arXiv. The public export bundle (`scripts/export_public.py`) includes only this content.
- **INSPIRE / journal PDFs**: ingested locally for retrieval research; *not* included in the public export and *not* redistributed by this repository. The `inspire:*` chunk IDs in the local index are filtered out by `export_public.py` before any upload.

The full local corpus is a research artifact for the maintainer; the public bundle is the redistributable subset.
