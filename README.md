# Cyber-Witten

Cyber-Witten is a local RAG corpus over Edward Witten papers. It stores parsed
paper chunks on disk, embeds them with `BAAI/bge-large-en-v1.5`, indexes them in
FAISS, and answers questions through Anthropic using retrieved passages only.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env` before asking for generated answers.

Run a fast corpus check:

```bash
.venv/bin/python scripts/healthcheck.py
```

Run a retrieval smoke test. This loads the BGE model, so the first run can take
a minute or two:

```bash
.venv/bin/python scripts/healthcheck.py --embed
```

Ask a question:

```bash
.venv/bin/python ask.py "What is the relation between Chern-Simons theory and the Jones polynomial?"
```

Show retrieved passages as well:

```bash
.venv/bin/python ask.py "What is the relation between Chern-Simons theory and the Jones polynomial?" --show-passages
```

## Data Pipeline

The scripts are numbered in pipeline order:

- `scripts/01_pull_metadata.py`: pull Edward Witten arXiv metadata.
- `scripts/02_download_sources.py`: download arXiv source archives.
- `scripts/03_parse_and_chunk.py`: parse LaTeX and write `data/chunks/chunks.jsonl`.
- `scripts/04_build_index.py`: build `data/index/bge.faiss` and `lookup.jsonl`.
- `scripts/05_delta_add.py`: append specific missing arXiv IDs from `/tmp/missing_arxiv_ids.txt`.
- `scripts/06_pre_arxiv.py`: hunt and ingest pre-arXiv PDFs.
- `scripts/07_ingest_manual_pdfs.py`: ingest PDFs manually placed in `data/pdfs/`.

The large `data/` artifacts are intentionally gitignored.

## Model Cache

By default, the embedding helper first looks for an existing local
`BAAI/bge-large-en-v1.5` HuggingFace snapshot. You can force a path with:

```bash
BGE_MODEL_PATH=/path/to/bge-large-en-v1.5/snapshot .venv/bin/python ask.py "..."
```
