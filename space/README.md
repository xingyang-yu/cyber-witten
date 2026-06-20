---
title: Cyber-Witten Retrieval Demo
emoji: 🔭
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Cyber-Witten — retrieval demo

Semantic search over Edward Witten's arXiv papers (BGE-large embeddings + FAISS
exact search). Type a physics question, see which papers the retriever surfaces,
with cosine similarity. **Retrieval only**: no LLM, no API key, runs on a free
CPU Space. The grounded-answer step (cite-or-fail generation) lives in the local
app: https://github.com/xingyang-yu/cyber-witten

## Data

Loads the arXiv-only public bundle produced by `scripts/export_public.py`
(`bge.faiss` + `lookup.jsonl`, ~82 MB). Pre-arXiv `inspire:*` papers are excluded
(not redistributable).

## Deploying to a HuggingFace Space

1. **Upload the bundle as a HF dataset** (one-time):
   ```bash
   # from the repo root, after running scripts/export_public.py
   huggingface-cli upload <user>/cyber-witten-corpus \
       data/public_export/bge.faiss bge.faiss --repo-type dataset
   huggingface-cli upload <user>/cyber-witten-corpus \
       data/public_export/lookup.jsonl lookup.jsonl --repo-type dataset
   ```
2. **Create a Gradio Space** and push the contents of this `space/` directory
   (`app.py`, `requirements.txt`, this `README.md`) to it.
3. In the Space settings, set the variable **`HF_BUNDLE_REPO=<user>/cyber-witten-corpus`**.
   On startup the Space fetches `bge.faiss` + `lookup.jsonl` from that dataset.

## Running locally

```bash
# from the repo root, after scripts/export_public.py has produced data/public_export/
pip install -r space/requirements.txt
BUNDLE_DIR=data/public_export python space/app.py
```
