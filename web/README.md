---
title: Cyber-Witten
emoji: 🔭
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
---

# Cyber-Witten — client-side retrieval demo (+ bring-your-own-key answers)

Semantic search over Edward Witten's arXiv papers, running **entirely in the
browser**: the query is embedded client-side with transformers.js
(BGE-small, 384-dim), and top-K is a brute-force cosine over ~11.7k precomputed
passage vectors. No server, no API, nothing sent anywhere. Free forever on a
HuggingFace **Static** Space.

By default it is a finder. A visitor can optionally paste **their own API key**
(Anthropic / OpenAI / DeepSeek / local Ollama / any OpenAI-compatible base URL)
under "Grounded answer": the browser then sends the top-8 passages plus the
strict cite-or-fail prompt directly to that provider, and a JS port of the
citation guardrail (`evals/validator.py` + `scripts/guardrail.py`) checks the
answer — fabricated or missing citations trigger one corrective retry, then
fail loudly with a visible badge. The key lives in the tab and goes only to
the chosen provider; the page has no backend. The full offline version is the
local app: https://github.com/xingyang-yu/cyber-witten

## Files

    index.html          the whole app (UI + retrieval + BYOK generation + guardrail)
    data/vectors.bin    float32 [N, 384], L2-normalized passage embeddings
    data/meta.json      [{id, year, title, snippet}] parallel to vectors
    data/texts.json.gz  full passage texts, gzipped (~9MB) — lazily fetched only
                        when a visitor uses BYOK generation
    data/config.json    {n, dim, model, query_prefix}

Regenerate `data/` with `python scripts/build_web_index.py` from the repo root
(needs the arXiv-only bundle from `scripts/export_public.py`).

## Deploy (free Static Space)

1. Create a **Static** Space at huggingface.co/new-space (SDK: Static — the free one).
2. Upload the contents of this `web/` directory to it:
   ```bash
   huggingface-cli upload <user>/cyber-witten web/ . --repo-type space
   ```
   (uploads index.html + data/ preserving structure). The Space serves
   `index.html` and goes live at `https://huggingface.co/spaces/<user>/cyber-witten`.

Note: BGE-small is smaller/less sharp than the BGE-large index used by the local
app and the README demo tables; this browser demo trades some retrieval quality
for zero-cost, no-server hosting.
