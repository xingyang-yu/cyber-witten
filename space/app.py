"""Cyber-Witten — retrieval-only demo (no LLM, no API key).

The free public window onto the corpus: type a physics question, see which of
Edward Witten's papers the retriever surfaces, with cosine similarity. There is
deliberately no generation step here — that is the local app's job. This Space
needs no API key and runs on a free CPU Space.

Self-contained on purpose (does not import the main repo's scripts/), so the
`space/` directory can be pushed to a HuggingFace Space as-is. The BGE encoder
below mirrors scripts/bge_embed.py.

Data: loads the arXiv-only public bundle (scripts/export_public.py output).
Locally it reads ./data/public_export/; on a Space, set HF_BUNDLE_REPO to a HF
dataset holding bge.faiss + lookup.jsonl and it is fetched on startup.
"""
import json
import os
from pathlib import Path

# OpenMP / tokenizer hygiene before any native lib (torch, faiss) loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import numpy as np
import torch  # must import before faiss (OpenMP load-order)
import faiss
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH = 512

BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "data/public_export"))
HF_BUNDLE_REPO = os.environ.get("HF_BUNDLE_REPO")  # e.g. "xingyang-yu/cyber-witten-corpus"

EXAMPLES = [
    "What is the relation between Chern-Simons theory and the Jones polynomial?",
    "How does mirror symmetry relate Calabi-Yau manifolds?",
    "What is the entropy of a black hole in JT gravity?",
    "How does electric-magnetic duality enter Seiberg-Witten theory?",
]


def _ensure_bundle():
    if (BUNDLE_DIR / "bge.faiss").exists() and (BUNDLE_DIR / "lookup.jsonl").exists():
        return
    if not HF_BUNDLE_REPO:
        raise RuntimeError(
            f"No bundle at {BUNDLE_DIR} and HF_BUNDLE_REPO not set. "
            "Run scripts/export_public.py locally, or point HF_BUNDLE_REPO at a HF dataset."
        )
    from huggingface_hub import hf_hub_download

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("bge.faiss", "lookup.jsonl"):
        hf_hub_download(
            repo_id=HF_BUNDLE_REPO, filename=fname, repo_type="dataset",
            local_dir=str(BUNDLE_DIR),
        )


_ensure_bundle()
_index = faiss.read_index(str(BUNDLE_DIR / "bge.faiss"))
_lookup = [json.loads(line) for line in (BUNDLE_DIR / "lookup.jsonl").open()]
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME).eval()


def _encode(question: str) -> np.ndarray:
    enc = _tokenizer(
        [QUERY_PREFIX + question], padding=True, truncation=True,
        max_length=MAX_SEQ_LENGTH, return_tensors="pt",
    )
    with torch.no_grad():
        out = _model(**enc)
    emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
    return emb.cpu().numpy().astype("float32")


def search(question: str, k: int) -> str:
    question = (question or "").strip()
    if not question:
        return "_Type a question above._"
    scores, idxs = _index.search(_encode(question), int(k))

    lines = [f"**Top {int(k)} passages for:** _{question}_\n"]
    for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), 1):
        p = _lookup[int(i)]
        aid = p["arxiv_id"]
        url = f"https://arxiv.org/abs/{aid}"
        snippet = " ".join(p["text"].split())[:320]
        lines.append(
            f"**{rank}. `{score:.3f}` · [{aid}]({url}) ({p['year']})** — {p['title']}  \n"
            f"> {snippet}…\n"
        )
    return "\n".join(lines)


with gr.Blocks(title="Cyber-Witten — retrieval demo") as demo:
    gr.Markdown(
        "# 🔭 Cyber-Witten · search Witten's papers\n"
        "Ask a physics question and this finds the most relevant passages from "
        f"**Edward Witten's papers** (his full arXiv corpus, {_index.ntotal:,} passages), "
        "ranked by meaning rather than keywords.\n\n"
        "It is a **finder, not a chatbot**: it returns the source passages (with arXiv links) "
        "that best match your question, it does not write an answer for you. The "
        "answer-writing version, which cites every claim and refuses when the papers don't "
        "support it, is the downloadable [local app](https://github.com/xingyang-yu/cyber-witten) "
        "and uses no paid API.\n\n"
        "_Pick an example below or type your own. The first search after the demo has been idle "
        "can take a minute while the model loads._"
    )
    with gr.Row():
        question = gr.Textbox(label="Question", scale=4, placeholder="Ask about Witten's physics…")
        k = gr.Slider(1, 15, value=5, step=1, label="Top-K", scale=1)
    btn = gr.Button("Search", variant="primary")
    out = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=question)
    btn.click(search, [question, k], out)
    question.submit(search, [question, k], out)


if __name__ == "__main__":
    demo.launch()
