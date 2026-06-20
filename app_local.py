"""Cyber-Witten — local QA app (fully local, no API key).

The "real" Cyber-Witten: retrieval + local LLM generation + the citation
guardrail, all on your machine. Type a physics question, get a grounded answer
that cites Witten's papers by ID (clickable), plus the passages it used. No
cloud, no per-query cost.

Prereqs:
    pip install gradio openai          # gradio = UI, openai = Ollama transport
    brew install ollama && ollama serve
    ollama pull qwen2.5:7b
    python app_local.py                # opens a local web UI

Backend defaults to Ollama / qwen2.5:7b; override with CW_PROVIDER / CW_MODEL.
Uses the full local index (data/index/) if present, else the public bundle
(data/public_export/ from scripts/export_public.py).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
from pathlib import Path

import gradio as gr
from scripts.bge_embed import encode_queries  # finalizes HF/OpenMP env on import

import torch  # noqa: F401  # must precede faiss
import faiss

from ask import SYSTEM_PROMPT, format_passages
from scripts.guardrail import generate_grounded
from scripts.llm_backends import get_backend

ROOT = Path(__file__).resolve().parent
PROVIDER = os.environ.get("CW_PROVIDER", "ollama")
MODEL = os.environ.get("CW_MODEL", "qwen2.5:7b")

_full = ROOT / "data" / "index"
_pub = ROOT / "data" / "public_export"
INDEX_DIR = _full if (_full / "bge.faiss").exists() else _pub
_index = faiss.read_index(str(INDEX_DIR / "bge.faiss"))
_lookup = [json.loads(line) for line in (INDEX_DIR / "lookup.jsonl").open()]

EXAMPLES = [
    "What is the relation between Chern-Simons theory and the Jones polynomial?",
    "How does mirror symmetry relate Calabi-Yau manifolds?",
    "Did Witten give the first proof of the positive energy theorem?",
]

_ID_RE = re.compile(
    r"\[((?:\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7}|inspire:\d+)(?:v\d+)?)\]"
)


def _url(aid: str) -> str:
    if aid.startswith("inspire:"):
        return f"https://inspirehep.net/literature/{aid.split(':', 1)[1]}"
    return f"https://arxiv.org/abs/{aid}"


def _linkify(text: str) -> str:
    """Turn [arxiv_id] / [inspire:N] citations into clickable links."""
    return _ID_RE.sub(lambda m: f"[{m.group(1)}]({_url(m.group(1))})", text)


def _retrieve(question: str, k: int):
    scores, idxs = _index.search(encode_queries([question]), k)
    return [(float(scores[0][j]), _lookup[idxs[0][j]]) for j in range(k)]


def ask(question: str, k: int, use_guardrail: bool):
    question = (question or "").strip()
    if not question:
        return "_Ask a question above._", ""

    passages = _retrieve(question, int(k))
    retrieved_ids = [p["arxiv_id"] for _, p in passages]
    user_msg = (
        f"<question>\n{question}\n</question>\n\n"
        f"<passages>\n{format_passages(passages)}\n</passages>\n\n"
        "Answer the question using only the passages above. Cite each claim."
    )

    backend = get_backend(PROVIDER, MODEL)
    if use_guardrail:
        answer, report = generate_grounded(backend, SYSTEM_PROMPT, user_msg, retrieved_ids)
        if report["grounded"]:
            status = f"✅ grounded · {report['attempts']} attempt(s)"
        else:
            status = f"⚠️ ungrounded ({report['problem']}) after {report['attempts']} attempts"
    else:
        answer = backend.generate(SYSTEM_PROMPT, user_msg)
        status = "guardrail off"

    passages_md = "\n".join(
        f"- `{s:.3f}` [{p['arxiv_id']}]({_url(p['arxiv_id'])}) ({p['year']}) — {p['title']}"
        for s, p in passages
    )
    return _linkify(answer), f"**{status}**\n\n**Retrieved passages**\n\n{passages_md}"


with gr.Blocks(title="Cyber-Witten — local QA") as demo:
    gr.Markdown(
        "# 🔭 Cyber-Witten — local QA\n"
        f"Grounded question answering over Edward Witten's papers, fully local "
        f"(**{PROVIDER} / {MODEL}**, {_index.ntotal:,} passages). Every claim is cited by "
        "paper ID; the guardrail rejects and regenerates an answer that cites a paper not "
        "retrieved, or that cites nothing at all.\n\n"
        "_Generation runs on your machine, so a query takes roughly 1-3 minutes with the "
        "guardrail on (it may regenerate). No API, no per-query cost._"
    )
    with gr.Row():
        question = gr.Textbox(label="Question", scale=4, placeholder="Ask about Witten's physics…")
        k = gr.Slider(4, 12, value=8, step=1, label="Top-K", scale=1)
    with gr.Row():
        guardrail = gr.Checkbox(value=True, label="Citation guardrail (regenerate until grounded)")
        btn = gr.Button("Ask", variant="primary", scale=2)
    answer = gr.Markdown(label="Answer")
    with gr.Accordion("Grounding + retrieved passages", open=False):
        sidebar = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=question)

    btn.click(ask, [question, k, guardrail], [answer, sidebar])
    question.submit(ask, [question, k, guardrail], [answer, sidebar])


if __name__ == "__main__":
    demo.launch()
