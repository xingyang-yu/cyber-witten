"""Ask Cyber-Witten a question, grounded in Witten's paper corpus.

Usage:
    # Default backend (Claude Sonnet via Anthropic):
    python ask.py "What is the relation between Chern-Simons and the Jones polynomial?"
    python ask.py "..." --show-passages    # also print retrieved passages
    python ask.py "..." -k 12              # retrieve more passages

    # Swap LLM backend:
    python ask.py "..." --provider openai --model gpt-4o
    python ask.py "..." --provider ollama --model llama3.1:8b    # local, no key

    # Skip generation entirely (no LLM call, no API key required):
    python ask.py "..." --retrieve-only

Requires:
- ANTHROPIC_API_KEY in .env  (only for the default --provider anthropic)
- data/index/bge.faiss + data/index/lookup.jsonl (built via scripts/04_build_index.py)
- BGE model auto-downloaded on first query (~1.3GB)
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from scripts.bge_embed import encode_queries
from scripts.llm_backends import available_providers, get_backend

load_dotenv()

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "data" / "index" / "bge.faiss"
LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"

DEFAULT_K = 8
DEFAULT_PROVIDER = "anthropic"

SYSTEM_PROMPT = """You are Cyber-Witten, an AI grounded exclusively in Edward Witten's papers. Answer using ONLY the passages provided.

Rules:
- Cite every non-trivial claim inline using the exact passage IDs, e.g. [hep-th/9112056], [1106.4789], or [inspire:193975].
- If the passages don't contain enough to answer, say so explicitly. Do NOT fall back to general knowledge or invent details.
- Preserve Witten's terseness — be precise, don't pad with generic textbook background.
- Math: use LaTeX (inline $...$ or display $$...$$).
- If the passages span different years and the view evolved, note that.
- If you must speculate beyond the passages, mark it clearly: "[outside corpus]".
"""


def retrieve(question, k):
    q_emb = encode_queries([question])

    import faiss

    index = faiss.read_index(str(INDEX_FILE))
    lookup = [json.loads(l) for l in LOOKUP_FILE.open()]

    scores, idxs = index.search(q_emb, k)
    return [(float(scores[0][j]), lookup[idxs[0][j]]) for j in range(k)]


def format_passages(passages):
    parts = []
    for j, (score, p) in enumerate(passages, 1):
        parts.append(
            f"--- Passage {j} (sim={score:.3f}) ---\n"
            f"Title: {p['title']}\n"
            f"ID: {p['arxiv_id']} ({p['year']})\n"
            f"Text:\n{p['text']}\n"
        )
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("-k", type=int, default=DEFAULT_K, help="Top-K passages")
    ap.add_argument("--show-passages", action="store_true",
                    help="Print retrieved passages before the answer")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER,
                    choices=available_providers(),
                    help=f"LLM backend (default: {DEFAULT_PROVIDER})")
    ap.add_argument("--model", default=None,
                    help="Model name override (default: backend's pick)")
    ap.add_argument("--retrieve-only", action="store_true",
                    help="Print top-K passages and exit — no LLM call, no key needed")
    ap.add_argument("--guardrail", action="store_true",
                    help="Enforce citation grounding: reject and regenerate an answer that "
                         "cites a paper not in the retrieved set (see scripts/guardrail.py)")
    args = ap.parse_args()

    if not args.question:
        args.question = input("Ask Cyber-Witten: ").strip()
    if not args.question:
        return

    # Resolve backend BEFORE the expensive retrieval — fails fast on missing key.
    backend = None
    if not args.retrieve_only:
        backend = get_backend(args.provider, args.model)

    passages = retrieve(args.question, args.k)
    context = format_passages(passages)

    if args.show_passages or args.retrieve_only:
        print("=" * 70)
        print("RETRIEVED PASSAGES")
        print("=" * 70)
        print(context)
        print("=" * 70 + "\n")

    if args.retrieve_only:
        return

    user_msg = (
        f"<question>\n{args.question}\n</question>\n\n"
        f"<passages>\n{context}\n</passages>\n\n"
        "Answer the question using only the passages above. Cite each claim."
    )

    report = None
    if args.guardrail:
        from scripts.guardrail import generate_grounded
        retrieved_ids = [p["arxiv_id"] for _, p in passages]
        answer, report = generate_grounded(
            backend, SYSTEM_PROMPT, user_msg, retrieved_ids, max_tokens=2048
        )
    else:
        answer = backend.generate(SYSTEM_PROMPT, user_msg, max_tokens=2048)

    print("=" * 70)
    print(f"CYBER-WITTEN  ({backend.name} / {backend.model})")
    print("=" * 70)
    print(answer)
    if report and not report["grounded"]:
        bad = ", ".join(report["validation"]["invalid_citations"])
        print(f"\n⚠  ungrounded after {report['attempts']} attempts: "
              f"cited {bad} not in the retrieved passages.")
    print()


if __name__ == "__main__":
    main()
