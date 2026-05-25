"""Ask Cyber-Witten a question, grounded in Witten's paper corpus.

Usage:
    python ask.py "What is the relation between Chern-Simons and the Jones polynomial?"
    python ask.py "..." --show-passages    # also print retrieved passages
    python ask.py "..." -k 12              # retrieve more passages

Requires:
- ANTHROPIC_API_KEY in .env
- data/index/bge.faiss + data/index/lookup.jsonl (built via scripts/04_build_index.py)
- BGE model auto-downloaded on first query (~1.3GB)
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from scripts.bge_embed import encode_queries

load_dotenv()

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "data" / "index" / "bge.faiss"
LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"

LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_K = 8

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
    ap.add_argument("--show-passages", action="store_true")
    args = ap.parse_args()

    if not args.question:
        args.question = input("Ask Cyber-Witten: ").strip()
    if not args.question:
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set in .env")

    passages = retrieve(args.question, args.k)
    context = format_passages(passages)

    if args.show_passages:
        print("=" * 70)
        print("RETRIEVED PASSAGES")
        print("=" * 70)
        print(context)
        print("=" * 70 + "\n")

    user_msg = (
        f"<question>\n{args.question}\n</question>\n\n"
        f"<passages>\n{context}\n</passages>\n\n"
        "Answer the question using only the passages above. Cite each claim."
    )

    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=LLM_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    print("\n" + "=" * 70)
    print("CYBER-WITTEN")
    print("=" * 70)
    print(resp.content[0].text)
    print()


if __name__ == "__main__":
    main()
