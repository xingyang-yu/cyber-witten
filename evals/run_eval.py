"""Run the Cyber-Witten grounding eval across one or more LLM backends.

    # Generate answers for every gold question on two backends and score the
    # automated half (grounding, recall). Writes a timestamped run file.
    python -m evals.run_eval --providers anthropic,ollama

    # ABLATION: also run each backend closed-book (no retrieval) as a no-RAG
    # baseline, so the table shows RAG vs no-RAG for the SAME model — the
    # cleanest demonstration of what grounding buys you.
    python -m evals.run_eval --providers anthropic --baseline

    python -m evals.run_eval --providers anthropic -k 12 --limit 5
    python -m evals.run_eval --providers ollama --gold evals/gold/gold_set.jsonl

    # No LLM, no API key (loads only the local BGE model): for each gold
    # question, retrieve top-K and report whether the expected papers showed up.
    # A pre-flight check that the questions are actually answerable from the index.
    python -m evals.run_eval --retrieve-only

    # After you fill in the human scores (correctness/faithfulness/refusal_ok,
    # notes) in the run file, re-render the tables — no LLM calls:
    python -m evals.run_eval --report evals/results/run_YYYYMMDD-HHMMSS.jsonl

Retrieval and the system prompt are imported from ask.py so the eval exercises
exactly the production path. Heavy deps (faiss, torch via BGE) are imported
lazily so --report stays instant.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Native-lib safety: faiss and torch each ship an OpenMP runtime; on macOS
# loading both crashes unless this is set BEFORE either native lib loads. They
# are imported lazily inside functions, so set it at module top. Mirrors
# scripts/bge_embed.py and scripts/healthcheck.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.rubric import (  # noqa: E402
    aggregate,
    blank_human_scores,
    render_failure_lines,
    render_markdown,
)
from evals.validator import validate_citations  # noqa: E402

GOLD_DEFAULT = ROOT / "evals" / "gold" / "gold_set.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_retriever(k: int):
    """Load FAISS index + lookup once and return a retrieve(question) closure.

    Mirrors ask.retrieve but avoids re-reading the 53MB index per question.
    """
    # Load order matters on macOS: torch's OpenMP runtime must initialize BEFORE
    # faiss's, or the two collide in a segfault. bge_embed imports torch lazily,
    # so force it here before faiss. (Verified: torch-then-faiss is stable.)
    from scripts.bge_embed import encode_queries

    import torch  # noqa: F401  # must precede `import faiss`
    import faiss

    index = faiss.read_index(str(ROOT / "data" / "index" / "bge.faiss"))
    lookup = load_jsonl(ROOT / "data" / "index" / "lookup.jsonl")

    def retrieve(question: str):
        q_emb = encode_queries([question])
        scores, idxs = index.search(q_emb, k)
        return [(float(scores[0][j]), lookup[idxs[0][j]]) for j in range(k)]

    return retrieve


# The no-RAG baseline: the model answers from training memory, with no passages.
# This is the "just ask the LLM" condition the project exists to improve on. It
# still asks for citations — so a model that fabricates arXiv IDs from memory is
# caught by the validator (retrieved set is empty, so any citation is ungrounded).
CLOSED_BOOK_SYSTEM = """You are a theoretical physics expert, deeply familiar with the work of Edward Witten. Answer the question as precisely and accurately as you can from your own knowledge.

- Cite the specific paper(s) responsible for each claim, by arXiv ID or title.
- If you are unsure or do not know, say so explicitly rather than guessing.
"""


def run(providers: list[str], gold: list[dict], conditions: list[str], k: int, max_tokens: int) -> list[dict]:
    from ask import SYSTEM_PROMPT, format_passages
    from scripts.llm_backends import get_backend

    # Fail fast on missing keys before the expensive index load.
    backends = [get_backend(p) for p in providers]
    retrieve = build_retriever(k) if "rag" in conditions else None

    records: list[dict] = []
    for q in gold:
        expected = q.get("expected_citations", [])

        # Build each condition's prompt once; shared across all backends.
        prompts: dict[str, tuple[str, str, list[str]]] = {}
        if "rag" in conditions:
            passages = retrieve(q["question"])
            rids = [p["arxiv_id"] for _, p in passages]
            prompts["rag"] = (
                SYSTEM_PROMPT,
                f"<question>\n{q['question']}\n</question>\n\n"
                f"<passages>\n{format_passages(passages)}\n</passages>\n\n"
                "Answer the question using only the passages above. Cite each claim.",
                rids,
            )
        if "closed_book" in conditions:
            prompts["closed_book"] = (
                CLOSED_BOOK_SYSTEM,
                f"<question>\n{q['question']}\n</question>\n\n"
                "Answer the question. Cite the specific papers responsible for each claim.",
                [],  # nothing retrieved — any citation is from memory
            )

        for backend in backends:
            for condition in conditions:
                system, user_msg, retrieved_ids = prompts[condition]
                print(f"  [{backend.name}/{condition}] {q['qid']} ...", end="", flush=True)
                rec = {
                    "qid": q["qid"],
                    "question": q["question"],
                    "type": q.get("type", "in_corpus"),
                    "backend": backend.name,
                    "condition": condition,
                    "model": backend.model,
                    "k": k if condition == "rag" else 0,
                    "retrieved_ids": retrieved_ids,
                    "expected_citations": expected,
                    "human": blank_human_scores(),
                }
                t0 = time.time()
                try:
                    answer = backend.generate(system, user_msg, max_tokens=max_tokens)
                    rec["answer"] = answer
                    rec["auto"] = validate_citations(answer, retrieved_ids, expected)
                    rec["latency_s"] = round(time.time() - t0, 2)
                    flag = " VIOLATION" if rec["auto"]["grounding_violation"] else ""
                    print(f" {rec['latency_s']}s{flag}")
                except Exception as exc:  # one bad call shouldn't sink the whole run
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    print(f" ERROR: {rec['error']}")
                records.append(rec)
    return records


def run_retrieval_preview(gold: list[dict], k: int) -> None:
    """No-LLM pre-flight: retrieve for each gold question and report whether the
    expected papers (and at what rank) actually surface in the top-K. Flags
    in_corpus questions the index can't support, and shows what tempting
    passages out_of_corpus probes pull in."""
    import statistics

    from evals.validator import normalize_id

    retrieve = build_retriever(k)
    recalls: list[float] = []
    missed: list[tuple[str, str]] = []

    for q in gold:
        passages = retrieve(q["question"])
        ranked = [(round(s, 3), p["arxiv_id"]) for s, p in passages]
        norm_rank = [normalize_id(aid) for _, aid in ranked]
        expected = q.get("expected_citations", [])
        typ = q.get("type", "in_corpus")

        print(f"\n» {q['qid']}  [{typ}]")
        if expected:
            hits = 0
            for e in expected:
                ne = normalize_id(e)
                if ne in norm_rank:
                    rank = norm_rank.index(ne) + 1
                    hits += 1
                    print(f"   expect {e:<18} -> rank {rank} (sim {ranked[rank - 1][0]})")
                else:
                    print(f"   expect {e:<18} -> MISS (not in top-{k})")
            recalls.append(hits / len(expected))
            print(f"   retrieval_recall = {hits}/{len(expected)}")
            if hits < len(expected):
                missed.append((q["qid"], f"{hits}/{len(expected)}"))
        else:
            print("   (out_of_corpus — nothing should ground this; top hits that might tempt a model:)")
        top = "  ".join(f"{i + 1}.{aid}({s})" for i, (s, aid) in enumerate(ranked[:5]))
        print(f"   top: {top}")

    print("\n" + "=" * 64)
    if recalls:
        print(f"in_corpus retrieval_recall mean = {statistics.mean(recalls):.2f} over {len(recalls)} questions")
    if missed:
        print("questions where retrieval missed an expected paper (consider higher -k, or rework):")
        for qid, r in missed:
            print(f"   {qid}: {r}")
    else:
        print("every in_corpus question retrieved all its expected papers OK")


def report(records: list[dict]) -> None:
    summary = aggregate(records)
    print("\n## Backend comparison\n")
    print(render_markdown(summary))
    print("\n## Flagged answers (failure taxonomy seed)\n")
    print(render_failure_lines(records))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", default="anthropic",
                    help="Comma-separated backends, e.g. anthropic,openai,ollama")
    ap.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    ap.add_argument("-k", type=int, default=8, help="Top-K passages (matches ask.py default)")
    ap.add_argument("--baseline", action="store_true",
                    help="Also run each backend closed-book (no retrieval) as a no-RAG baseline")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N gold questions")
    ap.add_argument("--out", type=Path, default=None, help="Run file path (default: timestamped)")
    ap.add_argument("--report", type=Path, default=None,
                    help="Skip generation; re-render tables from an existing run file")
    ap.add_argument("--retrieve-only", action="store_true",
                    help="No LLM/API key: retrieve per gold question and report retrieval recall")
    args = ap.parse_args()

    if args.report:
        report(load_jsonl(args.report))
        return

    gold = load_jsonl(args.gold)
    if not gold:
        raise SystemExit(f"No gold questions in {args.gold}. Author some first — see evals/gold/README.md")
    if args.limit:
        gold = gold[: args.limit]

    if args.retrieve_only:
        run_retrieval_preview(gold, args.k)
        return

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    conditions = ["rag"] + (["closed_book"] if args.baseline else [])

    print(f"Running {len(gold)} questions x {len(providers)} backend(s) x {conditions}: {providers}\n")
    records = run(providers, gold, conditions, args.k, args.max_tokens)

    out = args.out or RESULTS_DIR / f"run_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    write_jsonl(out, records)
    print(f"\nWrote {len(records)} records -> {out.relative_to(ROOT)}")
    report(records)
    print(
        f"\nNext: fill in human scores (correctness/faithfulness/refusal_ok/notes) in\n"
        f"  {out.relative_to(ROOT)}\n"
        f"then re-render:  python -m evals.run_eval --report {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
