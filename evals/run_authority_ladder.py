"""Experiment 2: passage-authority dose-response for refusal.

Experiment 1 showed that fixing the retrieved passages to a supported twin's
context collapses refusal on a false premise. This isolates *why*: it holds the
false-premise QUESTION fixed and varies only what sits in the retrieval slot,
along a ladder of increasing apparent relevance to the false premise. Every rung
is real Witten text obtained by a retrieval operation, so nothing is hand-crafted:

    empty        no passages at all (closed-book control; refusal should be highest)
    gold         passages retrieved from the SUPPORTED question (real, on-topic,
                 but about the TRUE claim -- the Experiment 1 condition)
    adjacent     passages retrieved from a SIBLING pair's question (real Witten
                 text, authoritative, but wrong topic)
    adversarial  passages retrieved from the FALSE-PREMISE question itself (the
                 most premise-matching real text the retriever can surface -- what
                 a naive RAG pipeline would actually feed the model)

Hypothesis: refusal degrades monotonically as the passages look more like they
are about the false premise, lowest at `adversarial`. That would show it is not
"any passages" that kill refusal but passages that pattern-match the premise.

Naked `rag` only (the guardrail's effect is characterized in Experiment 1; here
we want the clean dose-response). Same 10 non-risky pairs as the counterfactual
set, same strict refusal_ok scoring afterwards.

    python -m evals.run_authority_ladder --provider ollama --model qwen2.5:7b
    python -m evals.run_authority_ladder --provider ollama --model cyber-witten-7b
    python -m evals.run_authority_ladder --report evals/results/ladder_<...>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.rubric import blank_human_scores  # noqa: E402
from evals.run_eval import build_retriever, load_jsonl, write_jsonl  # noqa: E402
from evals.validator import validate_citations  # noqa: E402

CF_DEFAULT = ROOT / "evals" / "gold" / "counterfactual.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

LADDER = ["empty", "gold", "adjacent", "adversarial"]


def build_passage_sets(pair, sibling, retrieve, k):
    """Return {rung: (passages, source_desc)} for one pair.

    passages is a list of (score, passage_dict), matching build_retriever's
    output; empty rung is an empty list.
    """
    sets = {}
    sets["empty"] = ([], "none")
    sets["gold"] = (retrieve(pair["retrieval_query"]), f"supported-query:{pair['pair_id']}")
    sets["adjacent"] = (retrieve(sibling["retrieval_query"]), f"sibling-query:{sibling['pair_id']}")
    sets["adversarial"] = (retrieve(pair["unsupported"]["question"]), f"false-premise-query:{pair['pair_id']}")
    return sets


def run(provider, model, pairs, k, max_tokens, rungs):
    from ask import SYSTEM_PROMPT, format_passages
    from scripts.llm_backends import get_backend
    from scripts.rerank import REFUSAL_THRESHOLD, refusal_check

    backend = get_backend(provider, model)
    retrieve = build_retriever(k, use_rerank=True)  # production path = rerank on

    records = []
    for i, pair in enumerate(pairs):
        sibling = pairs[(i + 1) % len(pairs)]  # next pair, cyclically -> real Witten, wrong topic
        sets = build_passage_sets(pair, sibling, retrieve, k)
        question = pair["unsupported"]["question"]  # the FALSE-PREMISE question, fixed across rungs
        expected = pair["unsupported"].get("expected_citations", [])  # correct answer supports nothing

        for rung in rungs:
            passages, source = sets[rung]
            rung_ids = [p["arxiv_id"] for _, p in passages]
            passages_block = format_passages(passages) if passages else "(no passages retrieved)"
            prerefusal = None
            if passages:
                should_refuse, best = refusal_check(passages)
                prerefusal = {"best_score": round(best, 3), "triggered": should_refuse,
                              "threshold": REFUSAL_THRESHOLD}

            user_msg = (
                f"<question>\n{question}\n</question>\n\n"
                f"<passages>\n{passages_block}\n</passages>\n\n"
                "Answer the question using only the passages above. Cite each claim."
            )
            print(f"  [{backend.name}/{model}] {pair['pair_id']} {rung} ...", end="", flush=True)
            rec = {
                "qid": f"{pair['pair_id']}--{rung}",
                "pair_id": pair["pair_id"],
                "arm": "unsupported",
                "rung": rung,
                "twin_type": pair["unsupported"].get("twin_type"),
                "question": question,
                "type": "out_of_corpus",
                "backend": backend.name,
                "condition": "rag",
                "model": backend.model,
                "k": k,
                "retrieved_ids": rung_ids,
                "passage_source": source,
                "expected_citations": expected,
                "prerefusal": prerefusal,
                "human": blank_human_scores(),
            }
            t0 = time.time()
            try:
                answer = backend.generate(SYSTEM_PROMPT, user_msg, max_tokens=max_tokens)
                rec["answer"] = answer
                rec["auto"] = validate_citations(answer, rung_ids, expected)
                rec["latency_s"] = round(time.time() - t0, 2)
                print(f" {rec['latency_s']}s")
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                print(f" ERROR: {rec['error']}")
            records.append(rec)
    return records


def report(records):
    """Dose-response: mean refusal_ok by rung (needs human refusal_ok filled)."""
    by = defaultdict(list)
    for r in records:
        ro = r.get("human", {}).get("refusal_ok")
        if ro is not None:
            by[(r.get("model", "?"), r["rung"])].append(ro)
    models = sorted({m for m, _ in by})
    if not by:
        print("No refusal_ok scores yet. Fill human.refusal_ok (0/1/2) then re-run --report.")
        # still show what ran
        counts = defaultdict(int)
        for r in records:
            counts[(r.get("model", "?"), r["rung"])] += 1
        print("\nGenerated (unscored):")
        for m in sorted({m for m, _ in counts}):
            row = "  ".join(f"{rung}={counts[(m, rung)]}" for rung in LADDER)
            print(f"  {m:22} {row}")
        return
    print(f"\n{'model':24}" + "".join(f"{rung:>13}" for rung in LADDER))
    print("-" * (24 + 13 * len(LADDER)))
    for m in models:
        cells = []
        for rung in LADDER:
            vals = by.get((m, rung), [])
            cells.append(f"{sum(vals)/len(vals):.2f} (n={len(vals)})" if vals else "-")
        print(f"{m:24}" + "".join(f"{c:>13}" for c in cells))
    print("\nrefusal_ok: 0=none 1=hedged 2=explicit (strict convention). "
          "Hypothesis: empty >= adjacent >= gold >= adversarial.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--cf", type=Path, default=CF_DEFAULT)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N pairs")
    ap.add_argument("--rungs", nargs="+", default=LADDER, choices=LADDER, help="Which rungs to run")
    ap.add_argument("--include-risky", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    if args.report:
        report(load_jsonl(args.report))
        return

    pairs = load_jsonl(args.cf)
    if not args.include_risky:
        dropped = [p["pair_id"] for p in pairs if p.get("verdict") == "risky"]
        pairs = [p for p in pairs if p.get("verdict") != "risky"]
        if dropped:
            print(f"Excluding {len(dropped)} risky pair(s): {dropped}")
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Running {len(pairs)} pairs x {len(args.rungs)} rungs {args.rungs} "
          f"(naked rag) on {args.provider}/{args.model}\n")
    records = run(args.provider, args.model, pairs, args.k, args.max_tokens, args.rungs)

    tag = (args.model or args.provider).replace(":", "").replace("/", "-")
    out = args.out.resolve() if args.out else RESULTS_DIR / f"ladder_{tag}_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    write_jsonl(out, records)
    print(f"\nWrote {len(records)} records -> {out.relative_to(ROOT)}")
    report(records)
    print(f"\nNext: fill human.refusal_ok (unsupported arm, strict) in\n  {out.relative_to(ROOT)}\n"
          f"then:  python -m evals.run_authority_ladder --report {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
