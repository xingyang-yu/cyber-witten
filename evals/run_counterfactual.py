"""Run the retrieval-aware counterfactual set (Experiment 1).

The whole point: for each pair, retrieve the shared passages P ONCE (from the
supported question, with the production rerank path), then run BOTH arms against
that identical P. Passages are held fixed, so the only thing that varies between
the arms is whether the premise is supported. That isolates the mechanism behind
"retrieval destroys refusal".

    # Base model, both naked and guardrailed, over all pairs x both arms:
    python -m evals.run_counterfactual --provider ollama --model qwen2.5:7b

    # The distilled model:
    python -m evals.run_counterfactual --provider ollama --model cyber-witten-7b

    # Re-render tables from a run file (no LLM), after filling human scores:
    python -m evals.run_counterfactual --report evals/results/cf_run_YYYYMMDD-HHMMSS.jsonl

Emits records in the SAME schema as run_eval (qid/type/condition/auto/human/...),
plus pair_id/arm/shared_ids, so evals.rubric and the human-scoring flow work
unchanged. refusal gate (prerefusal) is computed once per pair and copied to both
arms, to make visible that the gate scores passages only and cannot tell the arms
apart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.rubric import aggregate, blank_human_scores, render_failure_lines, render_markdown  # noqa: E402
from evals.run_eval import build_retriever, load_jsonl, write_jsonl  # noqa: E402
from evals.validator import validate_citations  # noqa: E402

CF_DEFAULT = ROOT / "evals" / "gold" / "counterfactual.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"


def run(provider: str, model: str | None, pairs: list[dict], k: int, max_tokens: int) -> list[dict]:
    from ask import SYSTEM_PROMPT, format_passages
    from scripts.guardrail import generate_grounded
    from scripts.llm_backends import get_backend
    from scripts.rerank import REFUSAL_THRESHOLD, refusal_check

    backend = get_backend(provider, model)
    retrieve = build_retriever(k, use_rerank=True)  # production path = rerank on

    records: list[dict] = []
    for pair in pairs:
        # Retrieve the shared passages ONCE, from the supported question.
        passages = retrieve(pair["retrieval_query"])
        shared_ids = [p["arxiv_id"] for _, p in passages]
        should_refuse, best = refusal_check(passages)
        prerefusal = {"best_score": round(best, 3), "triggered": should_refuse,
                      "threshold": REFUSAL_THRESHOLD}
        passages_block = format_passages(passages)

        for arm in ("supported", "unsupported"):
            spec = pair[arm]
            typ = "in_corpus" if arm == "supported" else "out_of_corpus"
            expected = spec.get("expected_citations", [])
            user_msg = (
                f"<question>\n{spec['question']}\n</question>\n\n"
                f"<passages>\n{passages_block}\n</passages>\n\n"
                "Answer the question using only the passages above. Cite each claim."
            )
            for condition in ("rag", "rag+guard"):
                cond = f"{arm}:{condition}"
                print(f"  [{backend.name}/{model}] {pair['pair_id']} {cond} ...", end="", flush=True)
                rec = {
                    "qid": f"{pair['pair_id']}--{arm}",
                    "pair_id": pair["pair_id"],
                    "arm": arm,
                    "twin_type": pair["unsupported"].get("twin_type") if arm == "unsupported" else None,
                    "question": spec["question"],
                    "type": typ,
                    "backend": backend.name,
                    "condition": condition,
                    "model": backend.model,
                    "k": k,
                    "retrieved_ids": shared_ids,
                    "shared_ids": shared_ids,  # explicit: identical across the pair
                    "expected_citations": expected,
                    "prerefusal": prerefusal,
                    "human": blank_human_scores(),
                }
                t0 = time.time()
                try:
                    if condition == "rag+guard":
                        answer, gr = generate_grounded(
                            backend, SYSTEM_PROMPT, user_msg, shared_ids, max_tokens=max_tokens,
                        )
                        rec["guardrail"] = {"grounded": gr["grounded"], "problem": gr["problem"],
                                            "attempts": gr["attempts"]}
                    else:
                        answer = backend.generate(SYSTEM_PROMPT, user_msg, max_tokens=max_tokens)
                    rec["answer"] = answer
                    rec["auto"] = validate_citations(answer, shared_ids, expected)
                    rec["latency_s"] = round(time.time() - t0, 2)
                    print(f" {rec['latency_s']}s")
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    print(f" ERROR: {rec['error']}")
                records.append(rec)
    return records


def report(records: list[dict]) -> None:
    # Split by arm so the two behaviours don't average together.
    for arm in ("supported", "unsupported"):
        sub = [r for r in records if r.get("arm") == arm]
        if not sub:
            continue
        print(f"\n## Arm: {arm} ({len(sub)} records)\n")
        print(render_markdown(aggregate(sub)))
    print("\n## Flagged answers\n")
    print(render_failure_lines(records))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--cf", type=Path, default=CF_DEFAULT)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N pairs")
    ap.add_argument("--include-risky", action="store_true",
                    help="Also run pairs the vetting marked 'risky' (excluded by default)")
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
            print(f"Excluding {len(dropped)} risky pair(s): {dropped} (use --include-risky to keep)")
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Running {len(pairs)} pairs x 2 arms x [rag, rag+guard] on {args.provider}/{args.model}\n")
    records = run(args.provider, args.model, pairs, args.k, args.max_tokens)

    tag = (args.model or args.provider).replace(":", "").replace("/", "-")
    out = args.out.resolve() if args.out else RESULTS_DIR / f"cf_{tag}_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    write_jsonl(out, records)
    print(f"\nWrote {len(records)} records -> {out.relative_to(ROOT)}")
    report(records)
    print(f"\nNext: fill human scores (supported->correctness/faithfulness, "
          f"unsupported->refusal_ok) in\n  {out.relative_to(ROOT)}\n"
          f"then:  python -m evals.run_counterfactual --report {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
