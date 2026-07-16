"""Calibrate the pre-generation refusal threshold on a proper probe set.

For every question (in-corpus gold + out-of-corpus probes), retrieve with the
cross-encoder reranker and record the BEST passage score. Print the two score
distributions, sweep candidate thresholds, and recommend one.

Also flags "suspect probes": an out-of-corpus probe whose best score lands in
the in-corpus range is probably answerable from the corpus after all (bad
probe), or has found a real coverage surprise — either way it needs a
physicist's review, not silent inclusion in the calibration.

    python -m evals.calibrate_refusal
    python -m evals.calibrate_refusal --gold evals/gold/gold_set.jsonl --probes evals/gold/probes.jsonl -k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.run_eval import build_retriever, load_jsonl  # noqa: E402

GOLD_DEFAULT = ROOT / "evals" / "gold" / "gold_set.jsonl"
PROBES_DEFAULT = ROOT / "evals" / "gold" / "probes.jsonl"


def sweep(in_scores: list[float], out_scores: list[float]) -> list[tuple[float, int, int]]:
    """Candidate thresholds (midpoints between adjacent distinct scores) with
    (threshold, false_refusals, missed_refusals). false = in-corpus below t;
    missed = out-of-corpus at/above t."""
    pts = sorted(set(in_scores + out_scores))
    cands = [pts[0] - 0.5] + [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 0.5]
    rows = []
    for t in cands:
        false_ref = sum(1 for s in in_scores if s < t)
        missed = sum(1 for s in out_scores if s >= t)
        rows.append((t, false_ref, missed))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    ap.add_argument("--probes", type=Path, default=PROBES_DEFAULT)
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    questions = load_jsonl(args.gold) + load_jsonl(args.probes)
    retrieve = build_retriever(args.k, use_rerank=True)

    scored = []
    for q in questions:
        hits = retrieve(q["question"])
        best = hits[0][0] if hits else float("-inf")
        fair = q.get("gate_calibration", True)
        scored.append({"qid": q["qid"], "type": q["type"], "best": round(best, 3), "fair": fair})
        tag = "OUT" if q["type"] == "out_of_corpus" else "in "
        excl = "" if fair else "  [EXCL: gate_calibration=false]"
        print(f"  {best:7.3f}  {tag:<3}  {q['qid']}{excl}", flush=True)

    in_scores = [r["best"] for r in scored if r["type"] != "out_of_corpus"]
    out_all = [r["best"] for r in scored if r["type"] == "out_of_corpus"]
    # Fair subset: probes where the corpus genuinely lacks answer material, so
    # a topical-coverage gate CAN in principle catch them. Probes marked
    # gate_calibration: false (mixed questions, corpus-covered topics, premise-
    # type falseness) are scored and listed but kept out of the sweep.
    out_fair = [r["best"] for r in scored if r["type"] == "out_of_corpus" and r["fair"]]

    print("\n" + "=" * 64)
    print(f"in-corpus        n={len(in_scores):2d}  min={min(in_scores):6.3f}  max={max(in_scores):6.3f}")
    print(f"out probes (all) n={len(out_all):2d}  min={min(out_all):6.3f}  max={max(out_all):6.3f}")
    print(f"out probes (fair)n={len(out_fair):2d}  min={min(out_fair):6.3f}  max={max(out_fair):6.3f}")

    min_in, max_out = min(in_scores), max(out_fair)
    print("\nthreshold sweep on the FAIR subset (false refusals = answerable "
          "question refused; missed = fair probe let through):")
    shown = set()
    for t, fr, ms in sweep(in_scores, out_fair):
        key = (fr, ms)
        if key in shown:  # only show the frontier, not every midpoint
            continue
        shown.add(key)
        print(f"  t={t:7.3f}   false_refusals={fr:2d}   missed={ms:2d}")

    print()
    if max_out < min_in:
        rec = (max_out + min_in) / 2
        print(f"fair subset separates: max(out)={max_out:.3f} < min(in)={min_in:.3f}, "
              f"margin={min_in - max_out:.3f}")
        print(f"max-margin midpoint = {rec:.2f} (default REFUSAL_THRESHOLD stays "
              f"conservative; tune via --refusal-threshold)")
    else:
        zero_false = max([t for t, fr, _ in sweep(in_scores, out_fair) if fr == 0])
        print(f"fair subset OVERLAPS: max(out)={max_out:.3f} >= min(in)={min_in:.3f}")
        print(f"largest zero-false-refusal threshold = {zero_false:.3f}")

    suspects = [r for r in scored if r["type"] == "out_of_corpus" and r["fair"]
                and r["best"] >= min(in_scores)]
    if suspects:
        print("\nSUSPECT PROBES (fair probe scoring >= in-corpus minimum — possibly "
              "answerable from the corpus; needs physics review):")
        for r in sorted(suspects, key=lambda x: -x["best"]):
            print(f"  {r['best']:7.3f}  {r['qid']}")


if __name__ == "__main__":
    main()
