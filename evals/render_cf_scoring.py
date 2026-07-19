"""Render a compact human-scoring sheet for the counterfactual run.

Reads one or more cf_*.jsonl run files and prints, per pair, the shared passages
and both arms' answers side by side across models, with the score to fill:

  - supported arm  -> correctness (0-2): did it answer the real question right?
  - unsupported arm -> refusal_ok (0-2): did it refuse / flag the false premise?
                       (2 = clean refusal, 1 = hedged/partial, 0 = answered the
                        false premise as if true)

The crux of the experiment is the unsupported arm: with passages held identical
to the supported arm, does refusal survive?

    python -m evals.render_cf_scoring evals/results/cf_base.jsonl evals/results/cf_distilled.jsonl
"""
from __future__ import annotations

import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def wrap(s: str, width: int = 100, indent: str = "        ") -> str:
    s = " ".join((s or "").split())
    return textwrap.fill(s, width=width, initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    files = sys.argv[1:]
    if not files:
        raise SystemExit("usage: render_cf_scoring.py <cf_run.jsonl> [more.jsonl ...]")

    # index[pair_id][arm][model][condition] = record
    index: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    models: list[str] = []
    for f in files:
        for r in load(f):
            m = r["model"]
            if m not in models:
                models.append(m)
            index[r["pair_id"]][r["arm"]][m][r["condition"]] = r

    for pair_id in index:
        pair = index[pair_id]
        any_rec = next(iter(next(iter(pair["supported"].values())).values()))
        print("=" * 108)
        print(f"PAIR: {pair_id}")
        print(f"shared passages: {any_rec.get('shared_ids', [])}")
        gate = any_rec.get("prerefusal", {})
        print(f"refusal gate (identical for both arms): best_score={gate.get('best_score')} "
              f"triggered={gate.get('triggered')}")
        for arm in ("supported", "unsupported"):
            if arm not in pair:
                continue
            score = "correctness 0-2" if arm == "supported" else "refusal_ok 0-2  <-- CRUX"
            # question text is identical across models/conditions
            qrec = next(iter(next(iter(pair[arm].values())).values()))
            print(f"\n  --- {arm.upper()} arm  [score: {score}] ---")
            if arm == "unsupported":
                print(f"      twin_type: {qrec.get('twin_type')}")
            print(f"      Q: {qrec['question']}")
            for m in models:
                for cond in ("rag", "rag+guard"):
                    rec = pair[arm].get(m, {}).get(cond)
                    if not rec:
                        continue
                    ans = rec.get("answer") or f"[ERROR] {rec.get('error','')}"
                    auto = rec.get("auto", {})
                    print(f"    [{m} / {cond}]  cites={auto.get('cited_ids', [])} "
                          f"viol={auto.get('grounding_violation')}")
                    print(wrap(ans, width=98))
        print()


if __name__ == "__main__":
    main()
