"""Export distillation samples to chat-format SFT data.

Converts scripts/distill_gen.py output into messages-format JSONL that
trl/axolotl/LLaMA-Factory all accept:

    {"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}

The user message is rebuilt with ask.py's format_passages so the training
distribution EXACTLY matches what the serving path feeds the model — same
system prompt, same passage framing, same closing instruction.

Dedupes by source_chunk_id across all input files (pilot + v1 overlap),
then makes a seeded train/val split.

    python -m scripts.distill_export data/distill/samples_pilot.jsonl data/distill/samples_v1.jsonl
    # -> data/distill/sft_train.jsonl, data/distill/sft_val.jsonl, stats on stdout
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ask import SYSTEM_PROMPT, format_passages  # noqa: E402

VAL_FRACTION = 0.02
SEED = 20260717


def to_messages(rec: dict) -> dict:
    passages = [(p["score"], p) for p in rec["passages"]]
    user = (f"<question>\n{rec['question']}\n</question>\n\n"
            f"<passages>\n{format_passages(passages)}\n</passages>\n\n"
            "Answer the question using only the passages above. Cite each claim.")
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": rec["answer"]},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/distill")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    seen: set[str] = set()
    records = []
    for path in args.inputs:
        n_file = 0
        for line in path.open():
            rec = json.loads(line)
            key = rec.get("source_chunk_id") or f"q:{rec['question']}"
            if key in seen:
                continue
            seen.add(key)
            records.append(to_messages(rec))
            n_file += 1
        print(f"  {path.name}: {n_file} unique samples")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_fraction))
    val, train = records[:n_val], records[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("sft_train.jsonl", train), ("sft_val.jsonl", val)]:
        with (args.out_dir / name).open("w") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Length stats (chars/4 ≈ tokens; the GPU box tokenizes for real).
    lens = sorted(sum(len(m["content"]) for m in r["messages"]) // 4 for r in records)
    pct = lambda q: lens[int(q * (len(lens) - 1))]
    print(f"\ntrain {len(train)} / val {len(val)}  (deduped total {len(records)})")
    print(f"approx tokens per sample: p50={pct(.5):,} p90={pct(.9):,} p99={pct(.99):,} max={lens[-1]:,}")
    print(f"-> choose max_seq_len >= p99 (training script default 8192)")
    print(f"wrote {args.out_dir}/sft_train.jsonl, sft_val.jsonl")


if __name__ == "__main__":
    main()
