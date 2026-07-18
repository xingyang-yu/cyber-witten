"""Generate distillation training data: (question, passages, grounded answer).

Teacher pipeline, one sample at a time:
  1. Sample a corpus chunk (one per paper per run, min length, seeded RNG).
  2. QUESTION model writes a self-contained physics question that chunk answers.
  3. Retrieve top-K passages for the question via the production retriever
     (same BGE-large index and k as ask.py).
  4. ANSWER model answers under the strict cite-or-fail system prompt.
  5. Gate the sample: the answer must pass the citation validator (grounded,
     at least one citation, no fabrication). Rejected samples are logged with
     a reason, not silently dropped. Whether the SOURCE paper made top-K is
     recorded as metadata but does NOT gate: the training signal is "answer
     from what was retrieved", not "answer from the seed". (Pilot: with real
     questions 47/50 stay anchored to their seed paper anyway; an earlier
     "98% drift" reading was an artifact of empty questions from a thinking-
     model token-budget bug, not corpus topology.)

The eval gold set (evals/gold/) is NEVER used here — those questions stay held
out for measuring the distilled model. A similarity guard additionally skips
any generated question sharing 6+ content words with a gold question.

Token usage is tracked from the API's usage fields and printed at the end, so
the 50-sample pilot gives an exact per-sample cost to extrapolate from.

    python -m scripts.distill_gen --limit 50
    python -m scripts.distill_gen --limit 2000 --out data/distill/samples_v1.jsonl

Requires DEEPSEEK_API_KEY in .env (or --question-model/--answer-model pointing
at another OpenAI-compatible provider via --base-url).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

QUESTION_MODEL = "deepseek-v4-flash"   # cheap: writing a question is easy
ANSWER_MODEL = "deepseek-v4-pro"       # the actual teaching signal
BASE_URL = "https://api.deepseek.com/v1"
K = 8
MIN_CHUNK_CHARS = 600
SEED = 20260716

QUESTION_PROMPT = """You will be shown a passage from a physics paper by Edward Witten. Write ONE self-contained research-level physics question that this passage substantially answers.

Rules:
- The question must stand alone: no "in this passage", no "according to the text".
- Ask about the physics (mechanism, relation, construction, implication), not about the paper's metadata.
- One sentence, ending with a question mark. Output ONLY the question.

Passage (from "{title}", {year}):
{text}"""


def load_gold_word_sets() -> list[set]:
    stop = set("the a an of in on for to and or is are was were what how why does do did with by from between".split())
    sets = []
    for path in [ROOT / "evals/gold/gold_set.jsonl", ROOT / "evals/gold/probes.jsonl"]:
        for line in path.open():
            q = json.loads(line)["question"].lower()
            words = {w for w in re.findall(r"[a-z][a-z-]+", q) if w not in stop}
            sets.append(words)
    return sets


def too_close_to_gold(question: str, gold_sets: list[set]) -> bool:
    words = set(re.findall(r"[a-z][a-z-]+", question.lower()))
    return any(len(words & g) >= 6 for g in gold_sets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "data/distill/samples_pilot.jsonl")
    ap.add_argument("--question-model", default=QUESTION_MODEL)
    ap.add_argument("--answer-model", default=ANSWER_MODEL)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fresh", action="store_true",
                    help="Overwrite --out instead of resuming (default: resume — skip chunks "
                         "already in the file and append until --limit total kept samples)")
    args = ap.parse_args()

    import os

    from openai import OpenAI  # raw client: the usage fields feed cost accounting

    from ask import SYSTEM_PROMPT, format_passages
    from evals.run_eval import build_retriever
    from evals.validator import validate_citations

    client = OpenAI(base_url=args.base_url, api_key=os.environ["DEEPSEEK_API_KEY"])

    usage = {"prompt": 0, "completion": 0, "calls": 0}

    def chat(model: str, system: str | None, user: str, max_tokens: int = 1200) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        resp = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=messages)
        usage["prompt"] += resp.usage.prompt_tokens
        usage["completion"] += resp.usage.completion_tokens
        usage["calls"] += 1
        return resp.choices[0].message.content.strip()

    # --- build the work list: interleaved passes over papers, a fresh chunk
    # per paper per pass (the corpus has only ~293 papers, so a >293-sample run
    # must revisit papers with different chunks), reproducible under --seed ---
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in (ROOT / "data/index/lookup.jsonl").open()]
    by_paper: dict[str, list[dict]] = {}
    for r in rows:
        if len(r["text"]) >= MIN_CHUNK_CHARS:
            by_paper.setdefault(r["arxiv_id"], []).append(r)
    papers = sorted(by_paper)
    rng.shuffle(papers)
    for p in papers:
        rng.shuffle(by_paper[p])
    max_pass = max(len(v) for v in by_paper.values())
    work = [(p, by_paper[p][i]) for i in range(max_pass) for p in papers if i < len(by_paper[p])]

    # --- resume: skip chunks already in the output, append until --limit total ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    used_chunks: set[str] = set()
    kept = 0
    if args.out.exists() and not args.fresh:
        for line in args.out.open():
            rec = json.loads(line)
            used_chunks.add(rec.get("source_chunk_id", ""))
            kept += 1
        if kept:
            print(f"resuming: {kept} samples already in {args.out.name}, "
                  f"target {args.limit} total", flush=True)

    gold_sets = load_gold_word_sets()
    retrieve = build_retriever(K)

    rejected = {"empty-question": 0, "ungrounded": 0, "uncited": 0, "gold-overlap": 0, "error": 0}
    consecutive_errors = 0
    with args.out.open("w" if (args.fresh or not used_chunks) else "a") as fout:
        for paper, chunk in work:
            if kept >= args.limit:
                break
            if chunk["chunk_id"] in used_chunks:
                continue
            try:
                question = chat(args.question_model, None, QUESTION_PROMPT.format(
                    title=chunk["title"], year=chunk["year"], text=chunk["text"][:4000]), max_tokens=2048)
                question = re.sub(r"^(?:Question\s*:)\s*", "", question.strip().strip('"'))
                if not question.endswith("?") or len(question) < 20:
                    rejected["empty-question"] += 1
                    continue
                if too_close_to_gold(question, gold_sets):
                    rejected["gold-overlap"] += 1
                    continue

                passages = retrieve(question)
                rids = [p["arxiv_id"] for _, p in passages]

                user_msg = (f"<question>\n{question}\n</question>\n\n"
                            f"<passages>\n{format_passages(passages)}\n</passages>\n\n"
                            "Answer the question using only the passages above. Cite each claim.")
                answer = chat(args.answer_model, SYSTEM_PROMPT, user_msg, max_tokens=3000)

                v = validate_citations(answer, rids)
                if v["grounding_violation"]:
                    rejected["ungrounded"] += 1
                    continue
                if v["uncited"]:
                    rejected["uncited"] += 1
                    continue

                fout.write(json.dumps({
                    "question": question,
                    "source_paper": paper,
                    "source_chunk_id": chunk["chunk_id"],
                    "source_in_topk": paper in rids,
                    "retrieved_ids": rids,
                    "passages": [{"score": s, **{k: p[k] for k in ("arxiv_id", "title", "year", "text")}}
                                 for s, p in passages],
                    "answer": answer,
                    "cited_ids": v["cited_ids"],
                    "question_model": args.question_model,
                    "answer_model": args.answer_model,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                kept += 1
                print(f"  [{kept}/{args.limit}] {paper}: {question[:80]}", flush=True)
            except Exception as exc:
                rejected["error"] += 1
                consecutive_errors += 1
                print(f"  ! {paper}: {type(exc).__name__}: {exc}", flush=True)
                if consecutive_errors >= 10:
                    print("ABORT: 10 consecutive errors (exhausted balance or API outage). "
                          "Top up / wait, then rerun the same command — resume will continue.",
                          flush=True)
                    break
                continue
            consecutive_errors = 0

    total_rej = sum(rejected.values())
    print(f"\nkept {kept}, rejected {total_rej} {rejected}")
    print(f"tokens: {usage['prompt']:,} prompt + {usage['completion']:,} completion "
          f"over {usage['calls']} calls")
    if kept:
        print(f"per kept sample: ~{usage['prompt'] // kept:,} prompt + {usage['completion'] // kept:,} completion tokens")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
