"""Resolve paper TITLES (+ optional year) to corpus arXiv/inspire IDs.

GPT proposes expected_citations as "Title (Year)" strings (it can't be trusted
to remember exact arXiv IDs). This maps them to the real IDs in the index, so
they drop straight into evals/gold/gold_set.jsonl and the validator can check
them.

Authoritative source is data/index/lookup.jsonl — the 289 papers actually in
the FAISS index, including the 44 pre-arXiv `inspire:` papers (which are NOT in
papers.jsonl).

    # Resolve a few titles interactively:
    python -m evals.resolve_citations "Mirror Manifolds And Topological Field Theory (1991)" \
                                      "Anti-de Sitter Space and Holography (1998)"

    # Pipe a list (one title per line):
    pbpaste | python -m evals.resolve_citations --stdin

    # Dry-run over the gold set: show what each title-shaped expected_citation
    # would resolve to (IDs already present are left untouched):
    python -m evals.resolve_citations --gold evals/gold/gold_set.jsonl

    # Same, but rewrite the file in place (high-confidence matches only):
    python -m evals.resolve_citations --gold evals/gold/gold_set.jsonl --write
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.validator import is_paper_id  # noqa: E402

LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"
GOLD_DEFAULT = ROOT / "evals" / "gold" / "gold_set.jsonl"

HIGH = 0.90   # >= : confident match
REVIEW = 0.60  # >= : plausible, eyeball it; below: no match

_YEAR_RE = re.compile(r"\(?\b(19|20)\d{2}\b\)?")
_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    t = title.lower()
    t = t.replace("\\", " ")           # LaTeX commands
    t = _NORM_RE.sub(" ", t)           # braces, punctuation -> space
    return " ".join(t.split())


def split_year(query: str) -> tuple[str, str | None]:
    """Pull a trailing/inline 4-digit year out of a query, return (title, year)."""
    q = re.sub(r"\(verify\)", "", query, flags=re.IGNORECASE).strip()
    years = _YEAR_RE.findall(q)
    year = None
    if years:
        m = list(_YEAR_RE.finditer(q))[-1]
        year = m.group(0).strip("()")
        q = (q[: m.start()] + q[m.end():]).strip()
    return q, year


def load_papers() -> list[dict]:
    """Unique papers from the index: {id, title, year, norm}."""
    seen: dict[str, dict] = {}
    with LOOKUP_FILE.open() as f:
        for line in f:
            d = json.loads(line)
            aid = d["arxiv_id"]
            if aid not in seen:
                seen[aid] = {
                    "id": aid,
                    "title": d.get("title", ""),
                    "year": str(d.get("year", "")),
                    "norm": normalize_title(d.get("title", "")),
                }
    return list(seen.values())


def resolve(query: str, papers: list[dict], top_n: int = 3) -> list[dict]:
    """Best paper matches for a title query, scored 0-1, year-aware."""
    title, year = split_year(query)
    qnorm = normalize_title(title)

    scored = []
    for p in papers:
        ratio = SequenceMatcher(None, qnorm, p["norm"]).ratio()
        # Small nudge for an exact year match; small penalty for a clear mismatch.
        if year and p["year"]:
            ratio += 0.05 if p["year"] == year else -0.05
        scored.append((min(ratio, 1.0), p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"score": round(s, 3), "id": p["id"], "title": p["title"], "year": p["year"]}
        for s, p in scored[:top_n]
    ]


def _conf(score: float) -> str:
    return "OK " if score >= HIGH else "?? " if score >= REVIEW else "XX "


def _print_query(query: str, matches: list[dict]) -> None:
    print(f"\n» {query}")
    for m in matches:
        print(f"   {_conf(m['score'])}{m['score']:.2f}  {m['id']:<18} ({m['year']}) {m['title']}")


def run_interactive(queries: list[str]) -> None:
    papers = load_papers()
    for q in queries:
        if q.strip():
            _print_query(q, resolve(q, papers))
    print(
        f"\nLegend: OK >= {HIGH:.2f} confident | ?? >= {REVIEW:.2f} eyeball it | XX no match\n"
        "Paste the chosen IDs into expected_citations, or use --gold --write to auto-fill."
    )


def run_gold(gold_path: Path, write: bool) -> None:
    papers = load_papers()
    rows = [json.loads(l) for l in gold_path.open() if l.strip()]

    changed = 0
    unresolved: list[str] = []
    for row in rows:
        cites = row.get("expected_citations", [])
        new_cites = []
        for c in cites:
            if is_paper_id(c):
                new_cites.append(c)
                continue
            matches = resolve(c, papers)
            best = matches[0] if matches else None
            if best and best["score"] >= HIGH:
                print(f"[{row.get('qid','?')}] {c!r}\n    -> {best['id']}  ({best['year']}) {best['title']}  [{best['score']:.2f}]")
                new_cites.append(best["id"])
                changed += 1
            else:
                cand = f" best: {best['id']} [{best['score']:.2f}]" if best else ""
                print(f"[{row.get('qid','?')}] {c!r}\n    -> UNRESOLVED (needs review){cand}")
                unresolved.append(f"{row.get('qid','?')}: {c}")
                new_cites.append(c)  # leave as-is for manual fixing
        row["expected_citations"] = new_cites

    print(f"\n{changed} title(s) resolved, {len(unresolved)} left for review.")
    if unresolved:
        print("Review:\n  " + "\n  ".join(unresolved))

    if write and changed:
        with gold_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote resolved IDs -> {gold_path.relative_to(ROOT)} (git diff to review).")
    elif changed:
        print("\nDry run. Re-run with --write to apply the high-confidence replacements.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("titles", nargs="*", help='Title queries, e.g. "Fivebranes and Knots (2011)"')
    ap.add_argument("--stdin", action="store_true", help="Read title queries from stdin, one per line")
    ap.add_argument("--gold", type=Path, help="Resolve title-shaped expected_citations in this gold file")
    ap.add_argument("--write", action="store_true", help="With --gold: rewrite the file in place")
    args = ap.parse_args()

    if not LOOKUP_FILE.exists():
        raise SystemExit(f"Missing {LOOKUP_FILE} — build the index first (scripts 01-04).")

    if args.gold:
        run_gold(args.gold, args.write)
    else:
        queries = args.titles
        if args.stdin:
            queries += [l.rstrip("\n") for l in sys.stdin]
        if not queries:
            ap.error("give title(s) as arguments, or --stdin, or --gold <file>")
        run_interactive(queries)


if __name__ == "__main__":
    main()
