"""Citation provenance: split "ungrounded" cites into real-off-passage vs fabricated.

The validator (`evals/validator.py`) flags a `grounding_violation` whenever the
model cites a paper ID that was not in the retrieved passages. That is the right
*rate* to report, but it lumps two very different failures into one bucket:

  - the model cited a REAL paper it wasn't shown (misremembered provenance), or
  - the model INVENTED an ID that resolves to nothing.

`evals/citation_audit.md` established the distinction by hand for one run. This
script does it reproducibly for any result file(s), classifying every off-passage
("invalid") citation through a cheap-to-expensive ladder:

  1. in-corpus        -- ID is one of the 293 papers in the index (a real Witten
                         paper, just not retrieved this turn).           [local]
  2. impossible date  -- the ID's embedded YYMM is outside any valid arXiv window
                         (e.g. hep-th/8910145, "1989"): a fabrication.    [local]
  3. INSPIRE lookup   -- for a plausible-date, out-of-corpus ID, ask INSPIRE-HEP
                         whether a record exists.                    [network, cached]

Verdicts:
    grounded              cited ID was in the retrieved set (not a violation)
    ungrounded_in_corpus  real Witten paper in the index, not retrieved
    ungrounded_out_real   real paper (INSPIRE record), outside the corpus
    fabricated_bad_date   impossible arXiv date -- invented
    fabricated_no_record  plausible date but no INSPIRE record -- invented
    unknown               out-of-corpus, plausible date, INSPIRE not consulted
                          (--offline) or unreachable

Only tier 3 touches the network, and it is cached to `evals/cache/inspire_existence.json`,
so a second run (and the paper's numbers) are reproducible with no live calls.

Usage:
    # Classify every off-passage cite across result files, print per-condition table:
    python -m evals.citation_provenance evals/results/full_*.jsonl evals/results/final_showdown.jsonl

    # Reproduce the hand audit's 15-ID verdict table:
    python -m evals.citation_provenance evals/results/full_run_merged.jsonl \
        evals/results/final_showdown.jsonl evals/results/full_[A-D]*.jsonl --by-id

    # No network (local tiers only; out-of-corpus plausible IDs -> unknown):
    python -m evals.citation_provenance evals/results/*.jsonl --offline

    # Dump per-ID verdicts as JSON for the writeup:
    python -m evals.citation_provenance evals/results/*.jsonl --json evals/results/provenance.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.validator import (  # noqa: E402
    is_paper_id,
    normalize_id,
    validate_citations,
)

LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"
CACHE_FILE = ROOT / "evals" / "cache" / "inspire_existence.json"

INSPIRE_API = "https://inspirehep.net/api/literature"
USER_AGENT = "Cyber-Witten/0.1 (personal research; citation provenance audit)"
DELAY = 0.5  # polite gap between live INSPIRE calls (matches the ingestion scripts)

CURRENT_YEAR = datetime.date.today().year  # self-maintaining upper date bound

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_title(title: str | None) -> str | None:
    """INSPIRE titles can carry MathML/HTML (<math>...</math>); strip tags."""
    if not title:
        return title
    return _TAG_RE.sub("", title).strip()

# --- verdict labels --------------------------------------------------------
GROUNDED = "grounded"
IN_CORPUS = "ungrounded_in_corpus"
OUT_REAL = "ungrounded_out_real"
BAD_DATE = "fabricated_bad_date"
NO_RECORD = "fabricated_no_record"
UNKNOWN = "unknown"

FABRICATED = {BAD_DATE, NO_RECORD}
UNGROUNDED = {IN_CORPUS, OUT_REAL, BAD_DATE, NO_RECORD, UNKNOWN}  # any off-passage cite

# order for display
VERDICT_ORDER = [GROUNDED, IN_CORPUS, OUT_REAL, BAD_DATE, NO_RECORD, UNKNOWN]


# --- local corpus ----------------------------------------------------------
def load_corpus_ids() -> set[str]:
    """Normalized arxiv_id of every paper in the index (the 'in-corpus' set)."""
    if not LOOKUP_FILE.exists():
        raise SystemExit(f"Missing {LOOKUP_FILE} -- build the index first (scripts 01-04).")
    ids = set()
    with LOOKUP_FILE.open() as f:
        for line in f:
            if line.strip():
                ids.add(normalize_id(json.loads(line)["arxiv_id"]))
    return ids


# --- tier 2: local date-plausibility --------------------------------------
_OLD_RE = re.compile(r"^[a-z][a-z-]*(?:\.[a-z]{2})?/(\d{2})(\d{2})(\d{3})$")
_NEW_RE = re.compile(r"^(\d{2})(\d{2})\.(\d{4,5})$")


def date_plausible(pid: str) -> bool | None:
    """True if the ID's embedded date lands in a valid arXiv window; False if
    impossible (a local fabrication signal); None if the ID carries no arXiv
    date to check (e.g. inspire:recid).

    Old scheme (archive/YYMMNNN) ran 1991-08 .. 2007-03; new scheme
    (YYMM.NNNNN) from 2007-04. The YY->year mapping is the subtle part and is
    exactly where the first hand-audit pass had a bug: 91..99 -> 1991..1999,
    00..07 -> 2000..2007, anything else old-style is impossible.
    """
    pid = normalize_id(pid)
    m = _OLD_RE.match(pid)
    if m:
        yy, mm = int(m.group(1)), int(m.group(2))
        if not 1 <= mm <= 12:
            return False
        if 91 <= yy <= 99:
            year = 1900 + yy
        elif 0 <= yy <= 7:
            year = 2000 + yy
        else:
            return False  # yy 08..90: no valid old-style arXiv date (pre-1991 or post-scheme)
        if year == 1991 and mm < 8:
            return False  # arXiv predates 1991-08
        if year == 2007 and mm > 3:
            return False  # old scheme ended 2007-03
        return True
    m = _NEW_RE.match(pid)
    if m:
        yy, mm = int(m.group(1)), int(m.group(2))
        if not 1 <= mm <= 12:
            return False
        year = 2000 + yy
        if year < 2007 or (year == 2007 and mm < 4):
            return False  # new scheme started 2007-04
        if year > CURRENT_YEAR:
            return False  # future-dated -> impossible
        return True
    return None  # inspire:recid or other non-dated form


def digit_slip_of(pid: str, corpus: set[str]) -> str | None:
    """If `pid` is a single digit-substitution away from an in-corpus ID, return
    that ID. Distinguishes a fat-fingered real corpus paper (e.g. 2306.10780 for
    2206.10780) from an ID hallucinated out of nothing."""
    key = normalize_id(pid)
    chars = list(key)
    for i, ch in enumerate(chars):
        if not ch.isdigit():
            continue
        for d in "0123456789":
            if d == ch:
                continue
            chars[i] = d
            cand = "".join(chars)
            if cand in corpus:
                return cand
        chars[i] = ch
    return None


# --- tier 3: INSPIRE existence (cached, network) --------------------------
def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False))


def inspire_exists(pid: str, cache: dict, session, offline: bool) -> dict:
    """{'exists': True|False|None, 'title': str|None, 'source': str}. Cached by ID."""
    key = normalize_id(pid)
    if key in cache:
        return cache[key]
    if offline or session is None:
        return {"exists": None, "title": None, "source": "offline"}

    if key.startswith("inspire:"):
        url = f"{INSPIRE_API}/{key.split(':', 1)[1]}"
        params = {"fields": "titles"}
    else:
        url = INSPIRE_API
        params = {"q": f"arxiv:{key}", "fields": "titles", "size": 1}

    try:
        r = session.get(url, params=params, timeout=20)
    except Exception as e:  # network down, DNS, timeout -- do not crash the run
        return {"exists": None, "title": None, "source": f"error:{type(e).__name__}"}
    time.sleep(DELAY)

    if r.status_code == 404:
        rec = {"exists": False, "title": None, "source": "inspire"}
    elif r.status_code != 200:
        return {"exists": None, "title": None, "source": f"http:{r.status_code}"}
    else:
        data = r.json()
        if key.startswith("inspire:"):
            md = data.get("metadata")
            title = md["titles"][0]["title"] if md and md.get("titles") else None
            rec = {"exists": bool(md), "title": _clean_title(title), "source": "inspire"}
        else:
            hits = data.get("hits", {}).get("hits", [])
            title = None
            if hits:
                titles = hits[0].get("metadata", {}).get("titles", [])
                title = titles[0]["title"] if titles else None
            rec = {"exists": bool(hits), "title": _clean_title(title), "source": "inspire"}
    cache[key] = rec
    return rec


# --- combined classifier ---------------------------------------------------
def classify(pid: str, corpus: set[str], cache: dict, session, offline: bool) -> tuple[str, str | None]:
    """Return (verdict, title_or_note) for one off-passage citation ID."""
    key = normalize_id(pid)
    if key in corpus:
        return IN_CORPUS, None
    if date_plausible(key) is False:
        return BAD_DATE, None
    rec = inspire_exists(key, cache, session, offline)
    if rec["exists"] is True:
        return OUT_REAL, _clean_title(rec.get("title"))
    if rec["exists"] is False:
        slip = digit_slip_of(key, corpus)
        return NO_RECORD, (f"digit-slip of {slip} (in corpus)" if slip else None)
    return UNKNOWN, rec.get("source")


# --- result-file loading ---------------------------------------------------
def iter_records(paths: list[Path]):
    for p in paths:
        with p.open() as f:
            for line in f:
                if line.strip():
                    yield p.name, json.loads(line)


def invalid_cites(rec: dict) -> list[str]:
    """Off-passage cites for a record. Trust the stored `auto` block when present
    (same validator that produced the reported numbers), else recompute."""
    auto = rec.get("auto")
    if auto and "invalid_citations" in auto:
        return auto["invalid_citations"]
    answer = rec.get("answer")
    retrieved = rec.get("retrieved_ids")
    if not answer or retrieved is None:
        return []
    return validate_citations(answer, retrieved)["invalid_citations"]


def valid_cites(rec: dict) -> list[str]:
    auto = rec.get("auto")
    if auto and "valid_citations" in auto:
        return auto["valid_citations"]
    answer = rec.get("answer")
    retrieved = rec.get("retrieved_ids")
    if not answer or retrieved is None:
        return []
    return validate_citations(answer, retrieved)["valid_citations"]


def condition_label(rec: dict) -> str:
    model = rec.get("model", "?")
    cond = rec.get("condition", "?")
    return f"{model} / {cond}"


# --- reporting -------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", nargs="+", type=Path, help="result .jsonl file(s)")
    ap.add_argument("--offline", action="store_true",
                    help="local tiers only; no INSPIRE calls (out-of-corpus plausible -> unknown)")
    ap.add_argument("--by-id", action="store_true",
                    help="also print one row per distinct off-passage ID (reproduces the hand audit)")
    ap.add_argument("--json", type=Path, metavar="OUT",
                    help="write per-ID verdicts + per-condition tallies to OUT")
    args = ap.parse_args()

    for p in args.results:
        if not p.exists():
            raise SystemExit(f"No such file: {p}")

    corpus = load_corpus_ids()
    cache = _load_cache()
    session = None
    if not args.offline:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT

    # distinct-ID verdicts (classify each unique off-passage ID once)
    id_verdict: dict[str, tuple[str, str | None]] = {}
    id_occurrences: Counter = Counter()
    # per-condition, per-verdict occurrence tallies (off-passage cites)
    by_cond: dict[str, Counter] = defaultdict(Counter)
    grounded_by_cond: Counter = Counter()
    answers_by_cond: Counter = Counter()
    violations_by_cond: Counter = Counter()

    n_records = 0
    for _fname, rec in iter_records(args.results):
        n_records += 1
        cond = condition_label(rec)
        answers_by_cond[cond] += 1
        grounded_by_cond[cond] += len(valid_cites(rec))
        inv = [c for c in invalid_cites(rec) if is_paper_id(c)]
        if inv:
            violations_by_cond[cond] += 1
        for cid in inv:
            key = normalize_id(cid)
            id_occurrences[key] += 1
            if key not in id_verdict:
                id_verdict[key] = classify(key, corpus, cache, session, args.offline)
            verdict = id_verdict[key][0]
            by_cond[cond][verdict] += 1

    if session is not None:
        _save_cache(cache)

    # ---- headline: distinct-ID rollup ----
    distinct = Counter(v for v, _ in id_verdict.values())
    occ = Counter()
    for key, (v, _) in id_verdict.items():
        occ[v] += id_occurrences[key]

    n_ids = len(id_verdict)
    n_fab_ids = sum(distinct[v] for v in FABRICATED)
    n_real_ids = distinct[IN_CORPUS] + distinct[OUT_REAL]
    print("=" * 74)
    print(f"CITATION PROVENANCE  ({n_records} answers, {n_ids} distinct off-passage IDs)")
    print("=" * 74)
    print(f"{'verdict':24} {'distinct IDs':>13} {'occurrences':>13}")
    print("-" * 74)
    for v in VERDICT_ORDER:
        if v == GROUNDED:
            continue
        if distinct[v] or occ[v]:
            print(f"{v:24} {distinct[v]:>13} {occ[v]:>13}")
    print("-" * 74)
    print(f"{'  -> real (ungrounded)':24} {n_real_ids:>13} {occ[IN_CORPUS] + occ[OUT_REAL]:>13}")
    print(f"{'  -> fabricated':24} {n_fab_ids:>13} {sum(occ[v] for v in FABRICATED):>13}")
    if distinct[UNKNOWN]:
        print(f"{'  -> unknown (no net)':24} {distinct[UNKNOWN]:>13} {occ[UNKNOWN]:>13}")
    print()

    # ---- per-condition table ----
    conds = sorted(answers_by_cond)
    cols = [IN_CORPUS, OUT_REAL, BAD_DATE, NO_RECORD, UNKNOWN]
    short = {IN_CORPUS: "in-corp", OUT_REAL: "out-real", BAD_DATE: "bad-date",
             NO_RECORD: "no-rec", UNKNOWN: "unknown"}
    print("PER CONDITION  (off-passage citation occurrences)")
    print("-" * 74)
    header = f"{'condition':22} {'ans':>4} {'viol':>5} {'grnd':>5} " + " ".join(f"{short[c]:>8}" for c in cols)
    print(header)
    for cond in conds:
        row = f"{cond[:22]:22} {answers_by_cond[cond]:>4} {violations_by_cond[cond]:>5} {grounded_by_cond[cond]:>5} "
        row += " ".join(f"{by_cond[cond][c]:>8}" for c in cols)
        print(row)
    print("-" * 74)
    print("ans=answers  viol=answers with >=1 off-passage cite  grnd=grounded cite occurrences")
    print()

    # ---- optional per-ID table (reproduces the hand audit) ----
    if args.by_id:
        print("PER DISTINCT OFF-PASSAGE ID")
        print("-" * 74)
        for key in sorted(id_verdict, key=lambda k: (id_verdict[k][0], k)):
            verdict, note = id_verdict[key]
            occ_n = id_occurrences[key]
            tail = f"  {note}" if note else ""
            print(f"  {verdict:22} x{occ_n:<3} {key}{tail}")
        print()

    # ---- optional JSON dump ----
    if args.json:
        out = {
            "n_answers": n_records,
            "distinct_ids": {k: {"verdict": v, "note": note, "occurrences": id_occurrences[k]}
                             for k, (v, note) in id_verdict.items()},
            "by_condition": {c: dict(by_cond[c]) for c in conds},
            "rollup": {"distinct": dict(distinct), "occurrences": dict(occ)},
        }
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"Wrote per-ID verdicts -> {args.json}")


if __name__ == "__main__":
    main()
