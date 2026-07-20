"""Report how complete THIS machine's Witten corpus is vs. the manifest, and
emit a download shopping list for the paywalled gap (grouped by publisher).

    Reads:  data/manifest/witten_corpus.jsonl   (from build_manifest.py)
            data/index/lookup.jsonl             (the local corpus)
    Writes: data/manifest/download_list.md      (what to fetch via your OWN access)

This tool NEVER downloads paywalled content. It stops at "here is the DOI". You
fetch through your own legitimate subscription, drop the PDFs (default filenames
are fine) into data/pdfs/, then run scripts/07_ingest_manual_pdfs.py to ingest
them. Free tiers (arxiv / oa_inspire) are pulled by scripts 01-04 / 06.

Usage:
    python scripts/corpus_coverage.py
    python scripts/corpus_coverage.py --articles-only
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest" / "witten_corpus.jsonl"
LOOKUP = ROOT / "data" / "index" / "lookup.jsonl"
LIST_OUT = ROOT / "data" / "manifest" / "download_list.md"

# foundational papers worth fetching first (matched by title keyword)
FAMOUS = {
    "dyons of charge": "Witten effect", "su(2) anomaly": "Witten anomaly",
    "constraints on supersymmetry": "Witten index", "nonabelian bosonization": "WZW model",
    "gravitational anomalies": "Alvarez-Gaume-Witten",
    "vacuum configurations for superstrings": "Candelas-Horowitz-Strominger-Witten",
    "instability of the kaluza": "bubble of nothing", "baryons in the 1": "large-N baryons",
    "search for a realistic kaluza": "Kaluza-Klein", "current algebra, baryons": "Skyrmion revival",
    "noncommutative geometry and string field": "open string field theory",
    "superconducting strings": "superconducting strings",
    "new manifolds for superstring": "CY compactification",
}


def famous(title: str) -> str | None:
    tl = title.lower()
    for k, v in FAMOUS.items():
        if k in tl:
            return v
    return None


def _norm(x: str) -> str:
    return re.sub(r"v\d+$", "", x.strip().lower())


def load_local_ids() -> tuple[set[str], set[str]]:
    ax, recids = set(), set()
    if not LOOKUP.exists():
        return ax, recids
    for line in LOOKUP.open():
        aid = json.loads(line)["arxiv_id"].strip().lower()
        if aid.startswith("inspire:"):
            recids.add(aid.split(":", 1)[1])
        else:
            ax.add(_norm(aid))
    return ax, recids


def in_corpus(row: dict, local_ax: set[str], local_recids: set[str]) -> bool:
    if row.get("arxiv_id") and _norm(row["arxiv_id"]) in local_ax:
        return True
    if row.get("recid") and str(row["recid"]) in local_recids:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--articles-only", action="store_true",
                    help="Only count/report document_type == article")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"No manifest at {args.manifest}. Run scripts/build_manifest.py first.")

    rows = [json.loads(l) for l in args.manifest.open() if l.strip()]
    if args.articles_only:
        rows = [r for r in rows if r.get("document_type") == "article"]
    local_ax, local_recids = load_local_ids()

    for r in rows:
        r["_have"] = in_corpus(r, local_ax, local_recids)
    have = [r for r in rows if r["_have"]]
    missing = [r for r in rows if not r["_have"]]

    print("=" * 66)
    print(f"CORPUS COVERAGE  ({len(have)}/{len(rows)} papers present"
          + (" [articles only]" if args.articles_only else "") + ")")
    print("=" * 66)
    for tier in ("arxiv", "oa_inspire", "paywalled", "metadata_only"):
        tot = sum(1 for r in rows if r["tier"] == tier)
        got = sum(1 for r in have if r["tier"] == tier)
        if tot:
            print(f"  {tier:14} {got:3}/{tot:<3}  missing {tot-got}")

    pay_missing = [r for r in missing if r["tier"] == "paywalled"]
    print(f"\nMissing paywalled (need your own access): {len(pay_missing)}")
    by_pub = Counter(r.get("publisher") for r in pay_missing)
    for pub, n in by_pub.most_common():
        print(f"  {pub:10} {n}")

    # write the download shopping list
    groups: dict[str, list] = defaultdict(list)
    for r in pay_missing:
        groups[r.get("publisher") or "other"].append(r)
    lines = ["# Witten corpus — download list (fetch via YOUR OWN subscription)\n",
             f"{len(pay_missing)} paywalled papers missing locally. Download keeping the "
             "publisher's DEFAULT filename, drop into `data/pdfs/`, then run "
             "`python scripts/07_ingest_manual_pdfs.py`.\n",
             "This tool does not fetch these for you; use your legitimate access.\n"]
    fam = [r for r in pay_missing if famous(r["title"])]
    if fam:
        lines.append(f"## Priority — {len(fam)} foundational\n")
        for r in sorted(fam, key=lambda x: x["year"]):
            lines.append(f"- **{famous(r['title'])}** — {r['year']} *{r['title']}* "
                         f"({r.get('journal')})  https://doi.org/{r.get('doi')}")
    for pub in sorted(groups):
        lines.append(f"\n## {pub} ({len(groups[pub])})\n")
        for r in sorted(groups[pub], key=lambda x: x["year"]):
            star = " (*)" if famous(r["title"]) else ""
            lines.append(f"- {r['year']}  {r['title'][:66]}{star}  https://doi.org/{r.get('doi')}")
    LIST_OUT.parent.mkdir(parents=True, exist_ok=True)
    LIST_OUT.write_text("\n".join(lines))

    n_free_missing = sum(1 for r in missing if r["tier"] in ("arxiv", "oa_inspire"))
    print(f"\nDownload list ({len(pay_missing)} papers, {len(fam)} foundational) "
          f"-> {LIST_OUT.relative_to(ROOT)}")
    if n_free_missing:
        print(f"Also {n_free_missing} FREE-tier papers missing — run scripts 01-04 / 06 to pull them.")


if __name__ == "__main__":
    main()
