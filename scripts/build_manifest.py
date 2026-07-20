"""Build the shareable Cyber-Witten corpus MANIFEST.

Identifiers + metadata for every Edward Witten paper on INSPIRE, tagged by how
its full text can be legally obtained. This is the artifact that makes the corpus
reproducible WITHOUT redistributing any copyrighted full text: it contains only
facts (arXiv id / INSPIRE recid / DOI / title / year / journal) and a `tier`
telling each user where to get the text themselves.

    tier = arxiv          full text is free on arXiv       (anyone; scripts 01-04)
    tier = oa_inspire      INSPIRE hosts an OA PDF          (anyone; script 06)
    tier = paywalled       behind a publisher paywall       (user's OWN subscription
                                                             -> data/pdfs/ -> script 07)
    tier = metadata_only   no full-text route found         (abstract-level at best)

Ship `data/manifest/witten_corpus.jsonl`. Ship NO full text. Each user runs
`corpus_coverage.py` to see their gap and rebuilds from sources they can access.

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --author "Witten, E" --out data/manifest/witten_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "data" / "manifest" / "witten_corpus.jsonl"
API = "https://inspirehep.net/api/literature"
USER_AGENT = "Cyber-Witten/0.1 (personal research; corpus manifest)"
FIELDS = ("control_number,titles,earliest_date,arxiv_eprints,publication_info,"
          "dois,documents,document_type")

PUBLISHERS = {"10.1016": "Elsevier", "10.1103": "APS", "10.1007": "Springer",
              "10.1088": "IOP", "10.1090": "AMS", "10.4310": "IntPress"}


def publisher_of(doi: str | None) -> str | None:
    if not doi:
        return None
    return PUBLISHERS.get(doi.split("/", 1)[0], "other")


def year_of(md: dict) -> str:
    pub = md.get("publication_info") or [{}]
    y = str(pub[0].get("year", "")) if pub else ""
    if not y:
        d = md.get("earliest_date", "") or ""
        y = d[:4] if d[:4].isdigit() else ""
    return y


def tier_of(md: dict) -> str:
    if md.get("arxiv_eprints"):
        return "arxiv"
    if md.get("documents"):
        return "oa_inspire"
    if md.get("dois"):
        return "paywalled"
    return "metadata_only"


def fetch_all(author: str) -> list[dict]:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    out, page = [], 1
    while True:
        r = s.get(API, params={"q": f'a {author}', "fields": FIELDS, "size": 100,
                               "page": page, "sort": "leastrecent"}, timeout=40)
        r.raise_for_status()
        hits = r.json()["hits"]["hits"]
        if not hits:
            break
        out.extend(hits)
        if len(hits) < 100:
            break
        page += 1
        time.sleep(0.5)
    return out


def record(md: dict) -> dict:
    doi = (md.get("dois") or [{}])[0].get("value")
    arxiv = (md.get("arxiv_eprints") or [{}])[0].get("value")
    return {
        "recid": md.get("control_number"),
        "arxiv_id": arxiv,
        "doi": doi,
        "title": (md.get("titles") or [{}])[0].get("title", "").strip(),
        "year": year_of(md),
        "journal": (md.get("publication_info") or [{}])[0].get("journal_title"),
        "document_type": (md.get("document_type") or [None])[0],
        "tier": tier_of(md),
        "publisher": publisher_of(doi) if tier_of(md) == "paywalled" else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Use Edward Witten's canonical INSPIRE BAI, NOT a name string: a fuzzy
    # name query ("Witten, E") wrongly pulls in other authors (e.g. the
    # astronomer Callum E.C. Witten). The BAI is disambiguated to recid 983328.
    ap.add_argument("--author", default="Edward.Witten.1",
                    help="INSPIRE author BAI (default: Edward.Witten.1 = Edward Witten)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    print(f"Querying INSPIRE for all papers by '{args.author}' ...")
    hits = fetch_all(args.author)
    rows = [record(h["metadata"]) for h in hits]
    # stable order: year then title
    rows.sort(key=lambda r: (r["year"] or "", r["title"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    tiers = Counter(r["tier"] for r in rows)
    dt = Counter(r["document_type"] for r in rows)
    print(f"\nWrote {len(rows)} papers -> {args.out.relative_to(ROOT)}")
    print("by tier:", dict(tiers))
    print("by document_type:", dict(dt))
    print("\nShip this file. It contains identifiers + metadata only, no full text.")


if __name__ == "__main__":
    main()
