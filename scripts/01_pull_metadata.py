"""Pull all arXiv papers authored by Edward Witten.

Writes one JSON record per paper to data/metadata/papers.jsonl.
Filters out namesakes (Louis Witten, Daniel Witten, etc.) by requiring
"Edward Witten" or "E. Witten" appears in the author list AND the primary
category is in a physics/math whitelist.
"""
import json
from pathlib import Path

import arxiv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "metadata" / "papers.jsonl"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Edward Witten works in these categories; anything outside is almost certainly
# a different "Witten" (statistician Daniel Witten in stat.*, etc.)
ALLOWED_CATEGORIES = {
    "hep-th", "hep-ph", "hep-lat", "math-ph",
    "gr-qc", "nucl-th", "cond-mat", "quant-ph",
    "math.AG", "math.QA", "math.SG", "math.GT", "math.DG",
    "math.RT", "math.CT", "math.AT", "math.NT",
}

def is_edward_witten(author_names):
    """Edward Witten typically appears as 'Edward Witten' or 'E. Witten'."""
    for name in author_names:
        n = name.strip()
        if n == "Edward Witten":
            return True
        if n in ("E. Witten", "Edward J. Witten"):
            return True
    return False

def category_ok(primary, all_cats):
    """Primary category must be in whitelist (handles math.* prefix)."""
    if primary in ALLOWED_CATEGORIES:
        return True
    # math.AG.something or cond-mat.str-el etc.
    base = primary.split(".")[0]
    if base in {"hep-th", "hep-ph", "hep-lat", "math-ph", "gr-qc",
                "nucl-th", "cond-mat", "quant-ph"}:
        return True
    return False

def main():
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query='au:"Witten, Edward"',
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    kept, skipped_author, skipped_cat = 0, 0, 0
    with OUTPUT.open("w") as f:
        for result in tqdm(client.results(search), desc="Pulling"):
            author_names = [a.name for a in result.authors]

            if not is_edward_witten(author_names):
                skipped_author += 1
                continue
            if not category_ok(result.primary_category, result.categories):
                skipped_cat += 1
                continue

            record = {
                "arxiv_id": result.entry_id.split("/abs/")[-1],
                "title": result.title.strip(),
                "authors": author_names,
                "abstract": result.summary.strip(),
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "primary_category": result.primary_category,
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "doi": result.doi,
            }
            f.write(json.dumps(record) + "\n")
            kept += 1

    print(f"\nKept:                 {kept}")
    print(f"Skipped (not E.W.):   {skipped_author}")
    print(f"Skipped (off-topic):  {skipped_cat}")
    print(f"Output: {OUTPUT}")

if __name__ == "__main__":
    main()
