"""Download LaTeX source archives from arXiv for each paper.

Saves raw bytes as data/sources/{safe_aid}.src — format-agnostic.
The byte format (tar.gz / gzipped tex / pdf) is detected at parse time.

Rate-limited: 3 sec between requests as arXiv requests. Resumes by
skipping already-downloaded files.
"""
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "metadata" / "papers.jsonl"
SOURCES = ROOT / "data" / "sources"
SOURCES.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Cyber-Witten/0.1 (personal research toy)"
DELAY_SECONDS = 3.0


def arxiv_id_no_version(aid):
    """'hep-th/9108004v1' -> 'hep-th/9108004' ; '2510.06376v2' -> '2510.06376'"""
    last = aid.rsplit("/", 1)[-1]
    if "v" in last:
        base = "v".join(last.split("v")[:-1])
        prefix = aid.rsplit("/", 1)[0] + "/" if "/" in aid else ""
        return prefix + base
    return aid


def safe_filename(aid):
    return aid.replace("/", "_")


def main():
    papers = [json.loads(l) for l in METADATA.open()]
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    downloaded, skipped, failed = 0, 0, 0
    failures = []

    for paper in tqdm(papers, desc="Downloading"):
        aid_versioned = paper["arxiv_id"]
        aid = arxiv_id_no_version(aid_versioned)
        target = SOURCES / f"{safe_filename(aid)}.src"

        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue

        url = f"https://arxiv.org/e-print/{aid}"
        try:
            time.sleep(DELAY_SECONDS)
            r = session.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 0:
                target.write_bytes(r.content)
                downloaded += 1
            else:
                failed += 1
                failures.append((aid, f"HTTP {r.status_code}, {len(r.content)} bytes"))
        except Exception as e:
            failed += 1
            failures.append((aid, str(e)))

    print(f"\nDownloaded: {downloaded}")
    print(f"Skipped:    {skipped} (already had)")
    print(f"Failed:     {failed}")
    if failures:
        print("\nFailures:")
        for aid, reason in failures[:20]:
            print(f"  {aid}: {reason}")
    print(f"\nOutput: {SOURCES}")


if __name__ == "__main__":
    main()
