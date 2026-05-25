"""Add missing arxiv IDs to the corpus incrementally.

Uses INSPIRE-HEP for metadata (no arxiv API rate limit) and arxiv.org/e-print
for source downloads (different endpoint than the search API).

Reads IDs from /tmp/missing_arxiv_ids.txt.
"""
import importlib.util
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from bge_embed import encode_texts, pick_device

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "metadata" / "papers.jsonl"
SOURCES = ROOT / "data" / "sources"
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_FILE = ROOT / "data" / "index" / "bge.faiss"
LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"
MISSING_LIST = Path("/tmp/missing_arxiv_ids.txt")

USER_AGENT = "Cyber-Witten/0.1 (personal research toy)"
DELAY = 4.0

spec = importlib.util.spec_from_file_location("p3", ROOT / "scripts" / "03_parse_and_chunk.py")
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)


def safe_filename(aid):
    return aid.replace("/", "_")


def fetch_metadata_from_inspire(arxiv_ids):
    """Query INSPIRE for metadata of these arxiv IDs."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    out = {}
    for aid in tqdm(arxiv_ids, desc="INSPIRE metadata"):
        # query by arxiv eprint
        r = session.get("https://inspirehep.net/api/literature", params={
            "q": f"arxiv:{aid}",
            "fields": "arxiv_eprints,titles,authors.full_name,publication_info,primary_arxiv_category",
            "size": 1,
        }, timeout=20)
        if r.status_code != 200:
            print(f"  {aid}: HTTP {r.status_code}")
            continue
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            print(f"  {aid}: not found in INSPIRE")
            continue
        md = hits[0].get("metadata", {})
        eprints = md.get("arxiv_eprints", [])
        primary_cat = "hep-th"
        if eprints and "categories" in eprints[0]:
            cats = eprints[0]["categories"]
            primary_cat = cats[0] if cats else "hep-th"
        pub_info = md.get("publication_info", [{}])[0]
        year = str(pub_info.get("year", ""))
        # fallback year extraction
        if not year:
            # arxiv id like "1401.8048" -> 2014; "hep-th/9407087" -> 1994
            if "/" in aid:
                yy = aid.split("/")[1][:2]
                year = f"19{yy}" if int(yy) > 50 else f"20{yy}"
            else:
                yy = aid[:2]
                year = f"19{yy}" if int(yy) > 50 else f"20{yy}"
        out[aid] = {
            "arxiv_id": aid,
            "title": md.get("titles", [{}])[0].get("title", "?").strip(),
            "authors": [a.get("full_name", "") for a in md.get("authors", [])],
            "abstract": "",
            "published": f"{year}-01-01T00:00:00",  # approximate
            "updated": f"{year}-01-01T00:00:00",
            "primary_category": primary_cat,
            "categories": [primary_cat],
            "pdf_url": f"https://arxiv.org/pdf/{aid}",
            "comment": None,
            "journal_ref": None,
            "doi": None,
        }
        time.sleep(0.5)
    return out


def download_source(aid):
    target = SOURCES / f"{safe_filename(aid)}.src"
    if target.exists() and target.stat().st_size > 0:
        return True
    url = f"https://arxiv.org/e-print/{aid}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    if r.status_code == 200 and len(r.content) > 0:
        target.write_bytes(r.content)
        return True
    return False


def parse_paper(aid, meta):
    src_path = SOURCES / f"{safe_filename(aid)}.src"
    if not src_path.exists():
        return []
    tex_files = p3.detect_and_extract(src_path.read_bytes())
    if not tex_files:
        return []
    main_pair = p3.find_main(tex_files)
    if not main_pair:
        return []
    text = p3.latex_to_text(main_pair[1])
    if len(text.split()) < 50:
        return []
    chunks = p3.chunk_text(text)
    return [{
        "chunk_id": f"{safe_filename(aid)}_{idx:03d}",
        "arxiv_id": aid,
        "title": meta["title"],
        "year": meta["published"][:4],
        "primary_category": meta["primary_category"],
        "chunk_idx": idx,
        "n_chunks": len(chunks),
        "text": chunk,
    } for idx, chunk in enumerate(chunks)]


def main():
    missing_ids = [l.strip() for l in MISSING_LIST.open() if l.strip()]
    print(f"To add: {len(missing_ids)} arxiv IDs\n")

    print("[1/4] Fetching metadata from INSPIRE...")
    meta_map = fetch_metadata_from_inspire(missing_ids)
    print(f"  Got {len(meta_map)} / {len(missing_ids)}\n")

    print("[2/4] Downloading sources from arxiv.org/e-print...")
    downloaded = []
    for aid in tqdm(meta_map):
        time.sleep(DELAY)
        if download_source(aid):
            downloaded.append(aid)
    print(f"  Downloaded {len(downloaded)}\n")

    print("[3/4] Parsing new papers...")
    new_chunks = []
    new_meta = []
    for aid in downloaded:
        chunks = parse_paper(aid, meta_map[aid])
        if chunks:
            new_chunks.extend(chunks)
            new_meta.append(meta_map[aid])
            print(f"  {aid}: {len(chunks)} chunks")
        else:
            print(f"  {aid}: FAILED to parse")
    print(f"\n  Total: {len(new_meta)} papers, {len(new_chunks)} new chunks")

    if not new_chunks:
        print("\nNothing to add. Done.")
        return

    print("\n[4/4] Embedding + appending to index...")
    device = pick_device()
    new_embs = encode_texts(
        [c["text"] for c in new_chunks],
        batch_size=32,
        show_progress=True,
        device=device,
    )

    import faiss

    index = faiss.read_index(str(INDEX_FILE))
    before = index.ntotal
    index.add(new_embs)
    print(f"  Index: {before} -> {index.ntotal} vectors (+{index.ntotal - before})")
    faiss.write_index(index, str(INDEX_FILE))

    with LOOKUP_FILE.open("a") as f:
        for c in new_chunks:
            f.write(json.dumps(c) + "\n")
    with CHUNKS.open("a") as f:
        for c in new_chunks:
            f.write(json.dumps(c) + "\n")
    with METADATA.open("a") as f:
        for m in new_meta:
            f.write(json.dumps(m) + "\n")

    print(f"\nDone. Added {len(new_meta)} papers, {len(new_chunks)} chunks.")


if __name__ == "__main__":
    main()
