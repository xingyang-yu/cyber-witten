"""Add pre-arXiv Witten papers (no arXiv ID) to the Cyber-Witten corpus.

Pipeline for each paper:
  1. INSPIRE 'documents' field  →  direct hosted PDF
  2. Semantic Scholar API       →  openAccessPdf URL
  3. Unpaywall API (by DOI)     →  best open-access PDF

PDF parsing:
  - pdfplumber  (digital/born-PDF)
  - pytesseract (OCR fallback for scanned images)

Output:
  - data/pdfs/inspire_<recid>.pdf      downloaded PDFs
  - data/pre_arxiv_failed.jsonl        papers with no recoverable text
  - appended to: chunks.jsonl, lookup.jsonl, bge.faiss, papers.jsonl

System deps (one-time):
    brew install tesseract poppler

Python deps (new):
    pip install pdfplumber pdf2image pytesseract

Usage:
    python scripts/06_pre_arxiv.py

Optional env var:
    UNPAYWALL_EMAIL=you@example.com   (Unpaywall requires any valid email)
"""

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from bge_embed import encode_texts, pick_device

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
METADATA    = ROOT / "data" / "metadata" / "papers.jsonl"
PDF_DIR     = ROOT / "data" / "pdfs"
CHUNKS      = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_FILE  = ROOT / "data" / "index" / "bge.faiss"
LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"
FAILED_FILE = ROOT / "data" / "pre_arxiv_failed.jsonl"

PDF_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
USER_AGENT      = "Cyber-Witten/0.1 (personal research toy)"
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "cyberwitten@localhost")
INSPIRE_DELAY   = 0.5   # between INSPIRE pagination requests
PDF_HUNT_DELAY  = 0.4   # between Semantic Scholar / Unpaywall calls
DOWNLOAD_DELAY  = 3.0   # between PDF downloads (be polite)

# ── import chunking helper from script 03 ────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "p3", ROOT / "scripts" / "03_parse_and_chunk.py"
)
p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3)


# ─────────────────────────────────────────────────────────────────────────────
# INSPIRE: fetch pre-arXiv Witten papers
# ─────────────────────────────────────────────────────────────────────────────

def _load_existing_ids():
    """Return set of IDs (arxiv or inspire:NNN) already in corpus."""
    ids = set()
    if METADATA.exists():
        for line in METADATA.open():
            m = json.loads(line)
            ids.add(m.get("arxiv_id", ""))
    return ids


def fetch_pre_arxiv_from_inspire():
    """Return list of INSPIRE metadata dicts for Witten papers with no arXiv ID."""
    existing = _load_existing_ids()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    papers, page, total_seen = [], 1, 0
    print("Paginating INSPIRE for pre-arXiv Witten papers…")

    while True:
        r = session.get(
            "https://inspirehep.net/api/literature",
            params={
                "q": 'a "Witten, Edward"',
                "fields": (
                    "arxiv_eprints,titles,authors.full_name,"
                    "publication_info,documents,dois,primary_arxiv_category"
                ),
                "size": 25,
                "page": page,
                "sort": "mostrecent",
            },
            timeout=30,
        )
        if r.status_code != 200:
            print(f"  INSPIRE HTTP {r.status_code} on page {page} — stopping.")
            break

        data    = r.json()
        hits    = data.get("hits", {}).get("hits", [])
        total_h = data.get("hits", {}).get("total", 0)
        if not hits:
            break

        total_seen += len(hits)
        for hit in hits:
            md    = hit.get("metadata", {})
            recid = str(hit.get("id", ""))
            if md.get("arxiv_eprints"):      # already on arXiv → in corpus
                continue
            inspire_id = f"inspire:{recid}"
            if inspire_id in existing:       # already ingested
                continue
            md["_recid"]      = recid
            md["_inspire_id"] = inspire_id
            papers.append(md)

        if total_seen >= total_h:
            break
        page += 1
        time.sleep(INSPIRE_DELAY)

    print(f"  {len(papers)} pre-arXiv papers not yet in corpus (out of ~{total_h} total hits)")
    return papers


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to extract fields from INSPIRE metadata
# ─────────────────────────────────────────────────────────────────────────────

def _title(md):
    ts = md.get("titles", [{}])
    return ts[0].get("title", "").strip() if ts else ""

def _year(md):
    pub = md.get("publication_info", [{}])
    y   = str(pub[0].get("year", "")) if pub else ""
    return y if y else "????"

def _doi(md):
    dois = md.get("dois", [])
    return dois[0].get("value", "") if dois else ""


# ─────────────────────────────────────────────────────────────────────────────
# PDF source hunting
# ─────────────────────────────────────────────────────────────────────────────

def _try_inspire_docs(md):
    """Return a PDF URL from INSPIRE's documents field, or None."""
    for doc in md.get("documents", []):
        url   = doc.get("url", "")
        fname = doc.get("filename", "").lower()
        key   = doc.get("key", "").lower()
        if not url:
            continue
        if (url.lower().endswith(".pdf")
                or fname.endswith(".pdf")
                or "pdf" in key
                or "inspirehep.net/files" in url):
            return url
    return None


def _try_semantic_scholar(title, doi=None):
    """Return open-access PDF URL from Semantic Scholar, or None."""
    ss = requests.Session()
    ss.headers["User-Agent"] = USER_AGENT

    # Precise: look up by DOI
    if doi:
        try:
            r = ss.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf"},
                timeout=15,
            )
            if r.status_code == 200:
                oa = r.json().get("openAccessPdf") or {}
                if oa.get("url"):
                    return oa["url"]
        except Exception:
            pass
        time.sleep(PDF_HUNT_DELAY)

    # Fuzzy: search by title
    try:
        r = ss.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title[:120], "fields": "openAccessPdf,title", "limit": 5},
            timeout=15,
        )
        if r.status_code == 200:
            for p in r.json().get("data", []):
                oa = p.get("openAccessPdf") or {}
                if oa.get("url"):
                    return oa["url"]
    except Exception:
        pass
    return None


def _try_unpaywall(doi):
    """Return best open-access PDF URL from Unpaywall, or None."""
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            timeout=15,
        )
        if r.status_code == 200:
            loc = r.json().get("best_oa_location") or {}
            return loc.get("url_for_pdf")
    except Exception:
        pass
    return None


def find_pdf_url(md):
    """Try all sources in order. Returns (source_label, url) or (None, None)."""
    doi   = _doi(md)
    title = _title(md)

    url = _try_inspire_docs(md)
    if url:
        return "inspire", url

    time.sleep(PDF_HUNT_DELAY)
    url = _try_semantic_scholar(title, doi)
    if url:
        return "s2", url

    time.sleep(PDF_HUNT_DELAY)
    url = _try_unpaywall(doi)
    if url:
        return "unpaywall", url

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# PDF download
# ─────────────────────────────────────────────────────────────────────────────

def download_pdf(url, dest):
    """Download PDF to dest. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 2000:
        return True
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=120,
            stream=True,
        )
        content = r.content
        if r.status_code == 200 and len(content) > 2000:
            # Sanity-check: first 4 bytes of a PDF are %PDF
            if content[:4] == b"%PDF" or b"%PDF" in content[:20]:
                dest.write_bytes(content)
                return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def _pdfplumber_text(pdf_path):
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except Exception:
        return ""


def _ocr_text(pdf_path):
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(str(pdf_path), dpi=300)
        return "\n\n".join(
            pytesseract.image_to_string(img, config="--psm 6") for img in images
        )
    except Exception:
        return ""


def pdf_to_text(pdf_path):
    """Extract text from PDF. Returns (text, method_used)."""
    text = _pdfplumber_text(pdf_path)
    if len(text.split()) >= 150:
        return text, "pdfplumber"
    # pdfplumber gave too little — might be a scan
    text = _ocr_text(pdf_path)
    if len(text.split()) >= 50:
        return text, "ocr"
    return "", "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Chunk building
# ─────────────────────────────────────────────────────────────────────────────

def _clean_pdf_text(text):
    """Light cleanup for PDF-extracted text."""
    # Remove hyphenation at line breaks: "super-\nsymmetry" → "supersymmetry"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def make_chunks(text, md):
    inspire_id = md["_inspire_id"]
    safe_id    = inspire_id.replace(":", "_")
    title      = _title(md)
    year       = _year(md)

    text   = _clean_pdf_text(text)
    chunks = p3.chunk_text(text)
    return [{
        "chunk_id":          f"{safe_id}_{idx:03d}",
        "arxiv_id":          inspire_id,   # "inspire:NNNNNN"
        "title":             title,
        "year":              year,
        "primary_category":  "hep-th",
        "chunk_idx":         idx,
        "n_chunks":          len(chunks),
        "text":              chunk,
    } for idx, chunk in enumerate(chunks)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 0. preflight check ───────────────────────────────────────────────────
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        sys.exit(
            "pdfplumber not installed.\n"
            "Run:  pip install pdfplumber pdf2image pytesseract\n"
            "      brew install tesseract poppler"
        )

    # ── 1. fetch INSPIRE metadata for pre-arXiv papers ───────────────────────
    papers = fetch_pre_arxiv_from_inspire()
    if not papers:
        print("Nothing new to add.")
        return

    # ── 2. hunt + download PDFs ──────────────────────────────────────────────
    print(f"\n[1/3] Hunting PDFs for {len(papers)} papers…")
    downloaded, failed = [], []

    for md in tqdm(papers, desc="PDF hunt"):
        title = _title(md)
        year  = _year(md)
        doi   = _doi(md)

        src, url = find_pdf_url(md)
        if not url:
            failed.append({
                "inspire_id": md["_inspire_id"],
                "title": title, "year": year, "doi": doi,
                "reason": "no_pdf_found",
            })
            continue

        pdf_path = PDF_DIR / f"inspire_{md['_recid']}.pdf"
        time.sleep(DOWNLOAD_DELAY)
        if download_pdf(url, pdf_path):
            downloaded.append((md, pdf_path, src))
        else:
            failed.append({
                "inspire_id": md["_inspire_id"],
                "title": title, "year": year, "doi": doi,
                "tried_url": url, "reason": "download_failed",
            })

    print(f"  Downloaded: {len(downloaded)}  |  No PDF found: {len(failed)}")

    # ── 3. parse PDFs ─────────────────────────────────────────────────────────
    print("\n[2/3] Parsing PDFs…")
    new_chunks, new_meta = [], []

    for (md, pdf_path, src) in tqdm(downloaded, desc="Parsing"):
        text, method = pdf_to_text(pdf_path)
        if not text:
            failed.append({
                "inspire_id": md["_inspire_id"],
                "title": _title(md), "year": _year(md),
                "reason": "parse_failed",
            })
            continue

        chunks = make_chunks(text, md)
        if not chunks:
            continue

        new_chunks.extend(chunks)
        doi = _doi(md)
        year = _year(md)
        new_meta.append({
            "arxiv_id":         md["_inspire_id"],
            "title":            _title(md),
            "authors":          [a.get("full_name", "") for a in md.get("authors", [])],
            "abstract":         "",
            "published":        f"{year}-01-01T00:00:00",
            "updated":          f"{year}-01-01T00:00:00",
            "primary_category": "hep-th",
            "categories":       ["hep-th"],
            "pdf_url":          f"https://doi.org/{doi}" if doi else "",
            "comment":          None,
            "journal_ref":      (md.get("publication_info") or [{}])[0].get("journal_title"),
            "doi":              doi,
        })
        print(
            f"  {md['_inspire_id']}  ({year})  [{method}]  "
            f"{len(chunks)} chunks   {_title(md)[:55]}"
        )

    print(f"\n  Parsed: {len(new_meta)} papers, {len(new_chunks)} chunks")

    # ── 4. embed + append ────────────────────────────────────────────────────
    if new_chunks:
        print("\n[3/3] Embedding + appending to index…")
        device = pick_device()
        new_embs = encode_texts(
            [c["text"] for c in new_chunks],
            batch_size=32,
            show_progress=True,
            device=device,
        )

        import faiss

        index  = faiss.read_index(str(INDEX_FILE))
        before = index.ntotal
        index.add(new_embs)
        faiss.write_index(index, str(INDEX_FILE))
        print(f"  Index: {before} → {index.ntotal}  (+{index.ntotal - before})")

        with LOOKUP_FILE.open("a") as f:
            for c in new_chunks:
                f.write(json.dumps(c) + "\n")
        with CHUNKS.open("a") as f:
            for c in new_chunks:
                f.write(json.dumps(c) + "\n")
        with METADATA.open("a") as f:
            for m in new_meta:
                f.write(json.dumps(m) + "\n")

    # ── 5. report failures ───────────────────────────────────────────────────
    if failed:
        with FAILED_FILE.open("w") as f:
            for rec in sorted(failed, key=lambda r: r.get("year", "")):
                f.write(json.dumps(rec) + "\n")

        print(f"\n⚠  {len(failed)} papers could not be added → {FAILED_FILE.name}")
        print("   (earliest / most notable ones shown below)")
        notable = sorted(failed, key=lambda r: r.get("year", "9999"))
        for rec in notable[:15]:
            doi_str = f"  doi:{rec['doi']}" if rec.get("doi") else ""
            print(f"   {rec['year']}  {rec['title'][:62]}{doi_str}")
        if len(notable) > 15:
            print(f"   … and {len(notable) - 15} more (see {FAILED_FILE.name})")

    print(f"\nDone. Added {len(new_meta)} papers, {len(new_chunks)} chunks.")


if __name__ == "__main__":
    main()
