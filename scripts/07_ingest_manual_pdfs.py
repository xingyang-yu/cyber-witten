"""Ingest manually downloaded PDFs into the Cyber-Witten corpus.

Scans data/pdfs/ for any PDF that hasn't already been processed
(i.e. not already in papers.jsonl), identifies each via INSPIRE,
parses text, chunks, embeds, and appends to the FAISS index.

Filename conventions handled:
  7701121.pdf          → INSPIRE record ID
  0033151.pdf          → INSPIRE record ID (with leading zeros)
  BF02100009.pdf       → Springer DOI 10.1007/BF02100009
  1-s2.0-...-main.pdf  → Elsevier PII → INSPIRE DOI lookup
  PhysRevD.17.2134.pdf → APS DOI 10.1103/PhysRevD.17.2134
  SDG.*.pdf / other    → title search from PDF text

Usage:
    python scripts/07_ingest_manual_pdfs.py
"""

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from bge_embed import encode_texts, pick_device
# faiss is deferred to the embedding step.

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
METADATA    = ROOT / "data" / "metadata" / "papers.jsonl"
PDF_DIR     = ROOT / "data" / "pdfs"
CHUNKS      = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_FILE  = ROOT / "data" / "index" / "bge.faiss"
LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"
UNMATCHED_FILE = ROOT / "data" / "manual_pdf_unmatched.jsonl"

USER_AGENT = "Cyber-Witten/0.1 (personal research toy)"

# ── import chunking helper ────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "p3", ROOT / "scripts" / "03_parse_and_chunk.py"
)
p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3)


# ─────────────────────────────────────────────────────────────────────────────
# Identify already-processed IDs
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_ids():
    """Return set of arxiv_id values already in corpus."""
    ids = set()
    if METADATA.exists():
        for line in METADATA.open():
            ids.add(json.loads(line).get("arxiv_id", ""))
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# INSPIRE metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


def _inspire_by_id(record_id):
    try:
        r = _session.get(
            f"https://inspirehep.net/api/literature/{record_id}",
            timeout=15,
        )
        if r.status_code == 200:
            md = r.json().get("metadata", {})
            if md:
                md["_recid"] = str(record_id)
                return md
    except Exception:
        pass
    return None


def _inspire_by_doi(doi):
    try:
        r = _session.get(
            "https://inspirehep.net/api/literature",
            params={"q": f"doi:{doi}", "size": 1,
                    "fields": "titles,authors.full_name,publication_info,arxiv_eprints,dois"},
            timeout=15,
        )
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            if hits:
                md = hits[0].get("metadata", {})
                md["_recid"] = str(hits[0].get("id", ""))
                return md
    except Exception:
        pass
    return None


def _inspire_by_report(report_num):
    """Query INSPIRE by report/preprint number (e.g. '89-01-237')."""
    try:
        r = _session.get(
            "https://inspirehep.net/api/literature",
            params={"q": f"r {report_num}", "size": 1,
                    "fields": "titles,authors.full_name,publication_info,arxiv_eprints,dois"},
            timeout=15,
        )
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            if hits:
                md = hits[0].get("metadata", {})
                md["_recid"] = str(hits[0].get("id", ""))
                return md
    except Exception:
        pass
    return None


def _inspire_by_title(title_snippet):
    try:
        r = _session.get(
            "https://inspirehep.net/api/literature",
            params={"q": f't "{title_snippet}"', "size": 1,
                    "fields": "titles,authors.full_name,publication_info,arxiv_eprints,dois"},
            timeout=15,
        )
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            if hits:
                md = hits[0].get("metadata", {})
                md["_recid"] = str(hits[0].get("id", ""))
                return md
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Filename → metadata
# ─────────────────────────────────────────────────────────────────────────────

def _elsevier_pii_to_doi(pii_raw):
    """
    Reconstruct a proper Elsevier DOI from the ScienceDirect filename PII.
    Filename pattern: 1-s2.0-{16-char-PII}-main  e.g. 0550321376901115
    Maps to DOI:      10.1016/XXXX-XXXX(YY)ZZZZZ-N

    Common journal prefixes by ISSN prefix:
      0550-3213  Nucl.Phys.B
      0003-4916  Annals of Physics
      0370-2693  Phys.Lett.B
    """
    p = pii_raw  # e.g. '0550321376901115'
    # Format: IIIIIIII YY NNNNN C  (8-digit ISSN digits, 2-digit year, 5-digit seq, 1 check)
    if len(p) >= 16:
        issn_raw = p[:8]   # e.g. 05503213
        yy       = p[8:10] # e.g. 76
        seq      = p[10:15]# e.g. 90111
        check    = p[15]   # e.g. 5
        issn     = f"{issn_raw[:4]}-{issn_raw[4:]}"  # 0550-3213
        doi      = f"10.1016/{issn}({yy}){seq}-{check}"
        md = _inspire_by_doi(doi)
        if md:
            return md
        time.sleep(0.2)
    # Fallback: try raw PII with common prefixes
    for prefix in ["10.1016", "10.1006"]:
        md = _inspire_by_doi(f"{prefix}/{pii_raw}")
        if md:
            return md
    return None


def identify_paper(pdf_path, extracted_text=""):
    """
    Return (inspire_metadata_dict, canonical_id_str) or (None, None).
    Tries filename patterns first, then title search from PDF text.
    """
    stem = pdf_path.stem

    # ── pure digits ─────────────────────────────────────────────────────────
    # Files like 0033151.pdf are zero-padded INSPIRE record IDs. Files like
    # 8712296.pdf are old preprint/report numbers, not modern INSPIRE recids;
    # treating them as recids can silently match unrelated recent papers.
    if re.match(r'^\d+$', stem):
        if stem.startswith("00"):
            rid = stem.lstrip("0") or "0"
            md = _inspire_by_id(rid)
            if md:
                return md, f"inspire:{rid}"
            time.sleep(0.3)

        # SPIRES preprint format: YYMMNNN → YY-MM-NNN
        if len(stem) == 7:
            report = f"{stem[0:2]}-{stem[2:4]}-{stem[4:]}"
            md = _inspire_by_report(report)
            if md:
                return md, f"inspire:{md['_recid']}"
            time.sleep(0.3)

    # ── Springer BF* ─────────────────────────────────────────────────────────
    if re.match(r'^BF\d+$', stem):
        doi = f"10.1007/{stem}"
        md = _inspire_by_doi(doi)
        if md:
            return md, doi
        time.sleep(0.3)

    # ── APS PhysRev* ─────────────────────────────────────────────────────────
    if re.match(r'^PhysRev', stem):
        doi = f"10.1103/{stem}"
        md = _inspire_by_doi(doi)
        if md:
            return md, doi
        time.sleep(0.3)

    # ── Elsevier 1-s2.0-{PII}-main ───────────────────────────────────────────
    m = re.match(r'^1-s2\.0-(.+)-main$', stem)
    if m:
        pii = m.group(1)
        md = _elsevier_pii_to_doi(pii)
        if md:
            return md, f"pii:{pii}"
        time.sleep(0.3)

    # ── title search from extracted text ─────────────────────────────────────
    if extracted_text:
        # grab first non-empty line that looks like a title (>10 chars)
        for line in extracted_text.splitlines():
            line = line.strip()
            if len(line) > 10 and not line.startswith("%"):
                md = _inspire_by_title(line[:80])
                if md:
                    return md, f"title:{line[:40]}"
                time.sleep(0.3)
                break

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# PDF parsing
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_text(pdf_path, timeout_sec=30):
    """Extract text from PDF with a hard timeout to avoid hangs."""
    import concurrent.futures

    def _pdfplumber():
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(p for p in pages if p.strip())

    def _ocr():
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(str(pdf_path), dpi=150)   # lower DPI = faster
        return "\n\n".join(
            pytesseract.image_to_string(img, config="--psm 6") for img in images
        )

    for fn, label, min_words in [(_pdfplumber, "pdfplumber", 100), (_ocr, "ocr", 50)]:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fn)
                text = future.result(timeout=timeout_sec)
            if len(text.split()) >= min_words:
                return text, label
        except concurrent.futures.TimeoutError:
            print(f"    ⏱ {pdf_path.name}: {label} timed out after {timeout_sec}s, skipping")
        except Exception:
            pass

    return "", "failed"


def _clean(text):
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md_title(md):
    ts = md.get("titles", [{}])
    return ts[0].get("title", "").strip() if ts else ""

def _md_year(md):
    pub = md.get("publication_info", [{}])
    y = str(pub[0].get("year", "")) if pub else ""
    return y if y else "????"

def _md_doi(md):
    dois = md.get("dois", [])
    return dois[0].get("value", "") if dois else ""

def _md_arxiv(md):
    eps = md.get("arxiv_eprints", [])
    return eps[0].get("value", "") if eps else ""

def _md_authors(md):
    return [a.get("full_name", "") for a in md.get("authors", [])]


def _has_witten_author(md):
    return any("witten" in author.lower() for author in _md_authors(md))


def _valid_year(year):
    return bool(re.match(r"^(19|20)\d\d$", str(year or "")))


def _year_from_filename(pdf_path):
    """Best-effort fallback for YYMMNNN-style preprint filenames."""
    stem = pdf_path.stem
    if re.match(r"^\d{7}$", stem):
        yy = int(stem[:2])
        return f"19{yy}" if yy > 50 else f"20{yy}"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    try:
        import pdfplumber  # noqa
    except ImportError:
        sys.exit("pip install pdfplumber pdf2image pytesseract")

    existing_ids = load_existing_ids()

    # Find all PDFs that are NOT already-processed inspire_* files
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    new_pdfs = [p for p in all_pdfs if not p.name.startswith("inspire_")]
    print(f"Found {len(new_pdfs)} manually downloaded PDFs to process\n")

    new_chunks, new_meta, unmatched = [], [], []

    for pdf_path in tqdm(new_pdfs, desc="Processing"):
        # 1. Parse text first (needed for title-search fallback)
        text, method = pdf_to_text(pdf_path)
        if not text:
            print(f"  ✗ {pdf_path.name}: failed to extract text")
            unmatched.append({"file": pdf_path.name, "reason": "no_text"})
            continue

        # 2. Identify paper → get INSPIRE metadata
        time.sleep(0.3)
        inspire_md, canonical_id = identify_paper(pdf_path, text)

        if inspire_md is None:
            print(f"  ? {pdf_path.name}: no INSPIRE match — skipping")
            first_line = next(
                (l.strip() for l in text.splitlines() if len(l.strip()) > 10), pdf_path.stem
            )
            unmatched.append({
                "file": pdf_path.name,
                "reason": "no_inspire_match",
                "first_line": first_line[:160],
            })
            continue

        if not _has_witten_author(inspire_md):
            print(f"  ? {pdf_path.name}: INSPIRE match has no Witten author — skipping")
            unmatched.append({
                "file": pdf_path.name,
                "reason": "no_witten_author",
                "matched_recid": inspire_md.get("_recid"),
                "matched_title": _md_title(inspire_md),
                "matched_authors": _md_authors(inspire_md)[:10],
            })
            continue

        # Determine the ID to use in corpus
        arxiv_id = _md_arxiv(inspire_md)
        if arxiv_id:
            corpus_id = arxiv_id
        elif inspire_md.get("_recid"):
            corpus_id = f"inspire:{inspire_md['_recid']}"
        else:
            corpus_id = canonical_id

        # 3. Skip if already in corpus
        if corpus_id in existing_ids:
            print(f"  ↷ {pdf_path.name}: already in corpus ({corpus_id})")
            continue

        # 4. Chunk
        text    = _clean(text)
        chunks  = p3.chunk_text(text)
        if not chunks:
            print(f"  ✗ {pdf_path.name}: no chunks produced")
            continue

        title  = _md_title(inspire_md)
        year   = _md_year(inspire_md)
        if not _valid_year(year):
            year = _year_from_filename(pdf_path)
        if not _valid_year(year):
            print(f"  ? {pdf_path.name}: no reliable year — skipping ({corpus_id})")
            unmatched.append({
                "file": pdf_path.name,
                "reason": "no_reliable_year",
                "corpus_id": corpus_id,
                "title": title,
            })
            continue
        doi    = _md_doi(inspire_md)
        safe   = corpus_id.replace(":", "_").replace("/", "_")

        chunk_recs = [{
            "chunk_id":          f"{safe}_{idx:03d}",
            "arxiv_id":          corpus_id,
            "title":             title,
            "year":              year,
            "primary_category":  "hep-th",
            "chunk_idx":         idx,
            "n_chunks":          len(chunks),
            "text":              chunk,
        } for idx, chunk in enumerate(chunks)]

        new_chunks.extend(chunk_recs)
        new_meta.append({
            "arxiv_id":          corpus_id,
            "title":             title,
            "authors":           _md_authors(inspire_md),
            "abstract":          "",
            "published":         f"{year}-01-01T00:00:00",
            "updated":           f"{year}-01-01T00:00:00",
            "primary_category":  "hep-th",
            "categories":        ["hep-th"],
            "pdf_url":           f"https://doi.org/{doi}" if doi else "",
            "comment":           None,
            "journal_ref":       (inspire_md.get("publication_info") or [{}])[0].get("journal_title"),
            "doi":               doi,
        })
        existing_ids.add(corpus_id)
        print(
            f"  ✓ [{method}] {year}  {len(chunks):3d} chunks  "
            f"{title[:55]}  ({corpus_id})"
        )

    print(f"\nParsed {len(new_meta)} papers, {len(new_chunks)} chunks")

    if not new_chunks:
        print("Nothing to embed.")
        return

    # ── Embed + append ────────────────────────────────────────────────────────
    print("\nLoading embedding model…", flush=True)

    device = pick_device()
    embs = encode_texts(
        [c["text"] for c in new_chunks],
        batch_size=32,
        show_progress=True,
        device=device,
    )

    import faiss

    index  = faiss.read_index(str(INDEX_FILE))
    before = index.ntotal
    index.add(embs)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Index: {before} → {index.ntotal}  (+{index.ntotal - before})")

    with LOOKUP_FILE.open("a") as f:
        for c in new_chunks:
            f.write(json.dumps(c) + "\n")
    with CHUNKS.open("a") as f:
        for c in new_chunks:
            f.write(json.dumps(c) + "\n")
    with METADATA.open("a") as f:
        for m in new_meta:
            f.write(json.dumps(m) + "\n")

    if unmatched:
        with UNMATCHED_FILE.open("w") as f:
            for u in unmatched:
                f.write(json.dumps(u) + "\n")
        print(f"\n⚠  {len(unmatched)} files skipped → {UNMATCHED_FILE.name}")

    print(f"\nDone. Added {len(new_meta)} papers, {len(new_chunks)} chunks.")


if __name__ == "__main__":
    main()
