"""Build the client-side (browser) retrieval index for the static web demo.

Re-embeds the arXiv-only public corpus with a SMALL model (bge-small-en-v1.5,
384-dim) so the whole thing fits in a browser: the query is encoded in-browser
via transformers.js (Xenova/bge-small-en-v1.5, the ONNX export of the same
model), and cosine similarity is a brute-force dot product over these vectors.

Why small: HuggingFace now gates compute Spaces (Gradio) behind PRO; only Static
Spaces are free. A static site has no server, so retrieval must run client-side,
and bge-large (1.3GB) is too big for a browser. bge-small trades some retrieval
sharpness for "free forever, no server."

Outputs (web/data/):
    vectors.bin    float32 [N, 384], row-major, L2-normalized (raw bytes)
    meta.json      [{id, year, title, snippet}] parallel to vectors
    texts.json.gz  [full chunk text] parallel to vectors, gzipped — fetched
                   lazily by the demo only when a visitor uses bring-your-own-key
                   generation (retrieval-only visitors never download it)
    config.json    {n, dim, model, query_prefix}
"""
import argparse
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "public_export" / "lookup.jsonl"
OUT = ROOT / "web" / "data"

MODEL = "BAAI/bge-small-en-v1.5"          # PyTorch; browser uses Xenova ONNX export
BROWSER_MODEL = "Xenova/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SNIPPET_CHARS = 300

# --- snippet cleaning -------------------------------------------------------
# Chunk text often opens with parsing debris (dates, ==== rules, LaTeX length
# tokens like `1.5in`, arXiv-stub headers). The embedding sees the full text,
# but the DISPLAYED snippet should start at real prose.
_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_NOISE_PATTERNS = [
    re.compile(rf"^{_MONTHS}\s+\d{{1,2}},\s+\d{{4}}"),          # leading date line
    re.compile(r"[=~_—-]{3,}"),                                  # ===== horizontal rules
    re.compile(r"(?<![A-Za-z0-9])\d*\.?\d+(?:in|cm|pt|mm|em)\b"),  # LaTeX lengths: 1.5in .1cm
    re.compile(r"\b[a-z-]{2,}/yymm\.nnnn\b"),                    # arXiv-id template stubs
]
# First "real prose" start: a Capitalized word (second char lowercase, so ALL-CAPS
# section headers are skipped) followed by at least 7 more tokens.
_PROSE_START = re.compile(r"[A-Z][a-z'’\-]+(?:\s+\S+){7,}")


def clean_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    t = " ".join(text.split())
    for pat in _NOISE_PATTERNS:
        t = pat.sub(" ", t)
    t = " ".join(t.split())
    m = _PROSE_START.search(t)
    if m:
        t = t[m.start():]
    return t[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-only", action="store_true",
                    help="Rewrite meta.json/config.json (e.g. after a snippet-cleaning change) "
                         "without re-embedding; vectors.bin must exist and match lookup.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in SRC.open()]
    n, dim = len(rows), 384

    OUT.mkdir(parents=True, exist_ok=True)
    if args.meta_only:
        vb = OUT / "vectors.bin"
        if not vb.exists() or vb.stat().st_size != n * dim * 4:
            raise SystemExit("--meta-only: vectors.bin missing or row count no longer matches "
                             "lookup.jsonl — run a full build instead")
        print(f"meta-only: keeping existing vectors.bin ({n} x {dim})")
    else:
        from bge_embed import encode_texts  # CLS pooling + L2 normalize, model_path override

        texts = [r["text"] for r in rows]
        print(f"Embedding {len(texts)} chunks with {MODEL} (384-dim)…", flush=True)
        embs = encode_texts(texts, batch_size=64, show_progress=True, model_path=MODEL)
        assert embs.dtype.name == "float32"
        n, dim = embs.shape
        print(f"vectors: {embs.shape}")
        (OUT / "vectors.bin").write_bytes(embs.tobytes(order="C"))

    texts = [" ".join(r["text"].split()) for r in rows]
    with gzip.open(OUT / "texts.json.gz", "wt", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)

    meta = [
        {
            "id": r["arxiv_id"],
            "year": r.get("year", ""),
            "title": r.get("title", ""),
            "snippet": clean_snippet(r["text"]),
        }
        for r in rows
    ]
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    (OUT / "config.json").write_text(json.dumps({
        "n": n, "dim": dim, "model": BROWSER_MODEL, "query_prefix": QUERY_PREFIX,
    }, indent=2))

    vb = (OUT / "vectors.bin").stat().st_size / 1e6
    mb = (OUT / "meta.json").stat().st_size / 1e6
    tb = (OUT / "texts.json.gz").stat().st_size / 1e6
    print(f"Wrote web/data/: vectors.bin {vb:.1f}MB, meta.json {mb:.1f}MB, "
          f"texts.json.gz {tb:.1f}MB, config.json")


if __name__ == "__main__":
    main()
