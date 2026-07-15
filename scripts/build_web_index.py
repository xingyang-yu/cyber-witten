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
    vectors.bin   float32 [N, 384], row-major, L2-normalized (raw bytes)
    meta.json     [{id, year, title, snippet}] parallel to vectors
    config.json   {n, dim, model, query_prefix}
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "public_export" / "lookup.jsonl"
OUT = ROOT / "web" / "data"

MODEL = "BAAI/bge-small-en-v1.5"          # PyTorch; browser uses Xenova ONNX export
BROWSER_MODEL = "Xenova/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SNIPPET_CHARS = 300


def main():
    from bge_embed import encode_texts  # CLS pooling + L2 normalize, model_path override

    rows = [json.loads(line) for line in SRC.open()]
    texts = [r["text"] for r in rows]
    print(f"Embedding {len(texts)} chunks with {MODEL} (384-dim)…", flush=True)

    embs = encode_texts(texts, batch_size=64, show_progress=True, model_path=MODEL)
    assert embs.dtype.name == "float32"
    n, dim = embs.shape
    print(f"vectors: {embs.shape}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vectors.bin").write_bytes(embs.tobytes(order="C"))

    meta = [
        {
            "id": r["arxiv_id"],
            "year": r.get("year", ""),
            "title": r.get("title", ""),
            "snippet": " ".join(r["text"].split())[:SNIPPET_CHARS],
        }
        for r in rows
    ]
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    (OUT / "config.json").write_text(json.dumps({
        "n": n, "dim": dim, "model": BROWSER_MODEL, "query_prefix": QUERY_PREFIX,
    }, indent=2))

    vb = (OUT / "vectors.bin").stat().st_size / 1e6
    mb = (OUT / "meta.json").stat().st_size / 1e6
    print(f"Wrote web/data/: vectors.bin {vb:.1f}MB, meta.json {mb:.1f}MB, config.json")


if __name__ == "__main__":
    main()
