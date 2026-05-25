"""Embed all chunks with BAAI/bge-large-en-v1.5 (local) and build a FAISS index.

Outputs:
- data/index/bge.faiss     (FAISS IndexFlatIP, L2-normalized)
- data/index/lookup.jsonl  (parallel to index rows; the chunk records)

No API key required. First run downloads the model (~1.3GB) from HuggingFace.
On CPU: ~30-60 min for ~11k chunks. On Apple Silicon (MPS): much faster.
"""
import json
from pathlib import Path

from bge_embed import encode_texts, pick_device

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = INDEX_DIR / "bge.faiss"
LOOKUP_FILE = INDEX_DIR / "lookup.jsonl"

BATCH = 32


def main():
    chunks = [json.loads(l) for l in CHUNKS.open()]
    print(f"Embedding {len(chunks)} chunks with BAAI/bge-large-en-v1.5...")

    device = pick_device()
    print(f"Device: {device}")

    texts = [c["text"] for c in chunks]
    arr = encode_texts(
        texts,
        batch_size=BATCH,
        show_progress=True,
        device=device,
    )
    print(f"\nEmbedding tensor: {arr.shape}")

    import faiss

    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, str(INDEX_FILE))

    with LOOKUP_FILE.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"\nIndex: {INDEX_FILE} ({index.ntotal} vectors, dim {arr.shape[1]})")
    print(f"Lookup: {LOOKUP_FILE}")


if __name__ == "__main__":
    main()
