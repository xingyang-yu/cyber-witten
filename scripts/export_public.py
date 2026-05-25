"""Export a redistribution-safe subset of the corpus (arXiv-only).

Drops every chunk whose `arxiv_id` starts with `inspire:` — those are pre-arXiv
journal PDFs (1976-1991) that we ingested locally for retrieval quality but
cannot redistribute. The result is a HuggingFace-uploadable bundle: arXiv
content only, all of which is freely licensed by the authors via arXiv.

The FAISS vectors are sliced from the existing index (`reconstruct_n` over
each kept row) — no re-embedding required, runs in seconds.

Inputs:
    data/chunks/chunks.jsonl
    data/index/bge.faiss
    data/index/lookup.jsonl

Outputs (under data/public_export/):
    chunks.jsonl     filtered source of truth (re-indexable from scratch)
    bge.faiss        FAISS IndexFlatIP, dim=1024, arXiv rows only
    lookup.jsonl     parallel to bge.faiss
    manifest.json    counts, sha256 of each artifact, generation timestamp
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
IN_INDEX = ROOT / "data" / "index" / "bge.faiss"
IN_LOOKUP = ROOT / "data" / "index" / "lookup.jsonl"

OUT_DIR = ROOT / "data" / "public_export"
OUT_CHUNKS = OUT_DIR / "chunks.jsonl"
OUT_INDEX = OUT_DIR / "bge.faiss"
OUT_LOOKUP = OUT_DIR / "lookup.jsonl"
OUT_MANIFEST = OUT_DIR / "manifest.json"

EXCLUDE_PREFIX = "inspire:"


def is_public(arxiv_id: str) -> bool:
    return not arxiv_id.startswith(EXCLUDE_PREFIX)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lookup_rows = [json.loads(l) for l in IN_LOOKUP.open()]
    keep_idx = [i for i, r in enumerate(lookup_rows) if is_public(r["arxiv_id"])]
    drop_idx = [i for i, r in enumerate(lookup_rows) if not is_public(r["arxiv_id"])]

    src_index = faiss.read_index(str(IN_INDEX))
    if src_index.ntotal != len(lookup_rows):
        raise SystemExit(
            f"index/lookup mismatch: faiss has {src_index.ntotal} vectors "
            f"but lookup has {len(lookup_rows)} rows — rebuild the index"
        )

    dim = src_index.d
    kept = np.empty((len(keep_idx), dim), dtype="float32")
    for new_i, src_i in enumerate(keep_idx):
        kept[new_i] = src_index.reconstruct(src_i)

    out_index = faiss.IndexFlatIP(dim)
    out_index.add(kept)
    faiss.write_index(out_index, str(OUT_INDEX))

    with OUT_LOOKUP.open("w") as f:
        for i in keep_idx:
            f.write(json.dumps(lookup_rows[i]) + "\n")

    kept_ids = {lookup_rows[i]["arxiv_id"] for i in keep_idx}
    with IN_CHUNKS.open() as src, OUT_CHUNKS.open("w") as dst:
        for line in src:
            row = json.loads(line)
            if row["arxiv_id"] in kept_ids:
                dst.write(line)

    dropped_ids = {lookup_rows[i]["arxiv_id"] for i in drop_idx}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_index": {
            "vectors": src_index.ntotal,
            "papers": len({r["arxiv_id"] for r in lookup_rows}),
        },
        "public_export": {
            "vectors": out_index.ntotal,
            "papers": len(kept_ids),
            "dim": dim,
        },
        "excluded": {
            "reason": f"arxiv_id starts with '{EXCLUDE_PREFIX}' — pre-arXiv journal PDFs, not redistributable",
            "vectors": len(drop_idx),
            "papers": len(dropped_ids),
        },
        "artifacts": {
            OUT_CHUNKS.name: sha256_file(OUT_CHUNKS),
            OUT_INDEX.name: sha256_file(OUT_INDEX),
            OUT_LOOKUP.name: sha256_file(OUT_LOOKUP),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Exported public bundle to {OUT_DIR}/")
    print(f"  kept:     {len(keep_idx):5d} vectors / {len(kept_ids):3d} papers")
    print(f"  dropped:  {len(drop_idx):5d} vectors / {len(dropped_ids):3d} papers (inspire:*)")
    for name, digest in manifest["artifacts"].items():
        print(f"  {name:14s} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
