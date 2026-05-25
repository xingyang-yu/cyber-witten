"""Health check for the Cyber-Witten RAG corpus and runtime."""
import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
INDEX_FILE = ROOT / "data" / "index" / "bge.faiss"
LOOKUP_FILE = ROOT / "data" / "index" / "lookup.jsonl"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
METADATA_FILE = ROOT / "data" / "metadata" / "papers.jsonl"


def line_count(path):
    with path.open() as f:
        return sum(1 for _ in f)


def status(label, ok, detail=""):
    tag = "OK" if ok else "FAIL"
    print(f"[{tag}] {label}{': ' + detail if detail else ''}")
    return ok


def warn(label, detail=""):
    print(f"[WARN] {label}{': ' + detail if detail else ''}")


def valid_year(value):
    return bool(re.fullmatch(r"(19|20)\d\d", str(value or "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="Load BGE and run one retrieval")
    ap.add_argument(
        "--question",
        default="What is the relation between Chern-Simons theory and the Jones polynomial?",
    )
    args = ap.parse_args()

    failures = 0

    for path in [INDEX_FILE, LOOKUP_FILE, CHUNKS_FILE, METADATA_FILE]:
        failures += not status(str(path.relative_to(ROOT)), path.exists())

    if failures:
        raise SystemExit(1)

    lookup_n = line_count(LOOKUP_FILE)
    chunks_n = line_count(CHUNKS_FILE)
    if args.embed:
        failures += not status(
            "lookup/chunks counts",
            lookup_n == chunks_n,
            f"lookup={lookup_n}, chunks={chunks_n}",
        )
    else:
        try:
            import faiss

            index = faiss.read_index(str(INDEX_FILE))
            failures += not status(
                "index/lookup/chunks counts",
                index.ntotal == lookup_n == chunks_n,
                f"faiss={index.ntotal}, lookup={lookup_n}, chunks={chunks_n}, dim={index.d}",
            )
        except Exception as exc:
            failures += 1
            status("FAISS readable", False, repr(exc))

    try:
        import torch
        import transformers

        status(
            "runtime imports",
            True,
            f"torch={torch.__version__}, transformers={transformers.__version__}",
        )
    except Exception as exc:
        failures += 1
        status("runtime imports", False, repr(exc))

    papers = [json.loads(line) for line in METADATA_FILE.open()]
    ids = [paper.get("arxiv_id") for paper in papers]
    failures += not status(
        "metadata ids unique",
        len(ids) == len(set(ids)),
        f"{len(papers)} papers, {len(set(ids))} unique ids",
    )

    bad_years = [
        (paper.get("arxiv_id"), paper.get("published", "")[:4], paper.get("title", ""))
        for paper in papers
        if not valid_year(paper.get("published", "")[:4])
    ]
    if bad_years:
        warn("metadata has nonstandard years", f"{len(bad_years)} records")
        for aid, year, title in bad_years[:5]:
            print(f"       {aid} {year} {title[:70]}")
    else:
        status("metadata years", True)

    if os.environ.get("ANTHROPIC_API_KEY"):
        status("ANTHROPIC_API_KEY", True, "set")
    else:
        warn("ANTHROPIC_API_KEY", "missing; retrieval works, answer generation will not")

    from bge_embed import resolve_model_path

    status("BGE model path", True, resolve_model_path())

    if args.embed:
        try:
            from bge_embed import encode_queries

            q_emb = encode_queries([args.question])
            import faiss

            index = faiss.read_index(str(INDEX_FILE))
            lookup = [json.loads(line) for line in LOOKUP_FILE.open()]
            failures += not status(
                "index/lookup/chunks counts",
                index.ntotal == lookup_n == chunks_n,
                f"faiss={index.ntotal}, lookup={lookup_n}, chunks={chunks_n}, dim={index.d}",
            )
            scores, idxs = index.search(q_emb, 3)
            status("BGE retrieval smoke test", True, f"query embedding={q_emb.shape}")
            for rank, idx in enumerate(idxs[0], 1):
                hit = lookup[int(idx)]
                print(
                    f"       {rank}. {scores[0][rank - 1]:.3f} "
                    f"{hit['arxiv_id']} {hit['title'][:80]}"
                )
        except Exception as exc:
            failures += 1
            status("BGE retrieval smoke test", False, repr(exc))

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
