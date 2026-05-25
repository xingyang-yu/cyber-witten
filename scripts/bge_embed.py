"""BGE embedding helpers used by Cyber-Witten.

This intentionally uses `transformers` directly instead of
`sentence_transformers`. The latter imports scikit-learn and several evaluator
modules at import time, which is slow and unnecessary for this retrieval-only
project.
"""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_SEQ_LENGTH = 512
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Keep HuggingFace writes inside the project so imports do not try to migrate or
# update ~/.cache/huggingface in restricted/sandboxed runs.
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_model_path():
    """Return a local BGE snapshot path when available, else the HF model id."""
    explicit = os.environ.get("BGE_MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"BGE_MODEL_PATH does not exist: {path}")

    cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--BAAI--bge-large-en-v1.5"
        / "snapshots"
    )
    if cache.exists():
        snapshots = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snapshot in snapshots:
            if (
                (snapshot / "config.json").exists()
                and (snapshot / "tokenizer.json").exists()
                and (
                    (snapshot / "model.safetensors").exists()
                    or (snapshot / "pytorch_model.bin").exists()
                )
            ):
                return str(snapshot)

    return MODEL_NAME


@lru_cache(maxsize=2)
def _load_model(model_path, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    local_only = Path(model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModel.from_pretrained(model_path, local_files_only=local_only)
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_texts(texts, batch_size=32, show_progress=False, device=None, model_path=None):
    """Encode texts with BGE CLS pooling and L2 normalization."""
    import torch
    import torch.nn.functional as F

    texts = list(texts)
    if not texts:
        return np.empty((0, 1024), dtype="float32")

    device = device or pick_device()
    model_path = model_path or resolve_model_path()
    tokenizer, model = _load_model(model_path, device)

    ranges = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm

        ranges = tqdm(ranges, total=(len(texts) + batch_size - 1) // batch_size)

    chunks = []
    with torch.no_grad():
        for start in ranges:
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            emb = output.last_hidden_state[:, 0]
            emb = F.normalize(emb, p=2, dim=1)
            chunks.append(emb.cpu().numpy().astype("float32"))

    return np.vstack(chunks)


def encode_queries(questions, **kwargs):
    return encode_texts([BGE_QUERY_PREFIX + q for q in questions], **kwargs)
