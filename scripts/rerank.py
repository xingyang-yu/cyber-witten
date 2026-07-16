"""Cross-encoder reranking for Cyber-Witten retrieval.

The bi-encoder (BGE cosine over FAISS) is fast but scores query and passage
independently, so a passage that is *about* the question ranks no higher than
one that merely *shares vocabulary* with it. A cross-encoder reads the
(question, passage) pair jointly and is much sharper — too slow to score 13k
chunks, but ideal for re-scoring a candidate pool: retrieve top-N by cosine,
rerank the pool, keep top-K.

Same design rules as bge_embed.py: `transformers` directly (no extra
dependency), lazy model load, MPS/CUDA/CPU auto-pick.

Model: BAAI/bge-reranker-v2-m3 (~2.3GB download on first use). Override with
RERANKER_MODEL_PATH or the model_path argument.
"""
from functools import lru_cache
import os

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512
DEFAULT_POOL = 100  # candidates fetched by cosine before reranking

# Pre-generation refusal: cross-encoder logits are absolute-ish, so the score of
# the BEST passage says whether the corpus covers the question at all. Measured
# on the gold set: out-of-corpus probes top out at 0.22 (three of four negative),
# in-corpus questions bottom out at 0.49 (most +1..+5). Threshold 0.0 is the
# conservative cut — it never comes near refusing an answerable question, and a
# missed refusal still falls through to the prompt + citation guardrail.
REFUSAL_THRESHOLD = 0.0
REFUSAL_TEXT = (
    "The retrieved passages do not cover this question — the corpus does not "
    "appear to contain relevant material (best relevance score {score:.2f}). "
    "Refusing before generation rather than risking an ungrounded answer."
)


def refusal_check(passages, threshold=REFUSAL_THRESHOLD):
    """Given rerank()-sorted (score, payload) passages, decide whether to refuse
    before generation. Returns (should_refuse, best_score)."""
    best = passages[0][0] if passages else float("-inf")
    return best < threshold, best


def _pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _load(model_path, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def rerank_scores(question, texts, batch_size=8, model_path=None, device=None):
    """Cross-encoder relevance score for each (question, text) pair.

    Returns a list of floats parallel to `texts` (higher = more relevant).
    """
    import torch

    model_path = model_path or os.environ.get("RERANKER_MODEL_PATH", RERANKER_MODEL)
    device = device or _pick_device()
    tokenizer, model = _load(model_path, device)

    scores = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                [question] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.view(-1)
            scores.extend(logits.float().cpu().tolist())
    return scores


def rerank(question, passages, top_k, **kwargs):
    """Re-order `passages` — a list of (cosine_score, payload) where payload has
    a "text" field, as returned by the retrieve() helpers — by cross-encoder
    relevance. Returns the top_k as (rerank_score, payload) tuples."""
    scores = rerank_scores(question, [p["text"] for _, p in passages], **kwargs)
    order = sorted(range(len(passages)), key=lambda i: -scores[i])
    return [(scores[i], passages[i][1]) for i in order[:top_k]]
