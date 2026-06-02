"""Citation grounding validator.

Pure text-in / metrics-out: no torch, no faiss, no network. This is the
automated, backend-independent half of the eval — it answers "did the model
cite only papers that were actually retrieved, and did it cite the papers the
expert expected?" without any human judgement.

A *citation* is a bracketed paper ID in one of the three forms the corpus uses:

    [1106.4789]        modern arXiv      (YYMM.NNNNN, optional vN)
    [hep-th/9112056]   legacy arXiv      (archive[.SS]/NNNNNNN, optional vN)
    [inspire:193975]   pre-arXiv paper   (synthetic INSPIRE recid)

Brackets that are not ID-shaped are ignored on purpose: passage text carries
`[cite]` placeholders (LaTeX citations were stripped during ingestion), and the
system prompt allows a `[outside corpus]` speculation marker. Neither is a
citation; counting them would be a false positive.
"""
from __future__ import annotations

import re

# --- ID grammars -----------------------------------------------------------
_ARXIV_NEW = r"\d{4}\.\d{4,5}"
_ARXIV_OLD = r"[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7}"
_INSPIRE = r"inspire:\d+"
_VERSION = r"(?:v\d+)?"

_ID_RE = re.compile(
    rf"^(?:{_ARXIV_NEW}|{_ARXIV_OLD}|{_INSPIRE}){_VERSION}$"
)
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_OUTSIDE_RE = re.compile(r"outside\s+(?:the\s+)?corpus", re.IGNORECASE)
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def normalize_id(paper_id: str) -> str:
    """Canonical form for comparison: lowercased, trailing version stripped."""
    pid = paper_id.strip().lower()
    return _VERSION_SUFFIX_RE.sub("", pid)


def is_paper_id(token: str) -> bool:
    """True if `token` already looks like a corpus paper ID (vs. a title)."""
    return bool(_ID_RE.match(token.strip()))


_is_id_shaped = is_paper_id  # internal alias


def extract_citations(answer: str) -> list[str]:
    """All ID-shaped tokens cited in `answer`, normalized, in order of first
    appearance (deduplicated). Comma/semicolon-separated IDs inside a single
    bracket (e.g. `[1106.4789, 1101.3216]`) are split out."""
    seen: dict[str, None] = {}
    for inner in _BRACKET_RE.findall(answer):
        for token in re.split(r"[,;]", inner):
            token = token.strip()
            if _is_id_shaped(token):
                seen.setdefault(normalize_id(token), None)
    return list(seen)


def has_outside_corpus_marker(answer: str) -> bool:
    return bool(_OUTSIDE_RE.search(answer))


def validate_citations(
    answer: str,
    retrieved_ids: list[str],
    expected_ids: list[str] | None = None,
) -> dict:
    """Score one answer's citation grounding.

    Args:
        answer: the model's generated text.
        retrieved_ids: arxiv_id of every passage shown to the model this turn
            (the only IDs it is allowed to cite).
        expected_ids: optional gold paper IDs a correct answer *should* rest on
            (physicist-authored). Enables recall metrics that separate a
            retrieval miss from a generation miss.

    Returns a flat dict of metrics. The headline is `grounding_violation`:
    True iff the answer cites a paper that was not retrieved (a fabricated or
    misremembered citation — the failure mode this whole project exists to
    make visible).
    """
    retrieved = {normalize_id(x) for x in retrieved_ids}
    cited = extract_citations(answer)
    cited_set = set(cited)

    valid = sorted(cited_set & retrieved)
    invalid = sorted(cited_set - retrieved)

    result = {
        "cited_ids": cited,
        "n_cited": len(cited),
        "valid_citations": valid,
        "invalid_citations": invalid,
        "n_invalid": len(invalid),
        "grounding_violation": bool(invalid),
        "uncited": len(cited) == 0,
        "marked_outside_corpus": has_outside_corpus_marker(answer),
    }

    if expected_ids is not None:
        expected = {normalize_id(x) for x in expected_ids}
        result["expected_ids"] = sorted(expected)
        if expected:
            cites_expected = sorted(cited_set & expected)
            retrieved_expected = sorted(retrieved & expected)
            result["cites_expected"] = cites_expected
            # Did the model cite the papers a correct answer should rest on?
            result["expected_citation_recall"] = len(cites_expected) / len(expected)
            # Did retrieval even surface them? Separates retrieval vs generation.
            result["retrieval_recall"] = len(retrieved_expected) / len(expected)
            result["retrieval_hit"] = bool(retrieved_expected)
        else:
            # Out-of-corpus probe: a correct answer cites nothing.
            result["expected_citation_recall"] = None
            result["retrieval_recall"] = None
            result["retrieval_hit"] = None

    return result
