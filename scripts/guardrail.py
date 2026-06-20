"""Live citation-grounding guardrail for the generation step.

The eval harness (`evals/`) measures grounding offline. This applies the same
check at serving time and corrects the model when it falls short.

Two failure modes, both caught here:
  - FABRICATION: cites a paper ID that was not in the retrieved set.
  - OMISSION:    gives a substantive answer with no citations at all. (This is
                 the common small-model failure: it ignores the cite-by-ID
                 instruction and just writes prose.)

A genuine refusal ("the passages do not support an answer") is allowed to be
uncited, so it is not treated as omission. For questions where no answer is
expected (out-of-corpus probes), pass `require_citation=False` so a correct
refusal is never pushed to cite.

Used by `ask.py --guardrail`, `evals/run_eval.py --guardrail`, and the local app.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.validator import validate_citations  # noqa: E402

_REFUSAL_RE = re.compile(
    r"do(?:es)? not (?:contain|support|cover|address|provide|mention|discuss|include)"
    r"|cannot (?:be )?(?:answer|determine|found)"
    r"|not enough (?:information|context|detail)"
    r"|no (?:relevant )?(?:information|passages?|details?)"
    r"|passages? do not|insufficient (?:information|context|detail)"
    r"|unable to answer|outside (?:the )?corpus",
    re.IGNORECASE,
)

_FABRICATED = (
    "Your previous answer cited {bad}, which {is_are} NOT among the provided passage IDs. "
    "Every bracketed citation must be one of these exact IDs: {allowed}. "
    "Rewrite citing ONLY those IDs; drop any claim the passages do not support. "
    "Do not invent or recall other paper IDs."
)
_UNCITED = (
    "Your previous answer included NO citations. Every non-trivial claim must end with a "
    "passage ID in square brackets, e.g. [hep-th/9112056], drawn ONLY from: {allowed}. "
    "Rewrite the answer with those inline citations. If the passages genuinely do not "
    "support an answer, say so explicitly instead."
)


def _diagnose(answer: str, retrieved_ids, require_citation: bool):
    """Return (problem_kind, validation_dict). problem_kind is 'fabricated',
    'uncited', or None when the answer is acceptably grounded."""
    v = validate_citations(answer, retrieved_ids)
    if v["grounding_violation"]:
        return "fabricated", v
    if require_citation and v["uncited"] and not _REFUSAL_RE.search(answer):
        return "uncited", v
    return None, v


def generate_grounded(backend, system, user, retrieved_ids, max_tokens=2048,
                      max_retries=2, require_citation=True):
    """Generate, then enforce citation grounding with a corrective retry loop.

    Returns `(answer, report)` where report is:
        {"grounded": bool, "problem": "fabricated"|"uncited"|None,
         "attempts": int, "validation": <validate_citations dict>}

    If still ungrounded after `max_retries` corrections, the last answer is
    returned with grounded=False so the caller can fail loudly.
    """
    allowed = ", ".join(sorted(set(retrieved_ids))) or "(none)"
    answer = backend.generate(system, user, max_tokens=max_tokens)

    for attempt in range(1, max_retries + 2):
        kind, v = _diagnose(answer, retrieved_ids, require_citation)
        if kind is None:
            return answer, {"grounded": True, "problem": None, "attempts": attempt, "validation": v}
        if attempt == max_retries + 1:
            return answer, {"grounded": False, "problem": kind, "attempts": attempt, "validation": v}
        if kind == "fabricated":
            bad = v["invalid_citations"]
            correction = _FABRICATED.format(
                bad=", ".join(bad), is_are="is" if len(bad) == 1 else "are", allowed=allowed,
            )
        else:  # uncited
            correction = _UNCITED.format(allowed=allowed)
        answer = backend.generate(system, f"{user}\n\n{correction}", max_tokens=max_tokens)
