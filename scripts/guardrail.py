"""Live citation-grounding guardrail for the generation step.

The eval harness (`evals/`) measures grounding offline. This applies the same
check at serving time: after the model answers, verify every cited paper ID was
actually in the retrieved set; if not, tell the model exactly what it got wrong
and regenerate, up to a few attempts. This is what lets a smaller local model be
trusted as the default backend (see README "Known limitations" / "Future work").

Used by `ask.py --guardrail` and intended for the local QA app.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.validator import validate_citations  # noqa: E402

_CORRECTION = (
    "Your previous answer cited {bad}, which {is_are} NOT among the provided passage IDs. "
    "Every bracketed citation must be one of these exact IDs: {allowed}. "
    "Rewrite the answer citing ONLY those IDs. If a claim is not supported by the passages, "
    "drop it or say the passages do not cover it. Do not invent or recall other paper IDs."
)


def generate_grounded(backend, system, user, retrieved_ids, max_tokens=2048, max_retries=2):
    """Generate an answer, then enforce citation grounding with a retry loop.

    Returns `(answer, report)` where `report` is:
        {"grounded": bool, "attempts": int, "validation": <validate_citations dict>}

    `grounded` is True if the final answer cites only retrieved papers (a refusal
    with no citations counts as grounded). If it still violates after `max_retries`
    corrections, the last answer is returned with `grounded=False` so the caller
    can fail loudly rather than silently serve an ungrounded answer.
    """
    allowed = ", ".join(sorted(set(retrieved_ids))) or "(none)"
    answer = backend.generate(system, user, max_tokens=max_tokens)

    for attempt in range(1, max_retries + 2):
        v = validate_citations(answer, retrieved_ids)
        if not v["grounding_violation"]:
            return answer, {"grounded": True, "attempts": attempt, "validation": v}
        if attempt == max_retries + 1:
            return answer, {"grounded": False, "attempts": attempt, "validation": v}
        bad = v["invalid_citations"]
        correction = _CORRECTION.format(
            bad=", ".join(bad),
            is_are="is" if len(bad) == 1 else "are",
            allowed=allowed,
        )
        answer = backend.generate(system, f"{user}\n\n{correction}", max_tokens=max_tokens)
