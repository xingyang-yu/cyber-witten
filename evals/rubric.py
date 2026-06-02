"""Scoring rubric and report rendering.

Two kinds of score per answer:

  AUTOMATED  (from validator.py, no human needed)
    grounding_violation   cites a paper that was not retrieved -> True is bad
    expected_citation_recall, retrieval_recall   did it find/cite the right papers

  HUMAN  (physicist fills these; this is the part the FirstPrinciples role is about)
    correctness     is the physics right?            0 wrong / 1 partial / 2 correct
    faithfulness    every claim backed by a passage? 0 unsupported / 1 minor drift / 2 clean
    refusal_ok      (out_of_corpus probes only)      0 hallucinated / 1 hedged / 2 declined cleanly

Scores are deliberately a coarse 0-1-2 so review stays fast. The interesting
artifact is not the mean — it is the *failure taxonomy* in the `notes` field:
the subtle conceptual errors and minimal counterexamples a domain expert spots.
"""
from __future__ import annotations

import statistics
from typing import Iterable

HUMAN_FIELDS = ("correctness", "faithfulness", "refusal_ok")

SCALE = {
    "correctness": {0: "wrong / conceptual error", 1: "partially correct or incomplete", 2: "correct"},
    "faithfulness": {0: "claims not supported by passages", 1: "mostly grounded, minor drift", 2: "every claim supported"},
    "refusal_ok": {0: "fabricated an answer", 1: "hedged but overreached", 2: "declined / flagged unsupported"},
}


def blank_human_scores() -> dict:
    """Null placeholders for the physicist to fill in a run file."""
    return {f: None for f in HUMAN_FIELDS} | {"notes": ""}


def _mean(values: Iterable) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 3) if vals else None


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Per-(backend, condition) summary across one run's records.

    When a run includes the no-RAG baseline (`--baseline`), each backend appears
    twice — e.g. `anthropic·rag` and `anthropic·closed_book` — so RAG vs no-RAG
    line up in the table for the same model.
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = f"{r['backend']}·{r.get('condition', 'rag')}"
        groups.setdefault(key, []).append(r)

    summary = {}
    for backend, recs in groups.items():
        scored = [r for r in recs if "auto" in r and not r.get("error")]
        in_corpus = [r for r in scored if r.get("type") != "out_of_corpus"]
        out_corpus = [r for r in scored if r.get("type") == "out_of_corpus"]
        n = len(scored)

        summary[backend] = {
            "model": recs[0].get("model", "?"),
            "n_questions": len(recs),
            "n_scored": n,
            "n_errors": sum(1 for r in recs if r.get("error")),
            # --- automated, headline ---
            "grounding_violation_rate": _mean(
                [1.0 if r["auto"]["grounding_violation"] else 0.0 for r in scored]
            ),
            "mean_invalid_citations": _mean([r["auto"]["n_invalid"] for r in scored]),
            "uncited_answer_rate": _mean(
                [1.0 if r["auto"]["uncited"] else 0.0 for r in scored]
            ),
            # --- automated, in-corpus recall ---
            "mean_expected_citation_recall": _mean(
                [r["auto"].get("expected_citation_recall") for r in in_corpus]
            ),
            "mean_retrieval_recall": _mean(
                [r["auto"].get("retrieval_recall") for r in in_corpus]
            ),
            # --- human ---
            "mean_correctness": _mean([r["human"].get("correctness") for r in in_corpus]),
            "mean_faithfulness": _mean([r["human"].get("faithfulness") for r in scored]),
            "mean_refusal_ok": _mean([r["human"].get("refusal_ok") for r in out_corpus]),
        }
    return summary


_COLUMNS = [
    ("model", "model"),
    ("n_scored", "n"),
    ("grounding_violation_rate", "ground.viol↓"),
    ("mean_invalid_citations", "bad cites↓"),
    ("uncited_answer_rate", "uncited↓"),
    ("mean_retrieval_recall", "retr.recall↑"),
    ("mean_expected_citation_recall", "cite.recall↑"),
    ("mean_correctness", "correct↑"),
    ("mean_faithfulness", "faithful↑"),
    ("mean_refusal_ok", "refusal↑"),
]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_markdown(summary: dict[str, dict]) -> str:
    """A backend×condition comparison table. Arrows: ↓ lower is better, ↑ higher."""
    header = "| run | " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 1)
    rows = [header, sep]
    for backend, s in sorted(summary.items()):
        cells = [_fmt(s.get(key)) for key, _ in _COLUMNS]
        rows.append(f"| {backend} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_failure_lines(records: list[dict]) -> str:
    """One line per answer that violated grounding or scored low — the seed of
    a failure taxonomy. Sorted worst-first."""
    def severity(r):
        a = r.get("auto", {})
        h = r.get("human", {})
        return (
            a.get("grounding_violation", False),
            -(h.get("correctness") if isinstance(h.get("correctness"), int) else 9),
            a.get("n_invalid", 0),
        )

    flagged = [
        r for r in records
        if r.get("error")
        or r.get("auto", {}).get("grounding_violation")
        or (isinstance(r.get("human", {}).get("correctness"), int) and r["human"]["correctness"] < 2)
    ]
    flagged.sort(key=severity, reverse=True)

    lines = []
    for r in flagged:
        tag = f"{r['backend']}·{r.get('condition', 'rag')}"
        if r.get("error"):
            lines.append(f"- [{tag}] {r['qid']}: ERROR {r['error']}")
            continue
        a = r["auto"]
        bits = []
        if a["grounding_violation"]:
            # closed-book has no retrieved set, so every citation is from memory;
            # in RAG mode an out-of-set citation is an outright fabrication.
            kind = "memory cites" if r.get("condition") == "closed_book" else "fabricated cites (not retrieved)"
            bits.append(f"{kind} {a['invalid_citations']}")
        c = r["human"].get("correctness")
        if isinstance(c, int) and c < 2:
            bits.append(f"correctness={c}")
        note = r["human"].get("notes") or ""
        lines.append(f"- [{tag}] {r['qid']}: {'; '.join(bits)}" + (f" — {note}" if note else ""))
    return "\n".join(lines) if lines else "(no flagged answers)"
