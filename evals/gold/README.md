# Gold set — authoring guide

This file is the physics half of the eval. The harness can measure *grounding*
(did the model cite only retrieved papers?) on its own; it cannot measure
whether the physics is **right**. That judgement is the gold set — and it is the
exact skill the FirstPrinciples Staff Physicist role is hiring for: *defining
what "good" looks like and benchmarking against it.*

Aim for **~15–25 questions**. Depth and adversarial design beat volume.

## Format

`gold_set.jsonl` — one JSON object per line:

| field | required | meaning |
|---|---|---|
| `qid` | yes | short stable id, e.g. `cs-jones-01` |
| `question` | yes | the question, as a user would actually ask it |
| `type` | yes | `in_corpus` (Witten's work covers it) or `out_of_corpus` (probe for graceful refusal) |
| `gold_answer` | for scoring | YOUR reference answer — the standard a model output is judged against |
| `expected_citations` | `in_corpus` | arXiv/inspire IDs a *correct* answer must rest on (what *should* ground it, not just what retrieval happened to return) |
| `key_claims` | optional | bullet facts a correct answer must contain — makes correctness scoring faster and more consistent |
| `notes` | optional | authoring rationale, known pitfalls, what a good answer must NOT do |

IDs use the corpus's three forms: `1106.4789` (modern arXiv), `hep-th/9112056`
(legacy arXiv), `inspire:193975` (pre-arXiv). Version suffixes (`v2`) are
ignored by the validator, so either form is fine.

## What makes a good question

- **Has a defensible right answer** grounded in specific Witten papers — you can
  name the paper(s) it should cite.
- **Discriminates** between a grounded answer and a fluent-but-wrong one. The
  best questions are ones a general LLM will answer confidently and subtly
  wrong from training memory.
- **Spans the corpus**: 1980s pre-arXiv work (SUSY breaking, the 1989 Jones
  paper), the 1990s string-duality era, recent JT-gravity / algebras work. Year
  coverage stresses retrieval and surfaces OCR-quality issues on early scans.
- **Includes ~3–5 `out_of_corpus` probes**: questions just outside Witten's
  work (experimental results, other authors' theorems, post-corpus events). A
  rigorous system declines; a hallucinating one invents. This directly tests
  the project's central claim.

Consider a few **adversarial pairs**: a real Witten result vs. a plausible
misattribution ("Didn't Witten prove X?" where X is someone else's / false).

## Scoring (after a run)

`python -m evals.run_eval --providers anthropic,ollama` writes a run file with
`human` scores left null. Fill them in (0/1/2 — see `evals/rubric.py` for the
scale):

- `correctness` — is the physics right? (in_corpus)
- `faithfulness` — is every claim supported by a shown passage, with no drift?
- `refusal_ok` — did it correctly decline? (out_of_corpus only)
- `notes` — **the most valuable field.** The subtle conceptual error, the
  missing assumption, the minimal counterexample. This is the failure taxonomy.

Then re-render: `python -m evals.run_eval --report <run-file>`.
