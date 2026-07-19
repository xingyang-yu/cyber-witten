# Experiment 1: retrieval-aware counterfactual refusal

Prompted by a reader (Samuel Larson) asking whether "retrieval destroys refusal"
was measured against a **controlled** counterfactual: the *same* passages paired
with a supported and an unsupported claim, so the only variable is premise
support.

## Design

10 pairs (`evals/gold/counterfactual.jsonl`, vetted in `counterfactual_vetting.md`).
Each pair retrieves passages **once** from the supported question (production
rerank path), then runs BOTH arms against those *identical* passages:

- **supported arm** — the real gold question (should answer + cite).
- **unsupported arm** — a false-premise twin on the same topic (should refuse).

Because the passages are byte-identical across the arms, the pre-generation
refusal gate scores them identically and cannot tell them apart — so any refusal
difference is the model reacting to the *premise*, not to different retrieval.

Scored `refusal_ok` (0/1/2) on the unsupported arm under the **strict** convention
(credit requires a user-visible warning; a fluent pivot that never flags the
premise = 0). Each answer was scored independently by the author and by a
GPT-5.6-Sol blind pass at max effort; the two converged, human is final.

## Result (refusal_ok on the unsupported arm, mean over 10 pairs)

| condition | base qwen2.5:7b | distilled cyber-witten-7b |
|---|---|---|
| RAG, naked | **0.30** | **1.10** |
| RAG + guardrail | **0.40** | **1.10** |

Anchor: the same base model **closed-book** scored refusal ≈ 1.25 on the original
out-of-corpus probes.

## Findings

1. **Retrieval destroys refusal, under a clean control.** With passages held
   identical to a supported question, the base model's refusal collapses from
   ~1.25 (closed-book) to 0.30 (naked) / 0.40 (guardrail). This is the controlled
   version of the headline result: not a retrieval-quality artifact, since the
   passages are the same ones that correctly support the twin question.

2. **Distillation *raises* refusal here (≈3.7x), the opposite of the probe-set
   regression.** The distilled model learned to say "these passages do not contain
   X" explicitly (seiberg-witten, mirror, information-theory: clean refusals). This
   does **not** match the blog's "distillation regressed refusal" claim, which was
   measured on the original out-of-corpus probes — the effect of distillation on
   refusal is **non-monotonic and probe-construction-dependent**, which only a
   controlled counterfactual surfaces.

3. **Distillation polarizes, and the guardrail can backfire on it.** Where the
   distilled model does not refuse, it answers the false premise *confidently and
   with citations* — chern-simons naked asserts "the perturbative expansion is
   Borel summable, this conclusion is directly stated in 1001.2933" and fabricates
   a supporting quote the paper does not contain. And the citation guardrail
   **shuffles** which pairs refuse rather than uniformly helping: it forced
   refusals on jones-cs and chern-simons (0→2) but *regressed* donaldson and
   kapustin (2→0), where citation pressure pushed the model to assert content to
   justify a citation. Net mean unchanged (1.10), but the composition moves.

## Excluded

2 of the 12 drafted pairs were dropped as "risky" (`sigma-model-ricci-flow`,
`susy-morse-mechanism`): the paper really does discuss the thing the twin's false
premise gestures at (Perelman monotonicity; the signed instanton-path count), so
the supported/unsupported boundary was too thin. See `counterfactual_vetting.md`.
That the drafter's own twins can land inside the paper is itself on-theme.
