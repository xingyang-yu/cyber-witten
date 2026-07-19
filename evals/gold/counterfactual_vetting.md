# Counterfactual twin vetting (Experiment 1)

Each unsupported twin must be a premise **genuinely absent** from the shared
passages, so the correct model behaviour is refusal. Twins were vetted two ways,
independently, and the verdicts converged:

1. **Corpus grep** — searched the actual chunk text of each paper in the index
   for the twin's load-bearing terms.
2. **Adversarial model review** — GPT 5.6 Sol at max reasoning effort, with web
   access, fetched the real papers and judged each premise [GOOD]/[RISKY]/[BAD].

Result: **10 GOOD (kept), 2 RISKY (excluded), 0 BAD.**

## Kept (verdict = good)

| pair_id | twin premise (false) | corpus-grep evidence | model verdict |
|---|---|---|---|
| jones-cs-wilson-loop | Jones polynomial detects the unknot (proved) | unknot detection is an open problem; not in paper | GOOD |
| topological-gravity-kdv | Witten *proves* the KdV tau-function statement | stated as conjecture; Kontsevich proved it | GOOD |
| mirror-a-b-model | proves every CY3 has a mirror partner | paper contrasts A/B models; no existence theorem | GOOD |
| information-theory-scope | sample-complexity bound for learning a quantum channel | essay covers entropy/DPI; no learning theory | GOOD |
| kapustin-witten-langlands-chain | geometric Langlands mathematically proved for arbitrary G | physical framework, not a proof | GOOD |
| seiberg-witten-n2-confinement | exact confining string tension in units of Lambda | "string tension" 0 hits; only BPS/central-charge "tension" | GOOD |
| chern-simons-analytic-continuation | Borel summability theorem for the perturbative series | "summability" 0, "resurgence" 0; only "Borel-Weil-Bott" | GOOD |
| donaldson-tqft-1988 | explicit K3 Donaldson invariants matched to known values | "K3" 0 hits; only general correlator formulas | GOOD |
| wormholes-averaging-n-2026 | theorem: Mellin averaging reproduces exact amplitude in ALL holographic CFTs | "all/every holographic" 0, "conjecture" 0; conditional language only | GOOD |
| gravity-von-neumann-algebras | observable algebra always type II_1 for any positive-Lambda spacetime | II_1, II_infinity, and type III all appear; classification is setup-dependent | GOOD |

## Excluded (verdict = risky)

| pair_id | why risky (both checks agreed) |
|---|---|
| susy-morse-mechanism | The paper really introduces the signed instanton-path count n(a,b) = sum ±1 over gradient trajectories. A model could reasonably answer with the index-difference-one condition and n(a,b), reading "formula" loosely. Too close to a real feature. |
| sigma-model-ricci-flow | The paper (Papadopoulos-Witten, 2404.19526) really discusses Perelman's monotonicity and "no periodic orbits", and adapts that argument. A model could plausibly describe it as a no-breather-style proof without hallucinating. Semantic boundary too thin. |

The two exclusions are themselves a finding: even a physicist-authored twin can
land inside the paper's real content. That is exactly the supported/unsupported
boundary the experiment probes, now visible in the probe design itself.

Provenance: model review session gpt-5.6-sol (max effort), 303,709 tokens,
2026-07-19.
