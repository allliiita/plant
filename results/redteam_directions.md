**C1 parity-aware gated fusion router.** (a) Prior: mixture-of-experts/gated fusion (e.g., learned modality gating); your Spearman −0.86 is the real hook—gate on *estimated per-organ accuracy gap*. Novelty risk moderate: gating is common, but accuracy-gap-conditioned routing is under-explored. (b) Failure: gate collapses to always/never-fuse; with only 20 train/species-organ, router overfits and the ~0 fusion gain on flower/leaf caps upside. (c) Realistic: +1–3 pts only on organs where models are near-parity; ~0 elsewhere. (d) Modest, safe, but low ceiling.

**C2 cross-organ invariance adapter.** (a) Prior: domain-adaptation/prototype alignment, cross-modal alignment, "universal" few-shot adapters; organ-as-domain alignment is plausible but I'm uncertain of a direct plant-specific precedent. Novelty risk: medium-high (generic alignment framing). (b) Failure: aligning prototypes on seen species may transfer organ-invariant *nuisance* (background, habit) rather than identity; unseen-species generalization is the hard part—adapter may just relearn seen-species shortcuts. (c) Realistic: recover part of the 0.88→0.55 drop, maybe +5–12 pts on cross-organ, but high variance. (d) Highest-upside, highest-risk, best fit to "when identifiability fails."

**C3 organ-conditioned text prompts.** (a) Prior: prompt engineering/CoOp-style; BioCLIP already taxonomic-text. (b) Failure: gains marginal; organ text already implicit in species codes. (c) +0–2 pts. (d) Cheap but likely null.

**C4 multi-organ aggregation.** (a) Prior: multi-view/multi-instance aggregation. (b) Failure: test sets are per-organ; aggregation only helps if multiple organs observed—may not match your protocol. (c) Small, protocol-dependent. (d) Weak fit.

**C5 organ-conditional centering/LEACE.** (a) Prior: LEACE (Belrose et al. 2023), concept erasure; training-free. (b) Failure: erasing organ direction may also erase identity signal; centering assumes linear organ subspace. (c) +2–6 pts if organ is linearly separable in features. (d) Excellent cheap baseline/ablation for C2—do it regardless.

**C6 benchmark.** (a) Prior: PlantCLEF, few-shot cross-domain benchmarks exist; author-disjoint cross-organ few-shot is a reasonable contribution but "benchmark-only" is weaker for a methods-focused applicant. (b) Failure: seen as incremental dataset paper. (c) N/A. (d) Package as contribution of C2, not standalone.

**Ranking: #1 C2, #2 C1** (C5 as mandatory baseline inside C2).

**Decisive go/no-go pilot for C2:** Train adapter on seen species (flower→leaf), evaluate cross-organ on held-out species vs. (i) no adapter, (ii) C5 centering. Go if adapter beats C5 by ≥3 pts on unseen-species cross-organ accuracy with consistent sign across ≥3 organs; no-go if it only matches C5 or gains vanish on unseen species (indicating shortcut learning).
