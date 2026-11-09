# Decision trail
- D1 (author-disjoint): 843 pool species → 657 feasible ≥250 → no relaxation. 300 chosen (91 BioImages-overlap). 40,444 imgs.
- D2 (contamination): only 12 GBIF-sourced test imgs fall in chosen species×organ cells (2 cells ≥3) → matched design infeasible inside the main subset.
  Decision: D2 moved to a dedicated side set (species with ≥8 GBIF & ≥8 Pl@ntNet test imgs in the same organ), run after round 1; main pair stays B (BioCLIP-1+DINOv3-B).
- fruit test 1968 / habit 2384 < 2400: some species use only one of fruit|habit (by design).
- R1 v1 (node buggy): DINOv3-L fp16 overflow → chance accuracy, ANOVA NaN. Fix: DINOv3 extracted in fp32 (both sizes), rerun both pairs.
- R1 v1 B-pair: null-Δ (same-model concat) = −27pp at k=1 → concat-probe Δ at low k is dominated by readout capacity/regularisation.
  Deviation (flagged, not pre-registered): add late-fusion Δ (mean of the two softmax outputs) as robustness readout; concat-Δ stays the pre-registered primary.
- D2: matched cells = 2 → infeasible in main subset (as predicted); side-set deferred.
- Pipeline gate fix: S4 chance-gate scoped to easiest setting (within-organ, max k); hard settings (k=1 cross-organ) legitimately near chance on 10-image smoke test. Threshold unchanged (2x smoke / 3x full).
- R2 found S1 nondeterminism (author tie order via Python str hash; PYTHONHASHSEED env ineffective). Fix: explicit sorted+seeded order; new S1 gate re-runs selection in subprocess with PYTHONHASHSEED=12345 and requires identical hash.
- Speed: froze legacy author-disjoint split (pc_sel.parquet, sha-pinned in CFG) as full-mode S1 input instead of recomputing; validated by same S1 gates. Fresh S1 algorithm (deterministic) kept for smoke + future data. Avoids ~2h re-download; parallelism unchanged.
