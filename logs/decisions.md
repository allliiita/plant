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
- R-bug: S2 resume trusted shard existence; shards from an aborted run with a different S1 were reused (87 imgs missing) → caught by S2 gate. Fix: shard reused only if its image list equals the expected slice; stale/out-of-range shards rewritten/removed. Gate unchanged.
- reliability(): lazy np.load kept files open → vol.reload ConflictError. Fix: load-copy-close. R3/R4 stages already completed are reused via manifests.
- RELIABILITY: R1 pass; R2 pass (after S1 determinism fix; 150/150 byte-identical after 2 injected crashes); R3 pass (S4 B and L rows.json byte-identical);
  R4 pass as defined (same 500 imgs extracted twice: max diff 0.0 for bioclip and dinov3_l). Extra check vs full-run features: bioclip 1.9e-5 abs (dinov3_l 0.0);
  likely batch-shape-dependent kernels (last batch 52 vs 64) — not verified, reported as-is.
- B2 result: B pair within-rare Δ>0 at k>=10 (bark +3.7/+3.8pp); L pair no gain (Δ<=0), DINOv3-L weak-flagged on bark/habit. Pre-registered B2 criterion (replicate in B and L) NOT met.
- EXPLORATORY (post hoc, not pre-registered): across all within-organ k>=10 settings (n=20), fusion Δ vs accuracy gap (accA−accD): Spearman −0.86 (p=1e-6); all k>=10 settings n=52: −0.90.
  Gain appears only when gap <= ~8pp. Competing account to H*: complementarity governed by performance parity, organ acts via parity.
- s6 cost control: precomputed index; secondary LR classifier restricted to k=5 × {M0,M1,M3,M3a} (go criteria use proto only; unchanged).
- Early stop (agreed): G1 is AND over primaries → evaluate dinov3_b first; if G1 fails, stop (NO-GO), rerun dinov3_b only for byte-check; bioclip optional for write-up.
- POST HOC (user-approved, flagged): adapter+proto vs strongest training-free baseline (M1+LR) with shared cluster bootstrap; computed by rerunning primaries with tag _ph after the main run; main numbers must reproduce exactly.
- STOP (user): secondary backbones, byte reruns, and _ph rerun cancelled. Post hoc strongest-baseline CI computed from summaries as conservative unpaired approximation (independence assumption).
- smoke T2 crash: stale legacy shards; also tier2 'short' set could include stale tier2 results (nondeterminism). Fixed: decide from tier1 statuses only; delete shards without status column.
- v7.3 decided autonomously per user instruction.
- smoke T4 crash: bootstrap on empty sensitivity stratum (path untested: synthetic dead rates were constant). Guard + test t_small_and_varied + mutation M14.
