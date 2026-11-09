# program.md — Plant species×organ identifiability (AI-Scientist-v2 staged tree search + autoresearch discipline)
Seed claim: In fine-grained plant ID, the linearly identifiable signal of BioCLIP vs DINOv3 is asymmetric across the species and organ axes.
Data: PlantNet-300K (HF mikehemberger/plantnet300K), official train(+val) vs test split; organ labels from metadata.
  F-set: species with >=30 train & >=6 test images for BOTH flower and leaf (cap 30 train / 12 test per cell).
  O-set: 5 organs (flower/leaf/fruit/bark/habit), <=2500 imgs/organ, organ probe uses species-disjoint GroupKFold.
Models (frozen): bioclip(B/16), bioclip2(L/14), clip_b16, clip_l14, dinov3_b, dinov3_l, dinov2_b.
Readout: L2-regularised multinomial logistic regression on standardized features (lambda picked on 20% train holdout); 4 depths x {CLS, patch-mean} + final.
Rules: fixed protocol per stage; every run appended to results.tsv (stage, node, hypothesis, metric, value, CI, status keep/discard/buggy, note);
  a sub-claim is KEPT only if effect CI excludes 0 and it replicates across both model sizes (B and L) where available; otherwise DISCARD (still logged).
Stages: S1 initial asymmetry; S2 robustness (layers, token, readout, sizes); S3 sub-hypotheses (cross-organ transfer, variance decomposition, erasure, text supervision control, zero-shot); S4 ablations/failure conditions (gray, lowres, few-shot).

## v2 (2026-11-09) — reframed, pre-registered before any result
RQ: When identifying plant species from an arbitrary-organ image, under which conditions are a taxonomy-text contrastive representation (BioCLIP) and a self-distilled visual representation (DINOv3) complementary vs redundant, and is this determined by how each compresses vs retains organ variation?
Pairs: B = (bioclip, dinov3_b); L = (bioclip2, dinov3_l). Δ = acc(concat probe) − max(single); paired bootstrap 95% CI over test images.
Phenomenon  P1: within-organ Δ>0 (CI excl. 0, > null-Δ) and Δ_within − Δ_cross ≥ 1pp.
  sanity: acc_DINO(within) ≥ acc_BioCLIP(within) − 10pp, else switch main pair to L and report.
  null-Δ: same-model concat (final ⊕ 3/4-depth CLS) for each model; P1 Δ must exceed max null-Δ.
Mechanism   M1: step0 error overlap (only-one-correct rate, oracle). Organ-specificity index I = [acc_D(o→o) − acc_D(o'→o)] − [acc_B(o→o) − acc_B(o'→o)];
              prediction: Δ_within correlates positively with I across (test organ × k∈{1,3,10,30} × size) = 16 points. ANOVA organ-variance share reported descriptively.
            M2 (round 2, only if P1 holds): LEACE-erase organ subspace from DINOv3 → within-organ Δ drops (CI excl. 0), DINOv3 own acc ~unchanged.
            M3 (round 3): CLIP replaces BioCLIP; wording "consistent / inconsistent with taxonomic-text-anchor hypothesis" only.
Boundary    B1: Δ vs k. B2: rare organs; organ kept only if ≥20 species with ≥5 train & ≥3 test (threshold fixed now).
Robustness (not mainline): depth/token, gray/lowres, DINOv2, B vs L replication, contamination caveat (BioCLIP-1 vs -2).
Falsification is layered: P1 fails → "redundant" phenomenon bullet; M1 descriptive finding kept; M2 skipped; boundary per B1/B2.

## v3 (2026-11-09) — data switched to PlantCLEF2024/25 single-plant set; added before download, no results seen
Data: lab.plantnet.org PlantCLEF2024 single-plant metadata; Pl@ntNet-sourced images only for train/test (medium /image/m/).
Species pool: every organ (flower, leaf, bark, fruit|habit) >=20 distinct Pl@ntNet obs. Target 300 species (BioImages-overlap species first, then seeded random).
Per species x organ: 20 train obs + 8 test obs, one image per obs.
D1 author-disjoint feasibility: count species where test authors ∩ train authors = ∅ and each organ still has >=8 test & >=20 train obs.
   If < 250 feasible species: relax (allow author overlap for the shortfall species), flag them, report both.
D2 contamination control: probe trained on Pl@ntNet train; compare (BioCLIP-2 − BioCLIP-1) on GBIF-sourced vs Pl@ntNet-sourced test images,
   matched on same species × organ × short-side resolution bin. If matching impossible → organ-stratified only; an inconsistent
   sign across strata is NOT counted as contamination evidence.
P1 sanity (replaces 10pp rule): report each model's species-probe accuracy per setting; settings where DINOv3 < BioCLIP − 15pp →
   Δ flagged "weak-model-dominated", excluded from the P1 criterion.
k grid now {1,3,10,20} (20 train obs per cell). Round-1 scope = flower/leaf settings only; rare organs reserved for B2.
M2 downgrade: if LEACE erasure lowers DINOv3's own species accuracy by >3pp, conclusion downgrades to
   "organ subspace also carries species information"; may NOT claim "complementarity comes from organ-specific detail".
BioImages external check: wording limited to "directionally consistent"; never "replicated" (only 2 species with >=5 distinct individuals for both flower & leaf).
Order: download → D1 → features → P1 + M1 (+D2, cheap) → report → decide M2/B2 → M3 last.
