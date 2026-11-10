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

## v4 (2026-11-09) — B2 pre-registration (after R1 results; before any B2 result)
Author audit (frozen split): species with >=5 train & >=3 test distinct authors: bark 300, fruit 246, habit 297 → all >=150 → all in main criterion.
B2 settings: train/test organs in {bark, fruit, habit}; within-organ Δ is the criterion; rare↔rare cross reported descriptively.
  Test restricted to species present in the training organ. k ∈ {1,3,10,20}. Same readouts (concat primary, late-fusion robustness), same null-Δ, same 15pp weak flag.
B2 criterion (H* prediction): within-organ Δ on a rare organ > 0 with CI excluding 0, replicated in B and L, and larger than the flower/leaf within-organ Δ at the same k.
  If met → RQ narrows to "complementarity appears only on organs poorly covered by the text anchor". If not → H* fully falsified for linear readout.

## v5 (2026-11-09) — crossed-pair test: performance parity vs organ (H*). Written BEFORE any crossed-pair result.
Motivation (exploratory, post hoc, from B/L rows): fusion Δ tracked the accuracy gap (Spearman −0.86, n=20 within-organ k>=10). Organ and gap were confounded.
Goal: separate "organ" from "gap" with pairs whose gaps differ in the opposite way across organs.

### Pairs
X1 = (bioclip, dinov3_l)   expected small gap on flower/leaf, negative gap on bark
X2 = (bioclip2, dinov3_b)  expected large gap everywhere, incl. bark
B, L (already run) are re-scored with the same gap estimator for the pooled analysis.

### Gap estimation — train split only; test split is used ONLY to compute Δ
gap(pair, organ, k) = CV-acc(A) − CV-acc(D), estimated by 5-fold cross-validation inside the TRAIN split of that organ,
  folds grouped by author (no author in both fold-train and fold-val), same probe/λ rule, same k subsampling as the test run.
No test-set accuracy enters any gap, threshold, or prediction.

### Settings
Within-organ, organs {flower, leaf, bark, fruit, habit}, k ∈ {10, 20} (k<=3 excluded: readout-capacity regime, shown by null-Δ).
Readouts: concat probe (primary), late fusion (robustness). 95% paired-bootstrap CI over test images.

### Manipulation check (must pass, else crossed test = INCONCLUSIVE)
MC1: X1 CV-gap <= 8pp on flower AND leaf at k=20.   MC2: X2 CV-gap >= 15pp on bark at k=20.
(8pp / 15pp fixed now; 8pp from the exploratory boundary, 15pp = existing weak-model flag.)

### Primary predictions (parity hypothesis P vs organ hypothesis H*)
T1 (X1, flower & leaf, k=20):  P predicts Δ > 0 with CI excluding 0;  H* predicts Δ CI includes 0 or < 0.
T2 (X2, bark, k=20):           P predicts Δ upper CI < +1pp;          H* predicts Δ > 0 with CI excluding 0.
T3 (pooled, 4 pairs × 5 organs × k∈{10,20}, n=40): regression Δ ~ CV-gap + rare_organ(bark/fruit/habit).
   P: CV-gap coefficient < 0 (CI excl. 0) AND rare_organ coefficient CI includes 0.  H*: rare_organ coefficient > 0 (CI excl. 0) after controlling gap.
   CIs by bootstrap over the 40 settings' test images is not valid (shared images) → use cluster bootstrap over species (1000 reps).
Verdict: P supported iff T1 and T2 and T3 all favour P. H* supported iff T2 and T3 favour H*. Otherwise "mixed", reported per test.
Bullets: P supported → mechanism layer rewritten as "complementarity is governed by performance parity; organ acts through parity".

### Secondary: WHY does the stable organ-invariance difference not become fusion gain? (each pre-specified, all from existing features)
W1 Redundant discriminative subspace: canonical correlations between the two models' species-discriminant subspaces (LDA top-50 dirs, fit on train) within each organ.
   Prediction if redundancy explains it: mean top-50 CCA ρ >= 0.7 on flower/leaf, lower on organs/pairs where Δ>0.
W2 Disagreement is not exploitable: among test images where exactly one model is correct, can the probes' confidences tell which one?
   AUROC of (maxprob_A − maxprob_D) for "A is the correct one". AUROC ≈ 0.5 → disagreements unexploitable by any confidence-based fusion; high AUROC but no Δ → readout limitation.
   Oracle-weighted late fusion (weight tuned on train CV) as upper bound of confidence-based fusion.
W3 Invariance ≠ within-organ information: the invariance index is measured CROSS-organ; fusion gain is measured WITHIN-organ.
   Test: cross-organ fusion Δ (flower→leaf, leaf→flower) vs within-organ Δ for B/L/X1/X2. If invariance mattered for fusion, cross-organ Δ should be where DINOv3's extra organ-specific info shows up;
   prediction under "invariance irrelevant to within-organ fusion": within- and cross-organ Δ both governed by the CV-gap (same regression as T3 including a cross-organ indicator, whose coefficient CI includes 0).
W4 Sample-size bound: Δ vs k slope (k=10→20) per setting; positive slope where gap small → gain may grow with data (stated as limitation, not tested beyond k=20).
Order: S3 features exist → smoke X1/X2 → full X1/X2 → CV-gap for B, L, X1, X2 → T1–T3 → W1–W4 → report.

## v6 (2026-11-09) — METHOD pilot: organ-invariance adapter for cross-organ few-shot ID of UNSEEN species. Written BEFORE any v6 result.
Task: support images of an unseen species from organ A, query images from organ B (A≠B); identify species among all unseen species having both organs.
Assumption (stated): organ label of each image is known at test time (Pl@ntNet users tag the organ).

### Splits (all seeded, fixed now)
- Species: 300 → sorted ids, rng(0) permutation → 150 SEEN / 150 UNSEEN. Of SEEN, 30 → VAL-species (hyper-parameter selection only), 120 → FIT-species.
- Images: only TRAIN-split images of SEEN species are used for fitting anything (adapter, organ means, LEACE).
- UNSEEN evaluation: support = TRAIN split of organ A; query = TEST split of organ B (author-disjoint from support by construction).
- Gates: |SEEN ∩ UNSEEN| = 0; no UNSEEN image or species in any fit/selection; VAL ∩ FIT = 0.
- Robustness (not in go criterion): genus-disjoint variant (UNSEEN species whose genus is absent from SEEN), reported if ≥40 such species.

### Methods (on frozen final features; each backbone separately)
M0 raw (standardized with SEEN statistics).
M1 organ-conditional centering (C5): subtract per-organ mean estimated on SEEN train images. Training-free.
M2 LEACE: closed-form erasure of the 5-way organ concept, fit on SEEN train images.
M3a generic adapter (control): same architecture/training as M3 but SupCon positives = same species, ANY organ, no cross-organ emphasis.
M3  cross-organ adapter (ours): linear projection d→256 (primary) + 2-layer MLP (ablation); loss = 0.5·SupCon(positives = same species, DIFFERENT organ) + 0.5·SupCon(positives = same species, any organ).
    Hyper-parameter grid (fixed now): lr ∈ {1e-3, 3e-3}, epochs ∈ {50, 200}, temperature ∈ {0.05, 0.1}; selected by VAL-species mean cross-organ accuracy (k=5). Same grid & rule for M3a.
Classifier: cosine nearest-prototype (primary, no tuning); L2 logistic probe (secondary).

### Evaluation
Organ pairs: all 20 ordered A≠B pairs (primary = macro mean over pairs); 5 within-organ pairs A=B (harm check only).
k ∈ {1, 5, 20}; primary k = 5. Support sampling seeds: 5 (k<20), 1 (k=20).
CIs: cluster bootstrap over UNSEEN species (1000 reps).
Backbones: primary dinov3_b and bioclip; secondary replication dinov3_l and bioclip2.

### GO criteria (pilot; all four required)
G1  M3 − M1 ≥ +3pp with CI excluding 0 (macro cross-organ, k=5, prototype classifier) on BOTH dinov3_b and bioclip.
G2  M3 > M1 on ≥3 of 5 target organs (B) for each primary backbone.
G3  within-organ accuracy of M3 ≥ M0 − 1pp (macro over 5 organs, k=5).
G4  M3 − M3a > 0 with CI excluding 0 on both primary backbones; if G1–G3 pass but G4 fails → claim downgraded to
    "metric learning on seen plant species transfers to unseen species", NOT "organ invariance".
NO-GO: G1 fails, or gains vanish on UNSEEN while present on VAL (shortcut). Then fall back to C1 (parity router) or mechanism-only write-up.
Multiplicity: only k=5 macro is confirmatory; all else descriptive.

### Reliability (same standards as R1–R4)
New stage s6_adapter in pipeline.py with manifest + gates (split gates above, finite outputs, raw-M0 within-organ acc > 3×chance);
smoke first (5+5 species); full run executed twice → byte-identical results.json required before reporting.
Outcome → bullets: GO → method bullet (+Y pp over frozen backbones, +Z pp over LEACE/centering, unseen species, reproducible);
                   G4 fail → "metric-learning transfer" bullet; NO-GO → negative-result bullet + mechanism bullet (invariance ⟂ fusion).
### v6.1 amendment (before any v6 result): taxonomy-stratified species split
Replaces "rng(0) random 150/150". Within each family (families with 1 species pooled into one 'other' stratum), species sorted by id,
permuted with rng(0), assigned alternately SEEN/UNSEEN (odd remainder → the currently smaller side, tie → SEEN).
VAL (30) = systematic stratified sample of SEEN: sort SEEN by (family, permuted order), take every 5th.
New gate: total-variation distance between SEEN and UNSEEN family distributions ≤ 0.10 (full mode).
Rationale: G1 is paired (same queries), so main-effect difficulty cancels; stratification removes taxonomy × adapter interaction confounds.
Smoke for s6 = full features restricted to the first 20 species (sorted id) and a 1-config grid; gates on split/finite only.
### v6.2 amendment (approved; before any full v6 result)
Family-level TVD ≤0.10 is infeasible by construction (82 families, 37 singletons → measured 0.18). Gate redefined at the stratification-unit level
(multi-species families + pooled singleton stratum): TVD_strata ≤ 0.10 (measured 0.06). Family-level TVD reported as-is.
Added robustness subset "shared_family": UNSEEN species whose family also has SEEN species (removes taxonomy mismatch); G1 recomputed there (proto, descriptive).
Reproducibility: full s6 rerun for the two primary backbones (dinov3_b, bioclip) must be byte-identical before reporting; secondary backbones rerun deferred (cost).

=====================================================================
## v7 (2026-11-10) — NEW PROJECT: Is photo-based plant trait prediction taxonomic recall?  (written before any image result)
RQ: When predicting plant functional traits from citizen-science photos (species-level TRY labels), how much of the apparent skill is
recoverable from taxonomy alone, and how does any image-specific signal decay with taxonomic distance?
No BioCLIP / DINO in this project.

### Data
- PlantTraitNet release (HF ayushi3536/PlantTraitNet, CC-BY-4.0), train+val CSV; labels = species-level TRY medians (paper: ≥3 TRY obs, 5–99th pct trimmed; NOT gap-filled).
  Traits: Height, LeafArea, SLA, Leaf_N; analysed on log scale.
- Species pool: ≥30 photos (2,982 spp). Taxonomy: GBIF species/match; keep rank=SPECIES & confidence ≥90; name audit (100 random spp): 100/100 exact, 8% synonyms.
  Synonym collision rule: if ≥2 PTN names map to one accepted GBIF species, drop all of them; report count.
- Photos: 30 per species (seeded rng 0) from iNaturalist open data, medium size. Limitation (stated, not corrected): ≥30-photo filter selects common/photogenic species.

### Organ handling (route 1, replaces hard filtering; no species dropped)
Frozen organ classifier (/v/organ_clf.pt; ImageNet-1k ViT-B linear probe on PlantCLEF organ tags; diagnostic: PlantCLEF tags agree with our organ definitions ≈47% (n=30, Claude-labelled, non-expert),
so a 90%-precision hard filter is unattainable). Each photo gets p_leaf, p_habit.
Primary: species prediction = unweighted mean over its photos. Secondary (same species set): p-weighted mean (Height→p_habit; others→p_leaf) and top-10-by-p mean.
Organ definitions (for audits): leaf = leaf dominant subject; whole-plant = whole plant or most aboveground parts, mixed flower+leaf+plant scenes → whole-plant; flower/fruit/bark close-ups and branch fragments → excluded class.

### Image model
Backbone vit_base_patch16_224.augreg_in1k (ImageNet-1k only; contains some plant classes, no TRY/AusTraits species labels). Frozen CLS features.
Image-level ridge regression to the species log-median; species prediction = mean of its photos' predictions. No species label as input or auxiliary task.
Ridge alpha chosen by NESTED CV: inside each outer training fold, inner 5-fold grouped by the same taxonomic level; alpha grid 10^[-1..4] (11 values).
Control (secondary, round 2): small CNN trained from scratch, same protocol, to test dependence of ΔR² on pretraining.

### Cross-validation & taxonomic imputation
Outer 5-fold CV grouped at level L ∈ {species, genus, family, order}; group→fold by seeded permutation.
Imputation uses ONLY outer-training species: genus mean (≥2 spp) → family (≥2) → order (≥2) → global; record level used per test species.

### Primary test (per trait, Holm over 4 traits, one-sided α=0.05)
Unseen-species CV (L=species), stratum "genus-imputed" test species (≈2,030): ΔR² = R²_image − R²_imputation > 0.
Inference: cluster bootstrap over genera (2,000 reps). Report ΔR², 95% CI, and simulated MDE (power 0.8): Height ~0.03, LeafArea ~0.05, SLA ~0.065, Leaf_N ~0.06 (× ~1.2 for n≈2,030).
Pre-specified readings: CI includes 0 → "no detectable photo signal beyond genus for this trait (MDE = …)"; significant >0 → quantify and read decay (secondary).
Each trait reported separately; no averaged ΔR².

### Secondary
S1 Strata within unseen-species CV: family-imputed, order-or-global-imputed, and pooled; ΔR² each.
S2 Decay: L = genus, family, order (imputation then starts at family / order / global); ΔR² per level.
S3 Incremental R²: ridge on image features predicting the imputation residual (nested CV) → ΔR²_incr over imputation.
S4 Warm-up (not criterion): seen species — species-classifier→lookup vs direct regression.
S5 Organ control: weighted / top-10 vs unweighted, same species.
Negative-control-like: Leaf_N. If image beats imputation on N → flag "anomalous, inspect" (background/geography cues; note leaf colour may carry genuine N signal) — not auto-labelled leakage.
Scale-calibration control: AusTraits leaf aspect ratio (244 spp, strict measured), own CV + imputation, reported separately, not compared numerically with main traits.

### Reliability
Run on new Modal account (volume plantid-vol2) in a manifest/gated pipeline (traits_pipeline.py): smoke (60 spp) → full → full rerun byte-identical before reporting.
Gates: name/taxonomy filters applied; photo fetch success ≥98% and every species ≥20 photos; features finite; ridge R² on seen-species warm-up > 0.
### v7.1 amendment (after smoke T2 gate failure; before any analysis result)
Smoke found ~8% of PlantTraitNet photo links dead (iNat 404; 24/25 sampled failures deleted, 1 uppercase .JPG).
Rule change: per species a seeded rank order of up to 45 candidates; tier-1 fetch ranks 0–29, tier-2 fetch further ranks only for species short of 30;
use the first 30 successful photos by rank. Gates: ≥98% of species reach 30 photos; every species ≥20. Dead-link rate reported. Limitation: deletion assumed unrelated to traits.
### v7.2 amendment (approved 2026-11-10; before any analysis result)
Data fixes: drop rows without iNat photo id (80,353; non-iNat sources); drop 2,077 photos shared by >1 species (congeneric pairs, e.g. Betula fruticosa|ovalifolia);
species that shared photos form link groups (12 groups, 25 spp) kept in the same fold at every CV level; labels = first non-null species median.
Pool = species with ≥30 downloadable-id iNat photos → 2,355 spp (839 genera, 163 families, 49 orders). Primary stratum (genus-imputed) n≈1,521.
Fetch: statuses ok/deleted/failed, ≤3 retry rounds, gate transient failures = 0 and every species ≥20 photos; fraction <30 reported (expected 3–4%);
dead-link rate and its Spearman with each trait reported. 48 threads, single container.
Primary-stratum MDE (power .8, one-sided α .0125, ρ .3/.5/.7): Height .020/.020/.015 (genus-imputation R² .88); LeafArea .05/.05/.045 (.66); SLA .09/.085/.07 (.42); Leaf_N .07/.065/.06 (.53).
Reporting format (fixed): per trait ΔR² + 95% CI + MDE; no binary significant/non-significant wording. If CI spans 0: "MDE = X, observed ΔR² = Y".
  Holm-adjusted bootstrap p is computed internally only to trigger the checks below.
Height rule: genus imputation already explains R² .88. If Height ΔR² CI excludes 0 (>0) → mandatory leakage checks BEFORE any interpretation:
  (i) photo/label co-source (same observation/collection event as TRY record), (ii) geographic/background cues (location-blocked re-test),
  (iii) unrecognised infraspecific structure within genera. Only after passing is it read as signal.
Leaf_N rule unchanged (anomalous → inspect).
Deferred to round 2 and stated as limitations in the report: no scratch-trained CNN control (cannot fully exclude plant signal inherited from ImageNet pretraining);
no AusTraits aspect-ratio control (cannot separate "images uninformative" from "photos lack scale").
### v7.3 amendment (2026-11-10; before any analysis result)
Smoke: deletions are clustered (whole observers), not independent: dead 10.4%, 11.7% spp <30 photos, min 14; dead rate vs log trait Spearman (n=60): Height .29, LA .16, N .12, SLA .01.
Changes: tier-2 uses ALL remaining photos of short species (seeded order; ranks 0–44 unchanged). Species with <20 photos after tier-2 are EXCLUDED (pre-registered rule);
gate: excluded fraction ≤5% (else stop). Report excluded vs retained mean log traits.
Selection-bias checks (mandatory): dead-rate vs trait Spearman on full set; sensitivity = primary stratum restricted to species with tier-1 dead rate ≤ median
(same out-of-fold predictions, no refit). If any |ρ| > 0.1, the report contains a dedicated selection-bias paragraph.

## v8 (2026-11-10) — taxonomy-informed stacking as method; confirmation on new backbones
A (EXPLORATORY, post hoc on ImageNet ViT-B): comparison "stacking (imputation + photo features) − photo-only" (d_inc_minus_img) at every CV level/stratum.
  Motivated by the v7 results; reported as exploratory for this backbone.
B (CONFIRMATORY, written before any result on these backbones): repeat the full v7 analysis with frozen features from
  (i) MAE ViT-L/16 (masked-reconstruction SSL, no labels) and (ii) OpenAI CLIP ViT-L/14 (image-text contrastive, no species-level supervision targets).
  [amended before any B result: DINOv2 replaced by MAE because the project excludes BioCLIP/DINO models]
  Same photos (t2 of the full run), same splits, same code. Pre-registered predictions, per backbone, per trait:
  P-B1 genus-imputed stratum: photo-only − genus imputation ΔR² < 0 (CI upper < 0) for Height and Leaf_N.
  P-B2 stacking − photo-only > 0 (CI lower > 0) at the Species level (pooled) for all four traits.
  P-B3 stacking − imputation > 0 (CI lower > 0) at the Species level (pooled) for all four traits.
  P-B4 decay: photo-only − imputation increases monotonically from Species → genus → family held out (point estimates) for Height, LeafArea, SLA.
  A prediction is "confirmed" only if it holds for BOTH new backbones. Report all, including failures.

## v9 (2026-11-10) — empirical-Bayes density-conditioned shrinkage + taxonomy-stratified CV+ (written before any v9 result)
Design: ImageNet ViT-B = DEVELOPMENT backbone (results already seen → exploratory). MAE ViT-L and CLIP ViT-L = CONFIRMATION backbones (no result seen).
Method (fixed parametrization): ŷ = w·photo + (1−w)·taxonomic imputation, w = σ(a + b·z(log(1+n_congeners_train)) + c·z(log(1+n_confamilials_train))),
  z = standardization on training species; (a,b,c) per trait by L-BFGS-B (init 0, bounds ±10) on INNER out-of-fold photo predictions and inner-fold imputation
  of training species only. No MLP. Ablation = uniform weight (intercept only).
Conformal: cross-conformal CV+ (calibration residuals only from species in other outer folds), Mondrian by density category (genus / family / order-or-global
  imputation; fallback to pooled calibration if <20 calibration species) vs pooled. Report for α=0.05 and α=0.10: nominal 1−α, CV+ guarantee 1−2α,
  empirical coverage and mean width, per stratum and CV level. Strata with n<50: described, not judged.
Confirmatory criteria (both MAE and CLIP must satisfy; Species-level CV, pooled, each trait, genus-clustered bootstrap 95% CI):
  C1 density-shrinkage − uniform-weight shrinkage ΔR² > 0 (CI lower > 0)   [co-primary: without it "density conditioning" is not claimed]
  C2 density-shrinkage − photo-only ΔR² > 0 (CI lower > 0)
  C3 density-shrinkage − imputation ΔR² > 0 (CI lower > 0)
  C4 Mondrian CV+ at α=0.05: empirical coverage ≥ 0.90 in every stratum with n ≥ 50; pooled CV+ coverage < Mondrian in the sparsest stratum.
If C1 fails but C2/C3 hold → claim reduces to "taxonomy+photo stacking/shrinkage helps", no density claim.
Not claimed: any theoretical coverage bound beyond the standard CV+ 1−2α guarantee (no ε(d) theorem unless proven separately).
Offline: suite 16/16; mutations M16 (density ignored) and M17 (Mondrian pooled) caught by their targeted tests.

## v10 (2026-11-10) — ONE-SHOT attempt to rescue C1 for SLA and Leaf_N (written before any v10 result)
Single change: add a third, trait-specific density feature to the logistic weight:
  z(log(SD of the trait among TRAINING congeners)) — within-genus dispersion of the species' genus in the training set
  (genera with <2 training species: family-level SD; then order; then global SD). Rationale: EB-optimal shrinkage depends on within- vs between-group variance.
Everything else identical (same features, splits, code, bootstrap). One full run on MAE and CLIP (and ImageNet dev, reported).
Decision rule (fixed): C1 (density − uniform, Species-level pooled, CI lower > 0) must hold on BOTH MAE and CLIP for a trait.
  If SLA and Leaf_N pass → method claim 4/4. If not → keep the v9 2/4 wording. No second attempt. Time cap: half a day.
All other v9 criteria re-reported; if any v9 criterion that passed now fails, v10 is rejected and v9 stands.
