import json,sys
R=json.load(open(sys.argv[1])); MDE=json.load(open(sys.argv[2])); T=["Height","LeafArea","SLA","Leaf_N"]
f=lambda v: "NA" if v is None else f"{v:+.3f}"
ci=lambda d: "NA" if d[0] is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
L=["## Primary test — unseen species, genus-imputed stratum (ΔR² = image − genus imputation)","| trait | n | R² imputation | R² image | ΔR² [95% CI] | MDE (ρ .3/.5/.7) | incremental ΔR² [95% CI] |","|---|---|---|---|---|---|---|"]
for t in T:
    e=R["Species"][f"genus_imputed|{t}"]; m=MDE[t]["MDE"]
    L.append(f"| {t} | {e['n']} | {e['r2_imp']:.3f} | {e['r2_img']:.3f} | {ci(e['d_img_minus_imp'])} | {m['rho0.3']}/{m['rho0.5']}/{m['rho0.7']} | {ci(e['d_inc_minus_imp'])} |")
L+=["","## Strata within unseen-species CV (ΔR² image − imputation)","| trait | stratum | n | R² imp | R² img | ΔR² [CI] |","|---|---|---|---|---|---|"]
for t in T:
    for sn in ["genus_imputed","family_imputed","order_or_global","pooled"]:
        k=f"{sn}|{t}"
        if k in R["Species"]: e=R["Species"][k]; L.append(f"| {t} | {sn} | {e['n']} | {e['r2_imp']:.3f} | {e['r2_img']:.3f} | {ci(e['d_img_minus_imp'])} |")
L+=["","## Decay with taxonomic distance (pooled; ΔR² image − imputation; incremental)","| trait | level held out | R² imp | R² img | ΔR² [CI] | incremental ΔR² [CI] |","|---|---|---|---|---|---|"]
for t in T:
    for lv in ["Species","genus","family","order"]:
        e=R[lv][f"pooled|{t}"]; L.append(f"| {t} | {lv} | {e['r2_imp']:.3f} | {e['r2_img']:.3f} | {ci(e['d_img_minus_imp'])} | {ci(e['d_inc_minus_imp'])} |")
L+=["","## Organ control (genus-imputed stratum)","| trait | weighted − unweighted [CI] | top-10 − unweighted [CI] |","|---|---|---|"]
for t in T:
    e=R["Species"][f"genus_imputed|{t}"]; L.append(f"| {t} | {ci(e['d_weighted_minus_unweighted'])} | {ci(e['d_top10_minus_unweighted'])} |")
L+=["","## Warm-up (seen species, image level)","| trait | R² lookup (nearest species centroid) | R² direct regression |","|---|---|---|"]
for t in T: w=R["warmup_seen"][t]; L.append(f"| {t} | {w['r2_lookup_img']:.3f} | {w['r2_regress_img']:.3f} |")
L+=["","## Selection-bias checks","| trait | Spearman(dead rate, log trait) | sensitivity: low-dead genus-imputed n | ΔR² [CI] |","|---|---|---|---|"]
for t in T:
    s=R["sensitivity_low_dead"][t]; L.append(f"| {t} | {R['dead_vs_trait_spearman_full'][t]:+.3f} | {s['n']} | {ci(s['d_img_minus_imp'])} |")
L+=["","## Internal checks (Holm on one-sided bootstrap p; triggers only)", "```", json.dumps(R["internal_checks"],indent=1), "```", f"n_species={R['n_species']}, n_images={R['n_images']}, imputation levels (Species CV): {R['Species']['imputation_level_counts']}"]
open(sys.argv[3],"w").write("\n".join(L)); print("\n".join(L))
