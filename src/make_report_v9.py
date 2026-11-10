import json,sys
R=json.load(open(sys.argv[1])); T=["Height","LeafArea","SLA","Leaf_N"]
ci=lambda d: "NA" if d is None or d[0] is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
L=["## A / v9: comparisons by held-out level (pooled)","| trait | level | R² photo | R² imput | R² stack | R² density-shrink | stack−photo | shrink−photo | shrink−imput | shrink−uniform (C1) |","|---|---|---|---|---|---|---|---|---|---|"]
for t in T:
    for lv in ["Species","genus","family","order"]:
        e=R[lv].get(f"pooled|{t}")
        if not e: continue
        L.append(f"| {t} | {lv} | {e['r2_img']:.3f} | {e['r2_imp']:.3f} | {e['r2_inc']:.3f} | {e['r2_blend']:.3f} | {ci(e['d_inc_minus_img'])} | {ci(e['d_blend_minus_img'])} | {ci(e['d_blend_minus_imp'])} | {ci(e['d_blend_minus_single_w'])} |")
L+=["","## Same, unseen-species CV by density stratum","| trait | stratum | n | R² photo | R² imput | R² shrink | shrink−photo | shrink−imput | shrink−uniform |","|---|---|---|---|---|---|---|---|---|"]
for t in T:
    for sn in ["genus_imputed","family_imputed","order_or_global"]:
        e=R["Species"].get(f"{sn}|{t}")
        if e: L.append(f"| {t} | {sn} | {e['n']} | {e['r2_img']:.3f} | {e['r2_imp']:.3f} | {e['r2_blend']:.3f} | {ci(e['d_blend_minus_img'])} | {ci(e['d_blend_minus_imp'])} | {ci(e['d_blend_minus_single_w'])} |")
L+=["","## Conformal (CV+): empirical coverage / mean width (log units); nominal 1−α, CV+ guarantee 1−2α"]
for a in ["alpha=0.05","alpha=0.1"]:
    for lv in ["Species","genus","family","order"]:
        C=R[lv]["conformal"][a]; L+=[f"### {lv} held out, {a} (nominal {C['nominal']:.2f}, guarantee ≥{C['cvplus_guarantee']:.2f})","| method | calibration | stratum | n | " + " | ".join(f"{t} cov/width" for t in T)+" |","|---|---|---|---|"+"---|"*4]
        for meth in ["blend","img"]:
            for mode in ["pooled","mondrian"]:
                for st,v in C[meth][mode].items():
                    L.append(f"| {meth} | {mode} | {st} | {v['n']} | " + " | ".join(f"{c:.3f}/{w:.2f}" for c,w in zip(v['coverage'],v['width']))+" |")
open(sys.argv[2],"w").write("\n".join(L)); print("\n".join(L[:40]))
