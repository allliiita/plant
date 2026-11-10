import json,sys
T=["Height","LeafArea","SLA","Leaf_N"]; out={}
for bb,path in [(a.split('=')[0],a.split('=')[1]) for a in sys.argv[1:]]:
    R=json.load(open(path)); r={}
    lo=lambda d: d[1] if d and d[1] is not None else None; hi=lambda d: d[2] if d and d[2] is not None else None
    for t in T:
        e=R["Species"][f"pooled|{t}"]; g=R["Species"][f"genus_imputed|{t}"]
        r[t]={"C1 shrink>uniform":lo(e["d_blend_minus_single_w"])>0, "C2 shrink>photo":lo(e["d_blend_minus_img"])>0, "C3 shrink>imput":lo(e["d_blend_minus_imp"])>0,
              "P-B2 stack>photo":lo(e["d_inc_minus_img"])>0, "P-B3 stack>imput":lo(e["d_inc_minus_imp"])>0,
              "vals":{"C1":e["d_blend_minus_single_w"][:3],"C2":e["d_blend_minus_img"][:3],"C3":e["d_blend_minus_imp"][:3],"stack-photo":e["d_inc_minus_img"][:3],"stack-imput":e["d_inc_minus_imp"][:3],"genus photo-imput":g["d_img_minus_imp"][:3]}}
        if t in ("Height","Leaf_N"): r[t]["P-B1 genus photo<imput"]=hi(g["d_img_minus_imp"])<0
        if t in ("Height","LeafArea","SLA"):
            d=[R[lv][f"pooled|{t}"]["d_img_minus_imp"][0] for lv in ["Species","genus","family"]]; r[t]["P-B4 decay monotone"]=d[0]<d[1]<d[2]
    C=R["Species"]["conformal"]["alpha=0.05"]["blend"]; strata=[k for k in C["mondrian"] if k!="all" and C["mondrian"][k]["n"]>=50]
    sparse=min(strata,key=lambda k:C["mondrian"][k]["n"])
    r["C4"]={"mondrian>=0.90 all strata n>=50":all(min(C["mondrian"][k]["coverage"])>=0.90 for k in strata),
             "pooled<mondrian in sparsest":all(p<m for p,m in zip(C["pooled"][sparse]["coverage"],C["mondrian"][sparse]["coverage"])),
             "sparsest":sparse,"mondrian":{k:C["mondrian"][k]["coverage"] for k in strata},"pooled":{k:C["pooled"][k]["coverage"] for k in strata}}
    out[bb]=r
json.dump(out,open("/home/user/plantid/results/judge_b.json","w"),indent=1)
crit=["C1 shrink>uniform","C2 shrink>photo","C3 shrink>imput","P-B2 stack>photo","P-B3 stack>imput","P-B1 genus photo<imput","P-B4 decay monotone"]
for c in crit:
    row=[]
    for t in T:
        v=[out[bb][t].get(c) for bb in out]
        if None in v: continue
        row.append(f"{t}:{'✅' if all(v) else '❌'}({'/'.join('Y' if x else 'N' for x in v)})")
    print(f"{c:26s}", "  ".join(row))
for k in ["mondrian>=0.90 all strata n>=50","pooled<mondrian in sparsest"]:
    v=[out[bb]["C4"][k] for bb in out]; print(f"C4 {k:35s}", '✅' if all(v) else '❌', '/'.join('Y' if x else 'N' for x in v))
