import pandas as pd,numpy as np,json,sys
p=pd.read_parquet(sys.argv[1]); d=pd.concat([pd.read_csv(sys.argv[2],usecols=["Species"]+[c for c in pd.read_csv(sys.argv[2],nrows=1).columns if c.endswith(("lowerbound","upperbound"))]).drop_duplicates("Species")])
T=["Height","LeafArea","SLA","Leaf_N"]; out={}
p=p.merge(d,on="Species",how="left")
def impute(x,level,N=2,seed=0):
    groups=np.array(sorted(x[level].unique())); perm=groups[np.random.default_rng(seed).permutation(len(groups))]; fold=x[level].map({g:i%5 for i,g in enumerate(perm)})
    pred=pd.Series(np.nan,index=x.index)
    for f in range(5):
        tr=x[fold!=f]; te=fold==f; pr=pd.Series(np.nan,index=x.index[te])
        for lv in ["genus","family","order"]:
            g=tr.groupby(lv).y.agg(["mean","size"]); g=g[g["size"]>=N]["mean"]; v=x.loc[te,lv].map(g); pr=pr.fillna(v)
        pred[te]=pr.fillna(tr.y.mean())
    return pred
rng=np.random.default_rng(0)
for t in T:
    x=p[(p[t]>0)].copy(); x["y"]=np.log(x[t])
    iqr_sd=(np.log(x[f"{t}_upperbound"].clip(lower=1e-9))-np.log(x[f"{t}_lowerbound"].clip(lower=1e-9)))/1.349
    within=float(np.nanmedian(iqr_sd**2)); between=float(x.y.var())
    res={"n_species":len(x),"between_logvar":between,"within_logvar_iqr":within,"icc":between/(between+within)}
    for level in ["Species","genus"]:
        pi=impute(x,level); e=x.y-pi; vy=x.y.var(); r2i=1-(e**2).mean()/vy; n=len(x)
        mde={}
        for rho in (0.3,0.5,0.7):
            found=None
            for dR in np.arange(0.005,0.30,0.005):
                # image-model residual with target R2 = r2i + dR and corr(e_img, e_imp)=rho
                s_img=np.sqrt(max((1-(r2i+dR))*vy,1e-9)); hits=0
                for rep in range(200):
                    z=rng.standard_normal(n); ei=rho*s_img*(e.values/e.std())+np.sqrt(1-rho**2)*s_img*z
                    dd=(e.values**2-ei**2)/vy           # per-species contribution to ΔR2
                    # species-level bootstrap one-sided test, alpha=0.05/4 (Holm worst case)
                    m=dd.mean(); se=dd.std(ddof=1)/np.sqrt(n); hits+= (m/se)>2.50
                if hits/200>=0.8: found=round(float(dR),3); break
            mde[f"rho{rho}"]=found
        res[f"{level}_imputation_r2"]=round(float(r2i),3); res[f"{level}_MDE_dR2_power80"]=mde
    out[t]=res; print(t,json.dumps(res),flush=True)
json.dump(out,open(sys.argv[3],"w"),indent=1)
