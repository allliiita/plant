import modal,json
app=modal.App("plantid-ptn")
img=modal.Image.debian_slim(python_version="3.11").pip_install("pandas==2.2.3","pyarrow==17.0.0","numpy==1.26.4")
vol=modal.Volume.from_name("plantid-vol")
@app.function(image=img,timeout=3600,memory=16384,volumes={"/v":vol})
def stats():
    import urllib.request,urllib.parse,json,pandas as pd,numpy as np,os,io
    from concurrent.futures import ThreadPoolExecutor
    get=lambda u,t=120: urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"research-client"}),timeout=t)
    os.makedirs("/v/ptn",exist_ok=True); D=[]
    for sp in ["train","val"]:
        p=f"/v/ptn/{sp}.csv"
        if not os.path.exists(p): open(p,"wb").write(get(f"https://huggingface.co/datasets/ayushi3536/PlantTraitNet/resolve/main/{sp}/{sp}.csv",600).read())
        D.append(pd.read_csv(p).assign(src=sp))
    d=pd.concat(D); vol.commit()
    T=["Height","LeafArea","SLA","Leaf_N"]
    out={"n_img":len(d),"n_species":int(d.Species.nunique())}
    sp=d.groupby("Species").agg(n=("Image","size"),**{t:(f"{t}_median","first") for t in T},**{t+"_sd":(f"{t}_sd","first") for t in T}).reset_index()
    # GBIF taxonomy
    tp="/v/ptn/taxonomy.parquet"
    if os.path.exists(tp): tax=pd.read_parquet(tp)
    else:
        def match(nm):
            for _ in range(3):
                try: r=json.load(get("https://api.gbif.org/v1/species/match?"+urllib.parse.urlencode({"name":nm.replace("_"," "),"kingdom":"Plantae"}),30)); return {"Species":nm,"genus":r.get("genus"),"family":r.get("family"),"order":r.get("order"),"rank":r.get("rank"),"conf":r.get("confidence")}
                except Exception: pass
            return {"Species":nm}
        with ThreadPoolExecutor(4) as ex: tax=pd.DataFrame(list(ex.map(match,sp.Species)))
        tax.to_parquet(tp); vol.commit()
    sp=sp.merge(tax,on="Species",how="left"); out["tax_matched"]=int(sp.family.notna().sum())
    sp=sp[sp.family.notna()&sp.order.notna()&sp.genus.notna()]
    pool=sp[sp.n>=30].copy(); out["pool_ge30"]=len(pool)
    out["pool_taxa"]={"genera":int(pool.genus.nunique()),"families":int(pool.family.nunique()),"orders":int(pool.order.nunique())}
    for lv in ["genus","family","order"]:
        c=pool[lv].value_counts(); out[f"pool_species_in_{lv}_with_>=2sp"]=int(pool[pool[lv].isin(c[c>=2].index)].shape[0]); out[f"{lv}s_with_>=3sp"]=int((c>=3).sum())
    # variance components (log): between-species var of medians; mean within (from TRY sd on log approx: sd/median)
    vc={}
    for t in T:
        x=pool[[t,t+"_sd"]].dropna(); x=x[x[t]>0]; lm=np.log(x[t]); within=float(np.nanmean((x[t+"_sd"]/x[t])**2))
        vc[t]={"n":len(x),"between_logvar":float(lm.var()),"within_log_approx":within,"icc_approx":float(lm.var()/(lm.var()+within))}
    out["var_components"]=vc
    # imputation baseline R2 under taxonomic 5-fold CV (no images): folds grouped at level L; impute from training species at level>=L with fallback (N>=2 rule)
    def impute_r2(t,group_level,N=2,seed=0):
        x=pool.dropna(subset=[t]); x=x[x[t]>0].copy(); x["y"]=np.log(x[t])
        groups=np.array(sorted(x[group_level].unique())); perm=groups[np.random.default_rng(seed).permutation(len(groups))]
        fold={g:i%5 for i,g in enumerate(perm)}; x["fold"]=x[group_level].map(fold); x["pred"]=np.nan; x["used"]=None
        for f in range(5):
            tr=x[x.fold!=f]; te=x.fold==f; p=pd.Series(np.nan,index=x.index[te]); u=pd.Series(None,index=x.index[te],dtype=object)
            for lv in ["genus","family","order"]:
                g=tr.groupby(lv).y.agg(["mean","size"]); g=g[g["size"]>=N]["mean"]
                v=x.loc[te,lv].map(g); fill=p.isna()&v.notna(); p[fill]=v[fill]; u[fill]=lv
            fill=p.isna(); p[fill]=tr.y.mean(); u[fill]="global"
            x.loc[te,"pred"]=p; x.loc[te,"used"]=u
        y=x.y.values; r2=1-((y-x.pred.values)**2).sum()/((y-y.mean())**2).sum()
        return float(r2),x.used.value_counts().to_dict(),len(x)
    pool=pool.reset_index(drop=True)
    out["imputation_r2"]={t:{lv:impute_r2(t,lv) for lv in ["Species","genus","family","order"]} for t in T}
    pool.to_parquet("/v/ptn/pool_species.parquet"); vol.commit()
    return out
@app.local_entrypoint()
def main():
    r=stats.remote(); json.dump(r,open("/home/user/plantid/traits/ptn_stats.json","w"),indent=1); print("PTN",json.dumps(r))
