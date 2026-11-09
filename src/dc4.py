import modal,json
app=modal.App("plantid-dc4")
img=modal.Image.debian_slim(python_version="3.11").pip_install("pandas==2.2.3","pyarrow==17.0.0","numpy==1.26.4")
vol=modal.Volume.from_name("plantid-vol")
@app.function(image=img,timeout=3600,memory=16384,volumes={"/v":vol})
def run():
    import urllib.request,urllib.parse,json,pyarrow.parquet as pq,pandas as pd,numpy as np
    from concurrent.futures import ThreadPoolExecutor
    get=lambda u,t=60: urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"research-client"}),timeout=t)
    a=pq.read_table("/v/austraits-7.0.0-flattened.parquet",columns=["taxon_name","trait_name","value","entity_type","value_type","basis_of_record","observation_id","taxon_rank"]).to_pandas()
    T=["leaf_length","leaf_width","leaf_area","plant_height","leaf_N_per_dry_mass","leaf_mass_per_area"]
    a=a[a.trait_name.isin(T)&(a.taxon_rank=="species")].copy(); a["v"]=pd.to_numeric(a.value,errors="coerce"); a=a[a.v>0]
    meas={"field","field_experiment","preserved_specimen","lab"}
    a["strict"]=a.entity_type.isin(["individual","population"])&a.value_type.isin(["raw","mean","median"])&a.basis_of_record.fillna("").str.split().apply(lambda s: bool(set(s)&meas))
    # aspect ratio from same observation
    w=a[a.strict&a.trait_name.isin(["leaf_length","leaf_width"])].pivot_table(index=["observation_id","taxon_name"],columns="trait_name",values="v",aggfunc="mean").dropna().reset_index()
    ar=pd.DataFrame({"taxon_name":w.taxon_name,"trait_name":"leaf_aspect_ratio","v":w.leaf_length/w.leaf_width,"strict":True})
    s=pd.concat([a[a.strict][["taxon_name","trait_name","v","strict"]],ar])
    out={"strict_species":{t:int(s[s.trait_name==t].taxon_name.nunique()) for t in T+["leaf_aspect_ratio"]},
         "loose_species":{t:int(a[a.trait_name==t].taxon_name.nunique()) for t in T}}
    # variance components on log scale (strict, species with >=3 records)
    vc={}
    for t in T+["leaf_aspect_ratio"]:
        d=s[s.trait_name==t].assign(l=lambda x: np.log(x.v)); g=d.groupby("taxon_name").l; n=g.size(); keep=n[n>=3].index; d=d[d.taxon_name.isin(keep)]
        if len(keep)<20: vc[t]=None; continue
        m=d.groupby("taxon_name").l.mean(); within=float(d.groupby("taxon_name").l.var().mean()); between=float(m.var())
        vc[t]={"n_sp":len(keep),"between":between,"within":within,"icc":between/(between+within)}
    out["var_components_log"]=vc
    # iNat AU species photo counts
    fac={}
    for off in range(0,40000,5000):
        j=json.load(get("https://api.gbif.org/v1/occurrence/search?"+urllib.parse.urlencode({"datasetKey":"50c9509d-22c7-4a22-a47d-8c48425ef4a7","kingdomKey":6,"mediaType":"StillImage","country":"AU","limit":0,"facet":"speciesKey","facetLimit":5000,"facetOffset":off})))
        cs=j["facets"][0]["counts"] if j["facets"] else []
        if not cs: break
        fac.update({c["name"]:c["count"] for c in cs})
    names=sorted(set(s.taxon_name))
    def match(nm):
        for t in range(3):
            try: r=json.load(get("https://api.gbif.org/v1/species/match?"+urllib.parse.urlencode({"name":nm,"strict":"true"}),30)); return nm,r
            except Exception: pass
        return nm,{}
    with ThreadPoolExecutor(4) as ex: M=dict(ex.map(match,names))
    tax=pd.DataFrame([{"taxon_name":k,"key":str(v.get("speciesKey")),"genus":v.get("genus"),"family":v.get("family"),"order":v.get("order"),"n_photo_au":fac.get(str(v.get("speciesKey")),0)} for k,v in M.items()])
    tax.to_parquet("/v/traits_taxonomy.parquet"); s.to_parquet("/v/traits_strict_records.parquet"); vol.commit()
    ok=tax[tax.n_photo_au>=30].taxon_name
    out["matched"]=int((tax.key!="None").sum())
    out["strict_species_ge30photos"]={t:int(s[(s.trait_name==t)&s.taxon_name.isin(ok)].taxon_name.nunique()) for t in T+["leaf_aspect_ratio"]}
    for t in ["leaf_length","leaf_aspect_ratio","plant_height","leaf_N_per_dry_mass"]:
        sp=set(s[(s.trait_name==t)&s.taxon_name.isin(ok)].taxon_name); x=tax[tax.taxon_name.isin(sp)]
        fc=x.family.value_counts(); oc=x.order.value_counts()
        out[f"tax_{t}"]={"genera":int(x.genus.nunique()),"families":int(x.family.nunique()),"orders":int(x.order.nunique()),
                         "families_ge3sp":int((fc>=3).sum()),"orders_ge3sp":int((oc>=3).sum()),"top_family_share":round(float(fc.iloc[0]/fc.sum()),3) if len(fc) else None}
    return out
@app.local_entrypoint()
def main(): print("D4",json.dumps(run.remote()))
