import modal,json
app=modal.App("plantid-datacheck2")
img=modal.Image.debian_slim(python_version="3.11").pip_install("pandas==2.2.3","pyarrow==17.0.0")
@app.function(image=img,timeout=1800,memory=8192)
def check():
    import urllib.request,urllib.parse,json,pandas as pd,io
    get=lambda u,t=120: urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"research-client"}),timeout=t)
    import pyarrow.parquet as pq; a=pq.read_table(io.BytesIO(get("https://zenodo.org/records/15718081/files/austraits-7.0.0-flattened.parquet?download=1",600).read()),columns=["taxon_name","trait_name","value"]).to_pandas()
    out={"cols":list(a.columns)[:40],"n_rows":len(a)}
    tn=a.trait_name.value_counts(); want=[t for t in tn.index if any(k in t for k in ["leaf_length","leaf_width","leaf_area","plant_height","leaf_mass_per_area","leaf_N_per_dry_mass","leaf_shape","leaf_length_width","seed_dry_mass","leaf_dry_matter"])]
    out["trait_species_counts"]={t:int(a[a.trait_name==t].taxon_name.nunique()) for t in want[:15]}
    sp=a[a.trait_name.isin(["leaf_length","leaf_width"])].groupby("taxon_name").trait_name.nunique(); both=set(sp[sp==2].index)
    out["species_with_len_and_width"]=len(both)
    # iNat AU plant species photo counts via GBIF facet
    fac={}
    for off in range(0,30000,5000):
        j=json.load(get("https://api.gbif.org/v1/occurrence/search?"+urllib.parse.urlencode({"datasetKey":"50c9509d-22c7-4a22-a47d-8c48425ef4a7","kingdomKey":6,"mediaType":"StillImage","country":"AU","limit":0,"facet":"speciesKey","facetLimit":5000,"facetOffset":off})))
        cs=j["facets"][0]["counts"] if j["facets"] else []
        if not cs: break
        fac.update({c["name"]:c["count"] for c in cs})
    out["au_inat_species"]=len(fac); out["au_inat_species_ge30"]=sum(v>=30 for v in fac.values())
    # match AusTraits names (len+width) to GBIF keys (sample up to 1500)
    hit=0; hit30=0; tested=0; fams=set(); ords=set(); gens=set()
    for nm in sorted(both)[:1500]:
        try: r=json.load(get("https://api.gbif.org/v1/species/match?"+urllib.parse.urlencode({"name":nm,"strict":"true"}),30))
        except Exception: continue
        tested+=1
        k=str(r.get("speciesKey"))
        if k in fac:
            hit+=1
            if fac[k]>=30: hit30+=1; fams.add(r.get("family")); ords.add(r.get("order")); gens.add(r.get("genus"))
    out.update(matched_tested=tested,matched_in_inat=hit,matched_ge30_photos=hit30,ge30_genera=len(gens),ge30_families=len(fams),ge30_orders=len(ords))
    return out
@app.local_entrypoint()
def main(): print("DC2",json.dumps(check.remote()))
