import modal,json
app=modal.App("plantid-t2diag")
vol=modal.Volume.from_name("plantid-vol2")
img=modal.Image.debian_slim(python_version="3.11").pip_install("pandas==2.2.3","pyarrow==17.0.0")
@app.function(image=img,volumes={"/v":vol},timeout=900)
def run():
    import pandas as pd,pyarrow.parquet as pq,glob,urllib.request,urllib.error,collections,json
    ok=set()
    for p in glob.glob("/v/traits/smoke/t2/shard_*.parquet"):
        t=pq.read_table(p).to_pydict(); ok|={a for a,b in zip(t["pid"],t["bytes"]) if b}
    sel=pd.read_parquet("/v/traits/smoke/t1/sel.parquet"); bad=sel[~sel.pid.isin(ok)].pid.tolist()
    C=collections.Counter(); ex=[]
    for pid in bad[:25]:
        res={}
        for ext in ["jpg","jpeg","png","JPG","JPEG","gif"]:
            try: r=urllib.request.urlopen(urllib.request.Request(f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{ext}",method="HEAD"),timeout=20); res[ext]=r.status
            except urllib.error.HTTPError as e: res[ext]=e.code
            except Exception as e: res[ext]=str(e)[:20]
        try:
            j=json.load(urllib.request.urlopen(f"https://api.inaturalist.org/v1/photos/{pid}",timeout=20)); u=j["results"][0]["url"] if j.get("results") else None
        except urllib.error.HTTPError as e: u=f"api{e.code}"
        except Exception as e: u=str(e)[:30]
        hit=[k for k,v in res.items() if v==200]; C["ext_ok:"+",".join(hit) if hit else "no_s3"]+=1; ex.append((pid,hit,str(u)[:90]))
    return {"n_bad":len(bad),"n_total":len(sel),"counter":dict(C),"examples":ex[:12]}
@app.local_entrypoint()
def main(): print("T2D",json.dumps(run.remote()))
