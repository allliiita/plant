from common import *
@app.function(image=cpu_img,timeout=900,memory=8192)
def bm():
    import pandas as pd,urllib.request,urllib.parse,io,json
    g=lambda f: pd.read_csv(io.BytesIO(urllib.request.urlopen(f"https://raw.githubusercontent.com/baskaufs/Bioimages/master/{f}",timeout=120).read()),dtype=str,low_memory=False)
    out={}
    try: r=urllib.request.urlopen("https://rdf.library.vanderbilt.edu/sparql?query="+urllib.parse.quote("ASK{}"),timeout=30); out['sparql']=(r.status,r.read(120).decode())
    except Exception as e: out['sparql']=str(e)[:120]
    im,de,na=g("images.csv"),g("determinations.csv"),g("names.csv")
    out['cols']={'images':list(im.columns),'det':list(de.columns),'names':list(na.columns)}; out['n_img']=len(im)
    vcol=[c for c in im.columns if 'view' in c.lower() or 'cvterm' in c.lower()]; out['view_cols']=vcol
    if vcol: out['views_top']=im[vcol[0]].value_counts().head(40).to_dict()
    im.to_parquet("/tmp/x.parquet")
    return out,im.head(2).to_dict('records'),de.head(2).to_dict('records'),na.head(1).to_dict('records')
@app.local_entrypoint()
def main():
    import json; r=bm.remote(); json.dump(r,open("../results/audit_bioimages_raw.json","w"),default=str); print(json.dumps(r,default=str)[:5000])
