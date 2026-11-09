import modal
app=modal.App("plant-inspect")
img=modal.Image.debian_slim().pip_install("pyarrow","pandas","huggingface_hub")
@app.function(image=img,timeout=900)
def f():
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq, pandas as pd
    R="mikehemberger/plantnet300K"
    fs=list_repo_files(R,repo_type="dataset"); print(len(fs),[x for x in fs if 'val' in x][:3])
    p=hf_hub_download(R,[x for x in fs if x.startswith('data/test-0')][0],repo_type="dataset")
    pf=pq.ParquetFile(p); print(pf.schema_arrow, pf.metadata.num_rows)
    t=pf.read_row_group(0).slice(0,3).to_pylist()
    for r in t: print(r['label'], r['image'].get('path'), len(r['image']['bytes']))
    for s in ['test','train','val']:
        m=pd.read_csv(hf_hub_download(R,f"data/{s}/metadata.csv",repo_type="dataset"))
        print(s,len(m),m.organ.value_counts().to_dict(),m.species.nunique())
        if s=='test': print(m.file_name.head(3).tolist())
