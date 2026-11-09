import modal,json
app=modal.App("plantid-dc3")
img=modal.Image.debian_slim(python_version="3.11").pip_install("pandas==2.2.3","pyarrow==17.0.0")
vol=modal.Volume.from_name("plantid-vol")
@app.function(image=img,timeout=600,memory=8192,volumes={"/v":vol})
def cols():
    import urllib.request,io,pyarrow.parquet as pq,os
    p="/v/austraits-7.0.0-flattened.parquet"
    if not os.path.exists(p):
        open(p,"wb").write(urllib.request.urlopen(urllib.request.Request("https://zenodo.org/records/15718081/files/austraits-7.0.0-flattened.parquet?download=1",headers={"User-Agent":"research-client"}),timeout=600).read()); vol.commit()
    f=pq.ParquetFile(p); names=f.schema_arrow.names
    out={"cols":names}
    for c in ["value_type","basis_of_record","entity_type","replicates"]:
        if c in names:
            t=f.read(columns=[c]).column(c).to_pandas(); out[c]=t.value_counts().head(12).to_dict()
    return out
@app.local_entrypoint()
def main(): print("C",json.dumps(cols.remote()))
