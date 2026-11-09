from common import *
@app.function(image=cpu_img,timeout=3600,memory=32768,cpu=4,ephemeral_disk=60*1024)
def tol(obs_ids):
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    from huggingface_hub import hf_hub_download
    p=hf_hub_download("imageomics/TreeOfLife-200M","dataset/metadata/provenance.parquet",repo_type="dataset")
    f=pq.ParquetFile(p); print(f.schema_arrow)
    cols=[c for c in f.schema_arrow.names if c in ('source','data_source','publisher','source_id','source_url','datasetKey','gbifID','basisOfRecord')]
    n=0; pn=0; ex=[]; srcs={}; hit=set(); want=pa.array(list(obs_ids))
    for i in range(f.num_row_groups):
        t=f.read_row_group(i,columns=cols); n+=t.num_rows
        if 'publisher' in cols:
            pub=t.column('publisher').cast(pa.string()); msk=pc.fill_null(pc.match_substring_regex(pub,"(?i)pl@ntnet|plantnet"),False)
            s=t.filter(msk); pn+=s.num_rows
            if s.num_rows and len(ex)<5: ex+=s.slice(0,3).to_pylist()
            for c in ('source_id','source_url'):
                if c in cols and s.num_rows:
                    v=s.column(c).cast(pa.string())
                    for o in [str(x) for x in v.to_pylist()]:
                        tail=o.rstrip('/').split('/')[-1]
                        if tail in obs_ids: hit.add(tail)
        for c in ('source','data_source'):
            if c in cols:
                for k,v in zip(*pc.value_counts(t.column(c)).flatten()): srcs[str(k)]=srcs.get(str(k),0)+v.as_py()
    return dict(n=n,plantnet_publisher_rows=pn,examples=ex,sources=srcs,plantnet300k_obs_matched=len(hit),cols=cols)
@app.local_entrypoint()
def main():
    import pandas as pd,json
    m=pd.read_parquet("../data/meta_full.parquet")
    r=tol.remote(set(m.obs_id.astype(str))); print(json.dumps(r,indent=1,default=str)[:3000]); json.dump(r,open("../results/audit_tol.json","w"),default=str)
