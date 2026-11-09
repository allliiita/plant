from common import *
@app.function(image=cpu_img,volumes={"/v":vol},timeout=1800)
def select():
    import pandas as pd, numpy as np
    from huggingface_hub import hf_hub_download
    rng=np.random.default_rng(0)
    m=pd.concat([pd.read_csv(hf_hub_download(R,f"data/{s}/metadata.csv",repo_type="dataset")).assign(sp=s) for s in ['train','val','test']])
    m['split']=np.where(m.sp=='test','test','train')
    m=m[m.organ.isin(['flower','leaf','fruit','bark','habit'])]
    # all qualifying (species, organ) cells: >=5 train & >=3 test  (B2 feasibility threshold, pre-registered)
    c=m.groupby(['species','organ','split']).size().unstack('split').fillna(0)
    q=c[(c.train>=5)&(c.test>=3)].reset_index()
    feas=q.groupby('organ').species.nunique().to_dict(); print("B2 feasibility (#species with >=5 train & >=3 test):",feas)
    fl=c.reset_index(); fl=fl[(fl.train>=30)&(fl.test>=6)&fl.organ.isin(['flower','leaf'])]
    Fs=set(fl.groupby('species').organ.nunique().loc[lambda x:x==2].index); print("factorial F species:",len(Fs))
    keep=q.set_index(['species','organ']).index
    mm=m.set_index(['species','organ']); mm=mm[mm.index.isin(keep)].reset_index()
    F=mm.groupby(['species','organ','split'],group_keys=False).apply(lambda g:g.sample(min(len(g),30 if g.name[2]=='train' else 12),random_state=0)).assign(set='F')
    F['inFL']=F.species.isin(Fs)&F.organ.isin(['flower','leaf'])
    parts=[]
    for org,g in m.groupby('organ'): parts.append(g.sample(min(len(g),2500),random_state=1))
    O=pd.concat(parts).assign(set='O')
    sel=pd.concat([F,O]).drop_duplicates('file_name')
    sel['inF']=sel.file_name.isin(F.file_name); sel['inFL']=sel.file_name.isin(F[F.inFL].file_name); sel['inO']=sel.file_name.isin(O.file_name)
    sel=sel[['file_name','species','organ','split','inF','inFL','inO']].reset_index(drop=True)
    print('N',len(sel),'cells',F.groupby(['organ','split']).size().to_dict(),'FL',int(F.inFL.sum()),'O',O.organ.value_counts().to_dict())
    sel.to_parquet("/v/selection.parquet"); vol.commit()
@app.function(image=cpu_img,volumes={"/v":vol},timeout=1800,memory=8192)
def grab(fn):
    import pyarrow.parquet as pq, pandas as pd, pyarrow as pa, os
    from huggingface_hub import hf_hub_download
    want=set(pd.read_parquet("/v/selection.parquet").file_name)
    import pyarrow.compute as pc
    col=pq.read_table(hf_hub_download(R,fn,repo_type="dataset"),columns=['image']).column('image').combine_chunks()
    paths=col.field('path'); mask=pc.is_in(paths,value_set=pa.array(list(want)))
    rows=list(zip(pc.filter(paths,mask).to_pylist(),pc.filter(col.field('bytes'),mask).to_pylist()))
    os.makedirs("/v/imgs",exist_ok=True)
    pq.write_table(pa.table({'file_name':[a for a,_ in rows],'bytes':[b for _,b in rows]}),f"/v/imgs/{os.path.basename(fn)}"); vol.commit()
    return len(rows)
@app.local_entrypoint()
def main():
    from huggingface_hub import list_repo_files
    select.remote()
    fs=[f for f in list_repo_files(R,repo_type="dataset") if f.endswith('.parquet')]
    print("grabbed",sum(grab.map(fs)))
