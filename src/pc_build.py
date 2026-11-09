from common import *
@app.function(image=cpu_img,volumes={"/v":vol},timeout=3600,memory=32768,cpu=4)
def select():
    import pandas as pd,numpy as np,json
    m=pd.read_csv("/v/pc24_meta.csv",sep=';',usecols=['image_name','organ','species_id','obs_id','author','species','dataset','url'],dtype=str)
    m=m[m.organ.isin(['flower','leaf','bark','fruit','habit'])].copy(); m['author']=m.author.fillna('NA')
    m['bin']=m.species.str.extract(r'^([A-Z][a-z]+ [a-z\-]+)')[0]
    pn=m[m.dataset=='plantnet'].drop_duplicates('obs_id')        # one image per obs
    c=pn.groupby(['species_id','organ']).size().unstack(fill_value=0)
    for o in ['fruit','habit']: c[o]=c.get(o,0)
    pool=c[(c.flower>=20)&(c.leaf>=20)&(c.bark>=20)&((c.fruit>=20)|(c.habit>=20))].index
    rng=np.random.default_rng(0); rep={'pool_species':len(pool)}
    def split_species(g):
        orgs=[o for o in ['flower','leaf','bark','fruit','habit'] if (g.organ==o).sum()>=28]
        if not {'flower','leaf','bark'}<=set(orgs) or not ({'fruit','habit'}&set(orgs)): return None
        au=g.author.value_counts().sort_values(); au=au.sample(frac=1,random_state=0).sort_values(kind='stable')  # small authors first
        test=set()
        for a in au.index:
            need=[o for o in orgs if ((g.organ==o)&g.author.isin(test)).sum()<8]
            if not need: break
            if ((g.organ.isin(need))&(g.author==a)).any(): test.add(a)
        ok=[o for o in orgs if ((g.organ==o)&g.author.isin(test)).sum()>=8 and ((g.organ==o)&~g.author.isin(test)).sum()>=20]
        if not ({'flower','leaf','bark'}<=set(ok) and ({'fruit','habit'}&set(ok))): return None
        return test,ok
    bio=set(json.load(open("/v/bio_species.json")))
    feas={}
    for sid,g in pn[pn.species_id.isin(pool)].groupby('species_id'):
        r=split_species(g)
        if r: feas[sid]=r
    rep['D1_author_disjoint_feasible']=len(feas)
    bins=pn.drop_duplicates('species_id').set_index('species_id').bin
    fl=list(feas); ov=[s for s in fl if bins[s] in bio]; rest=[s for s in fl if s not in set(ov)]
    rng.shuffle(rest); chosen=(ov+rest)[:300]; rep['chosen']=len(chosen); rep['bio_overlap_chosen']=len([s for s in chosen if s in set(ov)])
    rep['relaxed']=0
    if len(feas)<250: rep['relaxed']=1   # (flag only; handled manually if triggered)
    rows=[]
    for sid in chosen:
        test,ok=feas[sid]; g=pn[pn.species_id==sid]
        for o in ok:
            t=g[(g.organ==o)&g.author.isin(test)].sample(8,random_state=0); tr=g[(g.organ==o)&~g.author.isin(test)].sample(20,random_state=0)
            rows+= [t.assign(split='test',src='plantnet'),tr.assign(split='train',src='plantnet')]
    sel=pd.concat(rows)
    # D2 contamination set: GBIF-sourced test images for the same species x organ cells (up to 8, authors not in train)
    gb=m[(m.dataset=='gbif')&m.species_id.isin(chosen)].drop_duplicates('obs_id'); cells=set(zip(sel.species_id,sel.organ))
    gb=gb.loc[np.array([(s,o) in cells for s,o in zip(gb.species_id,gb.organ)],dtype=bool)]; rep["gbif_imgs_avail"]=len(gb)
    gb=gb.sample(frac=1,random_state=0); gb=gb[gb.groupby(['species_id','organ']).cumcount()<8].assign(split='test_gbif',src='gbif')
    sel=pd.concat([sel,gb]).reset_index(drop=True)
    sel['dl']=np.where(sel.src=='plantnet',sel.url.str.replace('/image/o/','/image/m/'),sel.url.str.replace('/original.','/medium.'))
    rep['n_img']=len(sel); rep['by_split_organ']={f'{a}|{b}':int(v) for (a,b),v in sel.groupby(['split','organ']).size().items()}
    rep['gbif_cells_with>=3']=int((gb.groupby(['species_id','organ']).size()>=3).sum())
    sel.to_parquet("/v/pc_sel.parquet"); vol.commit(); return rep
@app.function(image=modal.Image.debian_slim().pip_install("pyarrow","pillow").add_local_python_source("common","probe"),volumes={"/v":vol},timeout=3600,memory=8192,max_containers=4,retries=1)
def fetch(chunk):
    import urllib.request,urllib.error,io,os,time,random,pyarrow as pa,pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    p=f"/v/pcimg/{chunk[0][0]}.parquet"; have={'idx':[],'bytes':[],'short':[]}
    if os.path.exists(p): have=pq.read_table(p).to_pydict()
    done=set(have['idx']); todo=[r for r in chunk if r[0] not in done]
    if not todo: return 0,len(chunk)
    def get(r):
        i,u=r
        for t in range(6):
            try:
                b=urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'research-client'}),timeout=30).read()
                im=Image.open(io.BytesIO(b)).convert('RGB'); w,h=im.size; im.thumbnail((800,800)); o=io.BytesIO(); im.save(o,'JPEG',quality=90)
                return i,o.getvalue(),min(w,h)
            except urllib.error.HTTPError as e:
                if e.code==404: break
                time.sleep(min(60,3*2**t)+random.random())
            except Exception: time.sleep(2*2**t)
        return i,None,0
    with ThreadPoolExecutor(4) as ex: res=[r for r in ex.map(get,todo) if r[1]]
    for r in res: have['idx'].append(r[0]); have['bytes'].append(r[1]); have['short'].append(r[2])
    pq.write_table(pa.table(have),p); vol.commit()
    return len(res),len(todo)
@app.function(image=cpu_img,volumes={"/v":vol},timeout=600)
def chunks(n=500):
    import pandas as pd
    d=pd.read_parquet("/v/pc_sel.parquet").dl.tolist(); L=list(enumerate(d)); return [L[i:i+n] for i in range(0,len(L),n)]
@app.local_entrypoint()
def main(skip_select:int=0):
    import json
    rep={} if skip_select else select.remote(); print("SELECT",json.dumps(rep,default=str),flush=True)
    if rep: json.dump(rep,open("../results/audit_pc_select.json","w"),default=str)
    ok=tot=0
    for r in fetch.map(chunks.remote(),return_exceptions=True):
        if isinstance(r,Exception): print("CHUNKERR",str(r)[:200],flush=True); continue
        ok+=r[0]; tot+=r[1]
    print("FETCHED",ok,tot,flush=True)
