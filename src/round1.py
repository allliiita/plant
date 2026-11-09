from common import *
PAIRS={"B":("bioclip","dinov3_b"),"L":("bioclip2","dinov3_l")}
def load(n):
    import numpy as np
    Z=np.load(f"/v/pcfeats_{n}.npz"); fn=Z['orig__fn']
    L=sorted({k.split('__')[1] for k in Z.files if k.endswith('_cls')},key=lambda s:int(s[1:].split('_')[0]))
    return fn,{'final':Z['orig__final'].astype(np.float32),'aux':Z[f'orig__{L[-2]}'].astype(np.float32)}
@app.function(image=gpu_img,volumes={"/v":vol},gpu="T4",timeout=7200,memory=32768)
def r1(size):
    import numpy as np,pandas as pd,json
    from probe import lr_fit,boot_diff,anova_var
    A,D=PAIRS[size]; sel=pd.read_parquet("/v/pc_sel.parquet")
    fa,ZA=load(A); fd,ZD=load(D); assert (fa==fd).all()
    m=sel.loc[fa].reset_index(drop=True); M=(m.src=='plantnet').values&m.organ.isin(['flower','leaf']).values; m=m[M].reset_index(drop=True)
    XA,XD={k:v[M] for k,v in ZA.items()},{k:v[M] for k,v in ZD.items()}
    F={'A':XA['final'],'D':XD['final'],'AD':np.hstack([XA['final'],XD['final']]),'Anull':np.hstack([XA['final'],XA['aux']]),'Dnull':np.hstack([XD['final'],XD['aux']])}
    tr=(m.split=='train').values; y=m.species_id.values; corr={}
    for k in [1,3,10,20]:
        for seed in (range(3) if k<20 else [0]):
            for o1 in ['flower','leaf']:
                pool=m[tr&(m.organ==o1).values]
                idx=pool.groupby('species_id').sample(n=k,random_state=seed).index.values if k<20 else pool.index.values
                lams=(1e-2,) if k<10 else (1e-4,1e-3,1e-2)
                for o2 in ['flower','leaf']:
                    te=np.where(~tr&(m.organ==o2).values)[0]
                    P={}
                    for f,X in F.items():
                        if f in ('A','D'):
                            cl,P[f]=lr_fit(X[idx],y[idx],X[te],lams=lams,seed=seed,proba=True); pr=cl[P[f].argmax(1)]
                        else: pr=lr_fit(X[idx],y[idx],X[te],lams=lams,seed=seed)[0]
                        corr.setdefault((k,o1,o2,f),[]).append(pr==y[te])
                    corr.setdefault((k,o1,o2,'late'),[]).append(cl[(P['A']+P['D']).argmax(1)]==y[te])
    rows=[]
    for (k,o1,o2,f) in [x for x in corr if x[3]=='A']:
        g=lambda f: np.mean(corr[(k,o1,o2,f)],0)
        a,d,ad,an,dn,lt=g('A'),g('D'),g('AD'),g('Anull'),g('Dnull'),g('late'); best=a if a.mean()>=d.mean() else d
        dl=boot_diff(ad,best); dlate=boot_diff(lt,best); ca=np.array(corr[(k,o1,o2,'A')]); cd=np.array(corr[(k,o1,o2,'D')])
        rows.append(dict(size=size,k=k,train=o1,test=o2,within=o1==o2,accA=a.mean(),accD=d.mean(),accAD=ad.mean(),delta=dl[0],lo=dl[1],hi=dl[2],late=dlate[0],late_lo=dlate[1],late_hi=dlate[2],
            nullA=boot_diff(an,a)[0],nullD=boot_diff(dn,d)[0],weak=bool(d.mean()<a.mean()-0.15),only_one=float((ca^cd).mean()),oracle=float((ca|cd).mean())))
    an={n:{kk:float(v) for kk,v in anova_var(X['final'],m.species_id.values,m.organ.values).items()} for n,X in [(A,XA),(D,XD)]}
    res=dict(rows=[{k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in r.items()} for r in rows],anova=an,n_species=int(m.species_id.nunique()),n_img=len(m))
    json.dump(res,open(f"/v/r1_{size}.json","w")); vol.commit(); return res
@app.function(image=gpu_img,volumes={"/v":vol},gpu="T4",timeout=3600,memory=32768)
def d2():
    import numpy as np,pandas as pd,json,glob,pyarrow.parquet as pq
    from probe import lr_fit
    sel=pd.read_parquet("/v/pc_sel.parquet"); sh={}
    for f in glob.glob("/v/pcimg/*.parquet"):
        t=pq.read_table(f,columns=['idx','short']).to_pydict(); sh.update(zip(t['idx'],t['short']))
    f1,Z1=load('bioclip'); f2,Z2=load('bioclip2'); m=sel.loc[f1].reset_index(drop=True); m['short']=[sh[i] for i in f1]
    out=[]
    for o in ['flower','leaf','bark','fruit','habit']:
        tr=((m.split=='train')&(m.organ==o)).values; te=(m.split.isin(['test','test_gbif'])&(m.organ==o)).values
        if tr.sum()==0: continue
        c={}
        for n,Z in [('b1',Z1),('b2',Z2)]:
            c[n]=lr_fit(Z['final'][tr],m.species_id.values[tr],Z['final'][te])[0]==m.species_id.values[te]
        t=m[te].copy(); t['d']=c['b2'].astype(float)-c['b1'].astype(float); t['acc1']=c['b1']; t['acc2']=c['b2']
        t['rbin']=pd.qcut(t.short,4,labels=False,duplicates='drop'); t['organ_']=o; out.append(t)
    t=pd.concat(out)
    cell=t.groupby(['species_id','organ_','rbin','src']).d.mean().unstack('src').dropna()
    res={'n_matched_cells':len(cell),'per_organ_raw':t.groupby(['organ_','src']).agg(acc1=('acc1','mean'),acc2=('acc2','mean'),n=('d','size')).reset_index().to_dict('records')}
    if len(cell)>=30:
        dd=(cell.gbif-cell.plantnet).values; r=np.random.default_rng(0); bs=[dd[r.integers(0,len(dd),len(dd))].mean() for _ in range(2000)]
        res['matched_gap_gbif_minus_pn']=(float(dd.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
        res['matched_by_organ']=cell.assign(dd=dd).reset_index().groupby('organ_').dd.agg(['mean','size']).to_dict('index')
    json.dump(res,open("/v/d2.json","w"),default=str); vol.commit(); return res
@app.local_entrypoint()
def r1_main():
    import json
    res=list(r1.map(["B","L"])); json.dump(res,open("../results/r1.json","w"),default=str); print("R1 DONE")
