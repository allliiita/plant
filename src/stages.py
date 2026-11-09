from common import *
@app.function(image=gpu_img,volumes={"/v":vol},gpu="T4",timeout=3600,memory=32768)
def analyze(name):
    import numpy as np,pandas as pd,json
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score
    from probe import lr_fit,boot_ci,anova_var
    Z=np.load(f"/v/feats_{name}.npz"); sel=pd.read_parquet("/v/selection.parquet").set_index('file_name')
    keys=sorted({k.split('__')[1] for k in Z.files if k.startswith('orig__') and not k.endswith('__fn')})
    meta=lambda c: sel.loc[Z[f"{c}__fn"]].reset_index()
    R={'model':name,'keys':keys,'species':{},'organ':{},'xfer':{},'anova':{},'cmatch':{},'cond':{},'fewshot':{},'erase':{},'zs':{}}
    m=meta('orig'); F=m.inF.values; O=m.inO.values
    def X(c,k): return Z[f"{c}__{k}"].astype(np.float32)
    for k in keys:
        x=X('orig',k); xf,mf=x[F],m[F].reset_index(drop=True); tr=(mf.split=='train').values; te=~tr
        pred,lam=lr_fit(xf[tr],mf.species[tr].values,xf[te]); cor=(pred==mf.species[te].values)
        R['species'][k]={'acc':boot_ci(cor),'by_organ':{o:float(cor[(mf.organ[te]==o).values].mean()) for o in ['flower','leaf']},'lam':lam,'correct':cor.astype(int).tolist() if k=='final' else None}
        # organ: 5-way, species-disjoint
        xo,mo=x[O],m[O].reset_index(drop=True); ba=[];allc=np.zeros(len(mo),bool)
        for a,b in GroupKFold(5).split(xo,groups=mo.species):
            p,_=lr_fit(xo[a],mo.organ.values[a],xo[b],lams=(1e-3,)); ba.append(balanced_accuracy_score(mo.organ.values[b],p)); allc[b]=p==mo.organ.values[b]
        R['organ'][k]={'bacc':float(np.mean(ba)),'bacc_sd':float(np.std(ba)),'acc':boot_ci(allc)}
        # cross-organ species transfer
        d={}
        for o1 in ['flower','leaf']:
            for o2 in ['flower','leaf']:
                a=tr&(mf.organ==o1).values; b=te&(mf.organ==o2).values
                p,_=lr_fit(xf[a],mf.species.values[a],xf[b],lams=(lam,)); d[f"{o1}->{o2}"]=float((p==mf.species.values[b]).mean())
        R['xfer'][k]=d
        R['anova'][k]={kk:float(v) for kk,v in anova_var(xf,mf.species.values,mf.organ.values).items()}
        # centroid matching: flower centroid of species s -> nearest leaf centroid (cosine, after global centering)
        xc=xf-xf.mean(0); sp=sorted(mf.species.unique())
        cf=np.stack([xc[((mf.species==s)&(mf.organ=='flower')).values].mean(0) for s in sp]); cl=np.stack([xc[((mf.species==s)&(mf.organ=='leaf')).values].mean(0) for s in sp])
        cf-=cf.mean(0); cl-=cl.mean(0); cf/=np.linalg.norm(cf,axis=1,keepdims=True); cl/=np.linalg.norm(cl,axis=1,keepdims=True)
        S=cf@cl.T; R['cmatch'][k]={'top1':float((S.argmax(1)==np.arange(len(sp))).mean()),'chance':1/len(sp)}
    # S4: degraded inputs, final + best layers only
    for c in ['gray','lowres']:
        mc=meta(c); tr=(mc.split=='train').values
        for k in keys:
            x=X(c,k); p,_=lr_fit(x[tr],mc.species.values[tr],x[~tr]); cor=p==mc.species.values[~tr]
            R['cond'][f"{c}|{k}"]={'acc':float(cor.mean()),'by_organ':{o:float(cor[(mc.organ[~tr]==o).values].mean()) for o in ['flower','leaf']}}
    # few-shot (final)
    x=X('orig','final'); xf,mf=x[F],m[F].reset_index(drop=True); tr=(mf.split=='train').values
    for K in [1,3,10,30]:
        accs=[]
        for s in range(3):
            idx=mf[tr].groupby('species').sample(n=K,replace=False,random_state=s).index.values if K<30 else np.where(tr)[0]
            p,_=lr_fit(xf[idx],mf.species.values[idx],xf[~tr],lams=(1e-2,) if K<10 else (1e-3,)); accs.append(float((p==mf.species.values[~tr]).mean()))
        R['fewshot'][K]=accs
    # erasure: LEACE-style removal of flower/leaf direction, then species probe; and species-centroid subspace removal then organ probe
    def leace(x,z):
        mu=x.mean(0); xc=x-mu; C=np.cov(xc.T)+1e-3*np.eye(x.shape[1]); ev,V=np.linalg.eigh(C); W=V@np.diag(ev**-.5)@V.T; Wi=V@np.diag(ev**.5)@V.T
        u=W@(xc.T@(z-z.mean())/len(z)); u/=np.linalg.norm(u); return x-(xc@W@np.outer(u,u))@Wi
    for k in ['final']:
        x=X('orig',k); xf,mf=x[F],m[F].reset_index(drop=True); tr=(mf.split=='train').values
        xe=leace(xf.astype(np.float64),(mf.organ=='flower').values.astype(float)).astype(np.float32)
        p,_=lr_fit(xe[tr],mf.species.values[tr],xe[~tr]); po,_=lr_fit(xe[tr],mf.organ.values[tr],xe[~tr],lams=(1e-3,))
        R['erase']['organ_erased_species_acc']=float((p==mf.species.values[~tr]).mean()); R['erase']['organ_erased_organ_acc']=float((po==mf.organ.values[~tr]).mean())
        # remove top-r species-centroid PCs (species subspace), then organ probe (species-disjoint)
        sp_c=pd.DataFrame(xf[tr]).groupby(mf.species.values[tr]).mean().values; sp_c-=sp_c.mean(0)
        U=np.linalg.svd(sp_c,full_matrices=False)[2]
        for r in [8,32,128]:
            xo,mo=x[O],m[O].reset_index(drop=True); xr=xo-(xo-xo.mean(0))@U[:r].T@U[:r]; ba=[]
            for a,b in GroupKFold(5).split(xr,groups=mo.species):
                p,_=lr_fit(xr[a],mo.organ.values[a],xr[b],lams=(1e-3,)); ba.append(balanced_accuracy_score(mo.organ.values[b],p))
            R['erase'][f'species_sub{r}_removed_organ_bacc']=float(np.mean(ba))
    if 'txt_species' in Z.files:
        x=X('orig','final'); x/=np.linalg.norm(x,axis=1,keepdims=True); mf=m[F].reset_index(drop=True); te=(mf.split=='test').values
        T=Z['txt_species']; T/=np.linalg.norm(T,axis=1,keepdims=True); names=Z['txt_species_names']
        Fs=set(mf.species); keep=np.array([n in Fs for n in names]); T2,n2=T[keep],names[keep]
        p=n2[(x[F][te]@T2.T).argmax(1)]; cor=p==mf.species.values[te]
        R['zs']['species']={'acc':float(cor.mean()),'by_organ':{o:float(cor[(mf.organ[te]==o).values].mean()) for o in ['flower','leaf']}}
        To=Z['txt_organ']; To/=np.linalg.norm(To,axis=1,keepdims=True); mo=m[O]; p=Z['txt_organ_names'][(x[O]@To.T).argmax(1)]
        R['zs']['organ_bacc']=float(balanced_accuracy_score(mo.organ.values,p))
    json.dump(R,open(f"/v/res_{name}.json","w"),default=lambda o: o.tolist() if hasattr(o,'tolist') else float(o)); vol.commit()
    return name
@app.local_entrypoint()
def main(models:str="bioclip,bioclip2,clip_b16,clip_l14,dinov3_b,dinov3_l,dinov2_b"):
    for r in analyze.map(models.split(","),return_exceptions=True): print(r)
