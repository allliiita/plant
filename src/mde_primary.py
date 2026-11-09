import pandas as pd,numpy as np,json
S='/tmp/claude-0/-home-user/fe3dc421-8c70-5172-bf54-6224b620a519/scratchpad/mig/ptn/'
T=['Height','LeafArea','SLA','Leaf_N']
d=pd.concat([pd.read_csv(S+f'{s}.csv') for s in ['train','val']]); d['pid']=d.Image.str.extract(r'photo_(\d+)')[0]
d=d[d.pid.notna()].drop_duplicates(['Species','pid']); multi=d.groupby('pid').Species.nunique(); d=d[~d.pid.isin(set(multi[multi>1].index))]
n=d.groupby('Species').size(); pool=n[n>=30].index
g=pd.read_parquet(S+'gbif_match_full.parquet'); g=g[g.Species.isin(pool)]
g=g[(g['rank']=='SPECIES')&(g.confidence>=90)&g.genus.notna()&g.family.notna()&g.order.notna()]; g=g[~g.speciesKey.duplicated(keep=False)]
sp=np.array(sorted(g.Species)); tax=g.set_index('Species').loc[sp]
d0=pd.concat([pd.read_csv(S+f'{s}.csv') for s in ['train','val']]); lab=d0.groupby('Species')[[f'{t}_median' for t in T]].first(); print('label conflicts (non-null distinct>1):',int((d0.groupby('Species')[[f'{t}_median' for t in T]].nunique()>1).any(axis=1).sum()),'| species with all-NaN labels:',int(lab.isna().any(axis=1).sum())); Y=np.log(lab.loc[sp].values)
u=sp.copy(); perm=u[np.random.default_rng(0).permutation(len(u))]; fo=np.array([{x:i%5 for i,x in enumerate(perm)}[x] for x in sp])
pred=np.full(Y.shape,np.nan); used=np.array(['none']*len(sp),dtype=object)
for k in range(5):
    te=fo==k; tr=~te
    for lv in ['genus','family','order']:
        G=pd.DataFrame(Y[tr],index=tax[lv].values[tr]).groupby(level=0); mean=G.mean()[G.size()>=2]
        v=mean.reindex(tax[lv].values).values; m=te&np.isnan(pred[:,0])&~np.isnan(v[:,0]); pred[m]=v[m]; used[m]=lv
gm=used=='genus'; print('genus stratum n',gm.sum(),'nan in pred',int(np.isnan(pred[gm]).sum()))
rng=np.random.default_rng(0); out={}
for j,t in enumerate(T):
    e=Y[gm,j]-pred[gm,j]; vy=Y[gm,j].var(); r2i=1-(e**2).mean()/vy; nS=int(gm.sum()); res={}
    for rho in (0.3,0.5,0.7):
        found=None
        for dR in np.arange(0.005,0.3,0.005):
            s_img=np.sqrt(max((1-(r2i+dR))*vy,1e-9)); hits=0
            for _ in range(200):
                ei=rho*s_img*(e/e.std())+np.sqrt(1-rho**2)*s_img*rng.standard_normal(nS); dd=(e**2-ei**2)/vy; hits+=(dd.mean()/(dd.std(ddof=1)/np.sqrt(nS)))>2.50
            if hits/200>=0.8: found=round(float(dR),3); break
        res[f'rho{rho}']=found
    out[t]={'n':nS,'r2_genus_imp':round(float(r2i),3),'MDE':res}; print(t,out[t],flush=True)
json.dump(out,open('/home/user/plantid/traits/mde_primary.json','w'),indent=1)
