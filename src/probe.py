import numpy as np, torch
def lr_fit(Xtr,ytr,Xte,lams=(1e-4,1e-3,1e-2),iters=150,dev="cuda",seed=0,proba=False):
    Xtr=torch.tensor(Xtr,dtype=torch.float32,device=dev); Xte=torch.tensor(Xte,dtype=torch.float32,device=dev)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    cls,yt=np.unique(ytr,return_inverse=True); yt=torch.tensor(yt,device=dev); C=len(cls)
    def fit(X,y,lam):
        W=torch.zeros(X.shape[1],C,device=dev,requires_grad=True); b=torch.zeros(C,device=dev,requires_grad=True)
        opt=torch.optim.LBFGS([W,b],max_iter=iters,history_size=20,line_search_fn="strong_wolfe")
        def cl():
            opt.zero_grad(); l=torch.nn.functional.cross_entropy(X@W+b,y)+lam*(W**2).sum(); l.backward(); return l
        opt.step(cl); return W.detach(),b.detach()
    g=torch.Generator().manual_seed(seed); p=torch.randperm(len(yt),generator=g).to(dev); nv=max(1,len(p)//5)
    va,tr=p[:nv],p[nv:]; best=None
    if len(lams)>1:
        for lam in lams:
            W,b=fit(Xtr[tr],yt[tr],lam); a=((Xtr[va]@W+b).argmax(1)==yt[va]).float().mean().item()
            if best is None or a>best[0]: best=(a,lam)
    lam=best[1] if best else lams[0]
    W,b=fit(Xtr,yt,lam); L=Xte@W+b
    if proba: return cls,torch.softmax(L,1).cpu().numpy()
    return cls[L.argmax(1).cpu().numpy()], lam
def boot_ci(correct,n=1000,seed=0):
    r=np.random.default_rng(seed); c=np.asarray(correct,float); m=[c[r.integers(0,len(c),len(c))].mean() for _ in range(n)]
    return c.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
def boot_diff(c1,c2,n=1000,seed=0):
    r=np.random.default_rng(seed); c1,c2=np.asarray(c1,float),np.asarray(c2,float); d=[]
    for _ in range(n): i=r.integers(0,len(c1),len(c1)); d.append(c1[i].mean()-c2[i].mean())
    return c1.mean()-c2.mean(),np.percentile(d,2.5),np.percentile(d,97.5)
def anova_var(X,sp,org):
    """two-way decomposition of (standardized) feature variance into species, organ, interaction, residual"""
    X=(X-X.mean(0))/ (X.std(0)+1e-6); tot=(X**2).sum()
    import pandas as pd
    df=pd.DataFrame({'s':sp,'o':org})
    def proj(keys):
        g=df.groupby(keys).ngroup().values; M=np.zeros_like(X)
        for k in np.unique(g): M[g==k]=X[g==k].mean(0)
        return M
    Ms,Mo,Mso=proj(['s']),proj(['o']),proj(['s','o'])
    inter=Mso-Ms-Mo
    return {'species':(Ms**2).sum()/tot,'organ':(Mo**2).sum()/tot,'interaction':(inter**2).sum()/tot,'residual':((X-Mso)**2).sum()/tot}
