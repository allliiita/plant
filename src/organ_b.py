import modal,json
app=modal.App("plantid-organ")
img=(modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1","torchvision==0.20.1","timm==1.0.20","huggingface_hub==0.34.4","numpy==1.26.4","pandas==2.2.3","pyarrow==17.0.0","pillow==10.4.0"))
vol=modal.Volume.from_name("plantid-vol")
ORG=["leaf","flower","fruit","bark","habit"]
@app.function(image=img,gpu="A10G",timeout=3600,memory=32768,volumes={"/v":vol})
def run():
    import torch,timm,numpy as np,pandas as pd,glob,io,urllib.request,os
    import pyarrow.parquet as pq
    from PIL import Image,ImageDraw
    from concurrent.futures import ThreadPoolExecutor
    torch.manual_seed(0); dev="cuda"
    net=timm.create_model("vit_base_patch16_224.augreg_in1k",pretrained=True,num_classes=0).to(dev).eval()
    cfg=timm.data.resolve_data_config({},model=net); pre=timm.data.create_transform(**cfg)
    def feats(ims):
        out=[]
        with torch.no_grad():
            for i in range(0,len(ims),64): out.append(net(torch.stack([pre(x) for x in ims[i:i+64]]).to(dev)).float().cpu())
        return torch.cat(out)
    # PlantCLEF organ labels (author-disjoint train/test from frozen split)
    sel=pd.read_parquet("/v/runs/full/s1/sel.parquet"); B={}
    for f in sorted(glob.glob("/v/runs/full/s2/shard_*.parquet")):
        t=pq.read_table(f,columns=["image_name","bytes"]).to_pydict(); B.update({n:b for n,b in zip(t["image_name"],t["bytes"]) if b})
    sel=sel[sel.image_name.isin(B)]; sel=pd.concat([g.sample(min(len(g),3000),random_state=0) for _,g in sel.groupby(["organ","split"])])
    X=feats([Image.open(io.BytesIO(B[n])).convert("RGB") for n in sel.image_name]); y=torch.tensor([ORG.index(o) for o in sel.organ])
    tr=torch.tensor((sel.split=="train").values); mu,sd=X[tr].mean(0),X[tr].std(0)+1e-6; Xs=((X-mu)/sd).to(dev)
    W=torch.zeros(X.shape[1],5,device=dev,requires_grad=True); b=torch.zeros(5,device=dev,requires_grad=True)
    opt=torch.optim.LBFGS([W,b],max_iter=300,line_search_fn="strong_wolfe"); yt=y.to(dev)
    def cl():
        opt.zero_grad(); l=torch.nn.functional.cross_entropy(Xs[tr.to(dev)]@W+b,yt[tr.to(dev)])+1e-3*(W**2).sum(); l.backward(); return l
    opt.step(cl)
    P=torch.softmax(Xs@W+b,1).detach().cpu(); te=~tr; pred=P.argmax(1)
    rep={"plantclef_test_n":int(te.sum())}
    for th in [0.0,0.5,0.7,0.8]:
        r={}
        for k,o in enumerate(ORG):
            sel_k=te&(pred==k)&(P.max(1).values>=th); r[o]={"precision":round(float((y[sel_k]==k).float().mean()),3) if sel_k.sum() else None,"recall":round(float((sel_k&(y==k)).sum()/((te&(y==k)).sum())),3)}
        rep[f"th{th}"]=r
    # PlantTraitNet family-stratified sample
    pool=pd.read_parquet("/v/ptn/pool_species.parquet")[["Species","family"]]
    d=pd.concat([pd.read_csv(f"/v/ptn/{s}.csv",usecols=["Species","Image"]) for s in ["train","val"]]).merge(pool,on="Species")
    d["pid"]=d.Image.str.extract(r"photo_(\d+)")[0]
    smp=pd.concat([g.sample(min(len(g),20),random_state=0) for _,g in d.groupby("family")]).reset_index(drop=True)
    def dl(pid):
        for ext in ["jpg","jpeg","png"]:
            try: return Image.open(io.BytesIO(urllib.request.urlopen(f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{ext}",timeout=30).read())).convert("RGB")
            except Exception: pass
        return None
    with ThreadPoolExecutor(8) as ex: ims=list(ex.map(dl,smp.pid))
    ok=[i for i,x in enumerate(ims) if x is not None]; smp=smp.iloc[ok].reset_index(drop=True); ims=[ims[i] for i in ok]
    Q=torch.softmax(((feats(ims)-mu)/sd).to(dev)@W+b,1).detach().cpu().numpy()
    smp["pred"]=[ORG[i] for i in Q.argmax(1)]; smp["conf"]=Q.max(1)
    fam_n=d.groupby("family").size(); fr=smp.groupby("family").pred.value_counts(normalize=True).unstack(fill_value=0)
    w=fam_n.reindex(fr.index)/fam_n.reindex(fr.index).sum()
    rep["ptn_sample_n"]=len(smp); rep["ptn_pred_unweighted"]=smp.pred.value_counts(normalize=True).round(3).to_dict()
    rep["ptn_pred_family_weighted"]=(fr.mul(w,axis=0).sum()).round(3).to_dict()
    # per-species expected usable photos (conf>=0.5)
    pf=smp[smp.conf>=0.5].groupby("family").pred.value_counts().unstack(fill_value=0).div(smp.groupby("family").size(),axis=0).fillna(0)
    n_sp=d.groupby(["Species","family"]).size().reset_index(name="n")
    for o in ["leaf","habit"]:
        n_sp[o]=n_sp.n*n_sp.family.map(pf[o] if o in pf else 0).fillna(0)
        rep[f"species_expected_{o}_ge10"]=int((n_sp[o]>=10).sum()); rep[f"species_expected_{o}_ge5"]=int((n_sp[o]>=5).sum())
    # contact sheets for manual check: 300 stratified (by predicted class) images, labels NOT drawn
    chk=pd.concat([g.sample(min(len(g),60),random_state=1) for _,g in smp.groupby("pred")]).sample(frac=1,random_state=2).head(300).reset_index()
    os.makedirs("/v/ptn/sheets",exist_ok=True)
    for s in range(0,len(chk),30):
        sheet=Image.new("RGB",(6*200,5*215),"white"); dr=ImageDraw.Draw(sheet)
        for j,(_,r) in enumerate(chk.iloc[s:s+30].iterrows()):
            im=ims[r["index"]].copy(); im.thumbnail((195,195)); x,yy=(j%6)*200,(j//6)*215
            sheet.paste(im,(x+2,yy+18)); dr.rectangle([x,yy,x+40,yy+16],fill="black"); dr.text((x+3,yy+2),str(s+j),fill="yellow")
        sheet.save(f"/v/ptn/sheets/sheet_{s//30:02d}.jpg",quality=80)
    chk[["Species","family","pid","pred","conf"]].to_csv("/v/ptn/sheets/key.csv",index_label="tile"); smp.drop(columns=[]).to_csv("/v/ptn/organ_sample.csv",index=False)
    vol.commit(); return rep
@app.local_entrypoint()
def main():
    r=run.remote(); json.dump(r,open("/home/user/plantid/traits/organ_b.json","w"),indent=1); print("ORG",json.dumps(r))
