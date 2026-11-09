import modal,json
app=modal.App("plantid-organ-diag")
img=(modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1","torchvision==0.20.1","timm==1.0.20","huggingface_hub==0.34.4","numpy==1.26.4","pandas==2.2.3","pyarrow==17.0.0","pillow==10.4.0"))
vol=modal.Volume.from_name("plantid-vol2")
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
    sel=pd.read_parquet("/v/pc/sel.parquet"); B={}
    for f in sorted(glob.glob("/v/pc/s2/shard_*.parquet")):
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
    cm=np.zeros((5,5),int)
    for t_,p_ in zip(y[te].numpy(),pred[te].numpy()): cm[t_,p_]+=1
    rep["confusion_rows_true_cols_pred"]={ORG[i]:cm[i].tolist() for i in range(5)}; rep["acc"]=float((pred[te]==y[te]).float().mean())
    torch.save({"mu":mu,"sd":sd,"W":W.detach().cpu(),"b":b.detach().cpu(),"classes":ORG,"backbone":"vit_base_patch16_224.augreg_in1k"},"/v/organ_clf.pt")
    tsel=sel[te.numpy()].reset_index(drop=True); tP=P[te]
    chk=pd.concat([g.sample(12,random_state=3) for _,g in tsel.groupby("organ")]).sample(frac=1,random_state=4).reset_index()
    os.makedirs("/v/pc/sheets",exist_ok=True)
    for s_ in range(0,60,30):
        sheet=Image.new("RGB",(6*200,5*215),"white"); dr=ImageDraw.Draw(sheet)
        for j,(_,r) in enumerate(chk.iloc[s_:s_+30].iterrows()):
            im=Image.open(io.BytesIO(B[r.image_name])).convert("RGB"); im.thumbnail((195,195)); x,yy=(j%6)*200,(j//6)*215
            sheet.paste(im,(x+2,yy+18)); dr.rectangle([x,yy,x+40,yy+16],fill="black"); dr.text((x+3,yy+2),str(s_+j),fill="yellow")
        sheet.save(f"/v/pc/sheets/pc_sheet_{s_//30}.jpg",quality=80)
    chk["pred"]=[ORG[int(tP[i].argmax())] for i in chk["index"]]; chk["p_leaf"]=[float(tP[i][0]) for i in chk["index"]]
    chk[["image_name","organ","pred","p_leaf"]].to_csv("/v/pc/sheets/pc_key.csv",index_label="tile")
    vol.commit(); return rep
@app.local_entrypoint()
def main():
    r=run.remote(); json.dump(r,open("/home/user/plantid/traits/organ_diag.json","w"),indent=1); print("ORG",json.dumps(r))
