from common import *
MODELS={"bioclip":("oc","hf-hub:imageomics/bioclip",None),"bioclip2":("oc","hf-hub:imageomics/bioclip-2",None),
 "clip_b16":("oc","ViT-B-16","openai"),"clip_l14":("oc","ViT-L-14","openai"),
 "dinov3_b":("timm","vit_base_patch16_dinov3.lvd1689m",None),"dinov3_l":("timm","vit_large_patch16_dinov3.lvd1689m",None),
 "dinov2_b":("timm","vit_base_patch14_dinov2.lvd142m",None)}
ORG_PROMPTS={"flower":"a photo of the flower of a plant.","leaf":"a photo of the leaf of a plant.","fruit":"a photo of the fruit of a plant.","bark":"a photo of the bark of a tree.","habit":"a photo of a whole plant in its habitat."}
@app.function(image=gpu_img,volumes={"/v":vol},gpu="A10G",timeout=3600,memory=32768)
def run(name,CONDS=("orig",)):
    import torch,numpy as np,pandas as pd,io,glob,pyarrow.parquet as pq
    from PIL import Image
    kind,mid,pt=MODELS[name]; dev="cuda"
    if kind=="oc":
        import open_clip
        model,_,pre=open_clip.create_model_and_transforms(mid,pretrained=pt) if pt else open_clip.create_model_and_transforms(mid)
        blocks=model.visual.transformer.resblocks; npre=1; tok=open_clip.get_tokenizer(mid)
    else:
        import timm
        model=timm.create_model(mid,pretrained=True,num_classes=0)
        cfg=timm.data.resolve_data_config({},model=model); cfg['input_size']=(3,224,224); pre=timm.data.create_transform(**cfg)
        blocks=model.blocks; npre=model.num_prefix_tokens
    fp=torch.float32 if kind=="timm" else torch.float16
    model=model.to(dev).eval().to(fp)
    L=len(blocks); idx=sorted(set([L//4-1,L//2-1,3*L//4-1,L-1])); acts={}
    for i in idx:
        blocks[i].register_forward_hook(lambda m,inp,out,i=i: acts.__setitem__(i,out))
    sel=pd.read_parquet("/v/pc_sel.parquet"); B={}
    for f in glob.glob("/v/pcimg/*.parquet"):
        t=pq.read_table(f,columns=['idx','bytes']).to_pydict(); B.update(zip(t['idx'],t['bytes']))
    sel['file_name']=sel.index; sel=sel[sel.file_name.isin(B)].reset_index(drop=True)
    def cond_img(im,c):
        if c=="gray": return im.convert("L").convert("RGB")
        if c=="lowres": w,h=im.size; s=56/min(w,h); return im.resize((max(1,int(w*s)),max(1,int(h*s))),Image.BILINEAR).resize((w,h),Image.BILINEAR)
        return im
    class DS(torch.utils.data.Dataset):
        def __init__(s,fns,c): s.fns,s.c=fns,c
        def __len__(s): return len(s.fns)
        def __getitem__(s,k): return pre(cond_img(Image.open(io.BytesIO(B[s.fns[k]])).convert("RGB"),s.c))
    out={}
    for c in CONDS:
        fns=sel.file_name.tolist() if c=="orig" else sel[sel.inFL].file_name.tolist()
        dl=torch.utils.data.DataLoader(DS(fns,c),batch_size=128,num_workers=8)
        feats={}
        with torch.no_grad():
            for x in dl:
                x=x.to(dev).to(fp)
                if kind=="oc": fin=model.encode_image(x)
                else: fin=model.forward_head(model.forward_features(x),pre_logits=True)
                rec={"final":fin}
                for i in idx:
                    a=acts[i]
                    if a.shape[0]!=x.shape[0]: a=a.permute(1,0,2)
                    rec[f"L{i+1}_cls"]=a[:,0]; rec[f"L{i+1}_pm"]=a[:,npre:].mean(1)
                for k,v in rec.items(): feats.setdefault(k,[]).append(v.float().cpu().numpy().astype(np.float16))
        for k,v in feats.items(): out[f"{c}__{k}"]=np.concatenate(v)
        out[f"{c}__fn"]=np.array(fns)
        print(name,c,len(fns),flush=True)
    if kind=="oc":
        with torch.no_grad():
            sp=sorted(sel.species.str.extract(r"^([A-Z][a-z]+ [a-z\-]+)")[0].dropna().unique()); 
            te=lambda P: torch.cat([model.encode_text(tok(P[i:i+256]).to(dev)).float() for i in range(0,len(P),256)]).cpu().numpy()
            out["txt_species"]=te([f"a photo of {s}." for s in sp]); out["txt_species_names"]=np.array(sp)
            out["txt_organ"]=te(list(ORG_PROMPTS.values())); out["txt_organ_names"]=np.array(list(ORG_PROMPTS))
            out["logit_scale"]=np.array(model.logit_scale.exp().item())
    np.savez(f"/v/pcfeats_{name}.npz",**out); vol.commit(); return name,len(sel),idx
@app.local_entrypoint()
def extract_main(models:str="bioclip,dinov3_b,bioclip2,dinov3_l"):
    ms=list(MODELS) if models=="all" else models.split(",")
    for r in run.map(ms,return_exceptions=True): print(r)
