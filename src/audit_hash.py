from common import *
himg=modal.Image.debian_slim().pip_install("pyarrow","pandas","huggingface_hub","numpy","imagehash","pillow")
@app.function(image=himg,timeout=3600,memory=8192,cpu=4)
def hshard(fn):
    import pyarrow.parquet as pq, io, numpy as np, imagehash
    from PIL import Image
    from huggingface_hub import hf_hub_download
    col=pq.read_table(hf_hub_download(R,fn,repo_type="dataset"),columns=['image']).column('image').to_pylist()
    names,ph,th=[],[],[]
    for r in col:
        try:
            im=Image.open(io.BytesIO(r['bytes'])); im.draft('L',(128,128)); im=im.convert('L')
            ph.append(int(str(imagehash.phash(im)),16)); th.append(np.asarray(im.resize((16,16)),np.float32).ravel()); names.append(r['path'])
        except Exception: pass
    return names,np.array(ph,dtype=np.uint64),np.stack(th).astype(np.float16)
@app.local_entrypoint()
def main():
    import numpy as np
    from huggingface_hub import list_repo_files
    fs=[f for f in list_repo_files(R,repo_type="dataset") if f.endswith('.parquet')]
    N,P,T=[],[],[]
    for n,p,t in hshard.map(fs): N+=n; P.append(p); T.append(t)
    np.savez("../data/hashes.npz",names=np.array(N),ph=np.concatenate(P),th=np.concatenate(T)); print("hashed",len(N))
