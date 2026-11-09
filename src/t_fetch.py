from pc_build import *
@app.local_entrypoint()
def t():
    c=chunks.remote(); print("nchunks",len(c),c[0][:2])
    print(fetch.remote(c[0][:20]))
