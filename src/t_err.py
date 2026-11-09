from common import *
@app.function(image=cpu_img,volumes={"/v":vol},timeout=600)
def t():
    import pandas as pd,urllib.request,collections,time
    d=pd.read_parquet("/v/pc_sel.parquet").dl.sample(40,random_state=1).tolist(); c=collections.Counter()
    for u in d:
        try: urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'research-client'}),timeout=30).read(); c['ok']+=1
        except Exception as e: c[str(e)[:60]]+=1
        time.sleep(0.5)
    o=d[0].replace('/image/m/','/image/o/')
    try: urllib.request.urlopen(o,timeout=30).read(); c['orig_ok']+=1
    except Exception as e: c['orig:'+str(e)[:50]]+=1
    return dict(c)
@app.local_entrypoint()
def main(): print(t.remote())
