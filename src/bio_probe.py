from common import *
@app.function(image=cpu_img,timeout=600)
def probe():
    import urllib.request,urllib.parse,json
    out={}
    for u in ["https://rdf.library.vanderbilt.edu/sparql?query="+urllib.parse.quote("ASK{}"),"https://sparql.vanderbilt.edu/sparql?query="+urllib.parse.quote("ASK{}"),"https://bioimages.vanderbilt.edu/"]:
        try: r=urllib.request.urlopen(u,timeout=30); out[u[:45]]=(r.status,r.read(200).decode('utf8','ignore'))
        except Exception as e: out[u[:45]]=str(e)[:150]
    try:
        t=json.load(urllib.request.urlopen("https://api.github.com/repos/baskaufs/Bioimages/git/trees/master?recursive=1",timeout=60))
        fs=[(x['path'],x.get('size')) for x in t['tree'] if x['type']=='blob' and x['path'].endswith(('.csv','.xml','.ttl','.rdf','.txt','.md')) and x.get('size',0)>50000]
        out['gh_big_files']=sorted(fs,key=lambda x:-x[1])[:25]; out['gh_n']=len(t['tree'])
    except Exception as e: out['gh']=str(e)[:200]
    return out
@app.local_entrypoint()
def main():
    import json; print(json.dumps(probe.remote(),indent=0))
