from common import *
@app.function(image=cpu_img,volumes={"/v":vol},timeout=900,memory=16384)
def ov(bio_species):
    import pandas as pd,urllib.request
    m=pd.read_csv("/v/pc24_meta.csv",sep=';',usecols=['species_id','species','obs_id','organ','image_backup_url','url'],dtype=str)
    m['bin']=m.species.str.extract(r'^([A-Z][a-z]+ [a-z\-]+)')[0]
    c=pd.read_csv("/v/pc24_species_organ_obs.csv",dtype={'species_id':str}).set_index('species_id')
    full=c[(c.flower>=20)&(c.leaf>=20)&(c.bark>=20)&((c.fruit>=20)|(c.habit>=20))].index
    names_full=set(m[m.species_id.isin(full)].bin); names_all=set(m.bin)
    out={'bio_overlap_all_pc':len(set(bio_species)&names_all),'bio_overlap_full4_subset':len(set(bio_species)&names_full)}
    sz=[]
    for u in m.sample(6,random_state=0).url:
        for v in [u,u.replace('/image/o/','/image/m/')]:
            try: r=urllib.request.urlopen(urllib.request.Request(v,method='HEAD'),timeout=20); sz.append((v[-50:],r.headers.get('Content-Length')))
            except Exception as e: sz.append((v[-50:],str(e)[:40]))
    u=m.image_backup_url.iloc[0]
    try: r=urllib.request.urlopen(urllib.request.Request(u,method='HEAD'),timeout=20); sz.append(('backup',r.headers.get('Content-Length')))
    except Exception as e: sz.append(('backup',str(e)[:40]))
    out['sizes']=sz; return out
@app.local_entrypoint()
def main():
    import pandas as pd,json
    b=pd.read_parquet("../data/bio_images.parquet"); fl=b[b.part.isin(['inflorescence','leaf'])].groupby('species').part.nunique()
    r=ov.remote(list(fl[fl==2].index)); print(json.dumps(r))
