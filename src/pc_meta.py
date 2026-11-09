from common import *
@app.function(image=cpu_img,volumes={"/v":vol},timeout=3600,memory=32768,cpu=4)
def meta():
    import pandas as pd,numpy as np,urllib.request,json,os
    p="/v/pc24_meta.csv"
    if not os.path.exists(p): urllib.request.urlretrieve("https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.csv",p); vol.commit()
    m=pd.read_csv(p,sep=';',usecols=['image_name','organ','species_id','obs_id','author','species','dataset','learn_tag','image_backup_url'],dtype=str)
    o={'n_img':len(m),'n_species':m.species_id.nunique(),'n_obs':m.obs_id.nunique(),'organ':m.organ.value_counts().to_dict(),
       'dataset':m.dataset.value_counts().head(6).to_dict(),'learn_tag':m.learn_tag.value_counts().to_dict()}
    ipo=m.groupby('obs_id').size(); o['imgs_per_obs_mean']=float(ipo.mean()); o['obs_multi_img_frac']=float((ipo>1).mean())
    o['obs_spanning_splits']=int((m.groupby('obs_id').learn_tag.nunique()>1).sum())
    sp=m.groupby('learn_tag').species_id.apply(set); o['species_per_split']={k:len(v) for k,v in sp.items()}
    o['test_species_not_in_train']=len(sp.get('test',set())-sp.get('train',set()))
    # organ coverage counted in distinct observations
    c=m.groupby(['species_id','organ']).obs_id.nunique().unstack(fill_value=0)
    for k in ['flower','leaf','bark','fruit','habit']: c[k]=c.get(k,0)
    def n(th): 
        q=(c.flower>=th)&(c.leaf>=th)&(c.bark>=th)&((c.fruit>=th)|(c.habit>=th)); return int(q.sum())
    o['full4_species_by_min_obs']={th:n(th) for th in [5,10,20,30]}
    o['flower&leaf_species_by_min_obs']={th:int(((c.flower>=th)&(c.leaf>=th)).sum()) for th in [5,10,20,30]}
    for org in ['bark','fruit','habit']: o[f'{org}_species_by_min_obs']={th:int((c[org]>=th).sum()) for th in [5,10,20]}
    q=c[(c.flower>=20)&(c.leaf>=20)&(c.bark>=20)&((c.fruit>=20)|(c.habit>=20))]
    o['full4@20obs_median_obs_per_organ']=q[['flower','leaf','bark','fruit','habit']].median().to_dict()
    o['author_imgs_frac_top1pct']=float(m.author.value_counts().head(max(1,m.author.nunique()//100)).sum()/len(m))
    c.to_csv("/v/pc24_species_organ_obs.csv"); vol.commit(); return o
@app.local_entrypoint()
def main():
    import json; r=meta.remote(); json.dump(r,open("../results/audit_pc24.json","w"),indent=1,default=str); print(json.dumps(r,default=str))
