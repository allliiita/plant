import json,pandas as pd,numpy as np,sys
d=json.load(open(sys.argv[1])); m=pd.DataFrame.from_dict(d,orient='index').rename_axis('fid').reset_index()
m['tr']=np.where(m.split=='test','test','train')   # train+val pooled as probe-train
out={}
mat=m.groupby(['species_id','organ']).size().unstack(fill_value=0); mat.to_csv('/home/user/plantid/results/audit_species_organ_matrix.csv')
out['n_img'],out['n_species'],out['n_obs']=len(m),m.species_id.nunique(),m.obs_id.nunique()
out['organ_counts']=m.organ.value_counts().to_dict()
for o in ['bark','fruit','habit','flower','leaf']: out[f'sp_{o}>=5']=int((mat.get(o,0)>=5).sum())
out['sp_flower>=5&leaf>=5']=int(((mat.flower>=5)&(mat.leaf>=5)).sum())
# split-aware version (what probes actually need): >=5 train AND >=3 test per organ
c=m.groupby(['species_id','organ','tr']).size().unstack('tr',fill_value=0)
ok=(c.train>=5)&(c.test>=3); okc=ok[ok].reset_index()
out['split_aware_sp_per_organ(>=5tr,>=3te)']=okc.groupby('organ').species_id.nunique().to_dict()
fl=okc[okc.organ.isin(['flower','leaf'])].groupby('species_id').size(); out['split_aware_flower&leaf']=int((fl==2).sum())
c30=(c.train>=30)&(c.test>=6); f30=c30[c30].reset_index(); f30=f30[f30.organ.isin(['flower','leaf'])].groupby('species_id').size()
out['FL_set(>=30tr,>=6te both)']=int((f30==2).sum())
rare=okc[okc.organ.isin(['bark','fruit','habit'])].groupby('species_id').organ.apply(set)
out['sp_with_>=1rare_organ_and_flower_or_leaf']=int(sum(1 for s,v in rare.items() if s in set(okc[okc.organ.isin(['flower','leaf'])].species_id)))
# split protocol
sp=m.groupby('split').species_id.apply(set)
out['species_per_split']={k:len(v) for k,v in sp.items()}
out['species_overlap_train_test']=len(sp['train']&sp['test']); out['species_only_in_test']=len(sp['test']-sp['train'])
fr=m.groupby('species_id').split.apply(lambda s:(s=='test').mean()); out['test_frac_per_species_mean_sd']=(float(fr.mean()),float(fr.std()))
# observation leakage
os_=m.groupby('obs_id').split.nunique(); out['obs_spanning_>1_split']=int((os_>1).sum()); out['obs_multi_img']=int((m.obs_id.value_counts()>1).sum())
te=m[m.split=='test']; tr_obs=set(m[m.split!='test'].obs_id)
out['test_imgs_whose_obs_in_train']=int(te.obs_id.isin(tr_obs).sum()); out['test_imgs_total']=len(te)
out['frac_test_obs_leak']=out['test_imgs_whose_obs_in_train']/len(te)
# author leakage (same photographer in train and test, same species)
ta=set(zip(m[m.split!='test'].species_id,m[m.split!='test'].author)); out['frac_test_imgs_same_author_species_in_train']=float(np.mean([(s,a) in ta for s,a in zip(te.species_id,te.author)]))
# leakage restricted to our F/L subset organs
for o in ['flower','leaf']:
    t=te[te.organ==o]; out[f'obs_leak_frac_test_{o}']=float(t.obs_id.isin(tr_obs).mean())
m.to_parquet('/home/user/plantid/data/meta_full.parquet')
print(json.dumps(out,indent=1,default=str))
json.dump(out,open('/home/user/plantid/results/audit_meta.json','w'),indent=1,default=str)
