import json,sys,re
sys.path.insert(0,'/home/user/plantid/src'); from ds import ask
P=json.load(open(sys.argv[1])); areas=sorted({v['area'] for v in P.values()}); G=[]
SYS="You are a senior plant-ML researcher. Extract research gaps strictly grounded in the given abstracts. Output valid JSON only."
for a in areas:
    ps=[v for v in P.values() if v['area']==a]
    txt="\n\n".join(f"[{i}] {p['title']} ({p['year']}, {p['venue']}, cites={p['cites']})\n{p['abstract'][:900]}" for i,p in enumerate(ps))
    prompt=f"""Area: {a}. Below are recent (2024+) paper abstracts.
List up to 6 concrete, unsolved research gaps that these papers expose (stated limitations, failure cases, missing evaluations, untested assumptions).
For each: {{"gap": one sentence, "evidence": [paper indices], "why_matters": one sentence, "public_data": named public dataset(s) that could test it or "unknown", "metric": how improvement would be measured, "ml_angle": which ML technique family could address it}}.
Only gaps supported by the abstracts. Return a JSON list.\n\n{txt}"""
    r=ask(prompt,system=SYS,max_tokens=2500)
    m=re.search(r'\[.*\]',r,re.S)
    try:
        L=json.loads(m.group(0))
        for g in L: g['area']=a; g['evidence_titles']=[ps[i]['title'] for i in g.get('evidence',[]) if isinstance(i,int) and i<len(ps)]
        G+=L; print(a,len(L),flush=True)
    except Exception as e: print(a,"PARSE_FAIL",flush=True)
json.dump(G,open(sys.argv[2],"w"),indent=1); print("total",len(G))
