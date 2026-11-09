import json,urllib.request,urllib.parse,time,sys,re
Q={"00":["species identity confound trait prediction images","trait prediction species recognition shortcut deep learning"],
"01":["phylogenetic cross-validation trait prediction images deep learning","taxonomically blocked validation plant trait prediction"],
"02":["intraspecific trait variation prediction from photographs","individual-level plant trait prediction images"],
"03":["self-supervised pretraining plant trait regression","contrastive learning masked autoencoder plant traits"],
"07":["species prior trait prediction images deep learning gap filling","trait prediction images versus taxonomic imputation"],
"10":["few-shot plant trait prediction new species images","meta-learning plant trait regression"],
"12":["flower colour shape traits citizen science photographs deep learning","floral trait extraction images machine learning"]}
def inv(ix):
    if not ix: return ""
    pos=sorted((p,w) for w,ps in ix.items() for p in ps); return " ".join(w for _,w in pos)
out={}
for k,qs in Q.items():
    res=[]
    for q in qs:
        u="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":q,"filter":"from_publication_date:2018-01-01,has_abstract:true","per_page":8,"mailto":"research-client@example.org"})
        try: j=json.load(urllib.request.urlopen(u,timeout=60))
        except Exception: time.sleep(5); j={"results":[]}
        res+=[{"title":w["title"],"year":w["publication_year"],"venue":((w.get("primary_location") or {}).get("source") or {}).get("display_name"),"abstract":inv(w.get("abstract_inverted_index"))[:600]} for w in j["results"]]
        u2="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":"all:"+q.replace(" "," AND all:"),"max_results":5})
        try:
            x=urllib.request.urlopen(u2,timeout=60).read().decode()
            for e in re.findall(r"<entry>(.*?)</entry>",x,re.S):
                res.append({"title":re.sub(r"\s+"," ",re.search(r"<title>(.*?)</title>",e,re.S).group(1)),"year":re.search(r"<published>(\d{4})",e).group(1),"venue":"arXiv","abstract":re.sub(r"\s+"," ",re.search(r"<summary>(.*?)</summary>",e,re.S).group(1))[:600]})
        except Exception: pass
        time.sleep(3)
    out[k]=res
json.dump(out,open(sys.argv[1],"w")); print(sum(len(v) for v in out.values()))
