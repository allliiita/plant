import json,urllib.request,urllib.parse,time,sys
T=["citizen science enable automated plant trait predictions from photographs","citizen science plant observations encode global trait patterns",
"PlantTraits","plant traits iNaturalist photographs","trait predictions from photographs","sPlotOpen","TRY plant trait database",
"predicting plant traits from images","global trait maps from citizen science","plant trait regression vision transformer"]
def inv(ix):
    if not ix: return ""
    pos=sorted((p,w) for w,ps in ix.items() for p in ps); return " ".join(w for _,w in pos)
out={}
for q in T:
    for flt in ["title.search:"+q, None]:
        params={"per_page":10,"mailto":"research-client@example.org","filter":"from_publication_date:2018-01-01"+(","+flt if flt else "")}
        if not flt: params["search"]=q
        u="https://api.openalex.org/works?"+urllib.parse.urlencode(params)
        try: j=json.load(urllib.request.urlopen(u,timeout=60))
        except Exception: time.sleep(5); continue
        for w in j["results"]:
            t=(w["title"] or "").lower()
            if not any(k in t for k in ["trait","plantrait","planttrait","splot","phenotyp"]): continue
            out.setdefault(w["id"],{"q":q,"title":w["title"],"year":w["publication_year"],"cites":w["cited_by_count"],"doi":w.get("doi"),
               "venue":((w.get("primary_location") or {}).get("source") or {}).get("display_name"),"abstract":inv(w.get("abstract_inverted_index"))[:1400]})
        time.sleep(0.5)
# arXiv
for q in ["plant traits citizen science images","PlantTraits2024","plant trait prediction images"]:
    u="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":"all:\""+q+"\"","max_results":15})
    try:
        x=urllib.request.urlopen(u,timeout=60).read().decode()
        import re
        for e in re.findall(r"<entry>(.*?)</entry>",x,re.S):
            ti=re.sub(r"\s+"," ",re.search(r"<title>(.*?)</title>",e,re.S).group(1)); ab=re.sub(r"\s+"," ",re.search(r"<summary>(.*?)</summary>",e,re.S).group(1))
            yr=re.search(r"<published>(\d{4})",e).group(1)
            out.setdefault("arxiv:"+ti,{"q":q,"title":ti,"year":int(yr),"cites":-1,"venue":"arXiv","abstract":ab[:1400]})
    except Exception as ex: print("arxiv fail",ex)
    time.sleep(3)
json.dump(out,open(sys.argv[1],"w"),indent=0); print(len(out))
