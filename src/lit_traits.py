import json,urllib.request,urllib.parse,time,sys
QS=["plant trait prediction citizen science photographs deep learning","PlantTraits2024","iNaturalist TRY trait prediction convolutional",
"global plant trait maps citizen science","leaf area specific leaf area prediction from images","plant height prediction images deep learning",
"leaf nitrogen prediction RGB images","intraspecific trait variation images","species mean trait labels noisy labels regression",
"plant trait prediction uncertainty out-of-distribution","multimodal plant trait prediction climate image","plant functional traits remote sensing citizen science fusion"]
def inv(ix):
    if not ix: return ""
    pos=sorted((p,w) for w,ps in ix.items() for p in ps); return " ".join(w for _,w in pos)
out={}
for q in QS:
    u="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":q,"filter":"from_publication_date:2019-01-01,has_abstract:true","per_page":12,"sort":"relevance_score:desc","mailto":"research-client@example.org"})
    for t in range(4):
        try: j=json.load(urllib.request.urlopen(u,timeout=60)); break
        except Exception: time.sleep(3*2**t); j={"results":[]}
    for w in j["results"]:
        out.setdefault(w["id"],{"q":q,"title":w["title"],"year":w["publication_year"],"cites":w["cited_by_count"],"doi":w.get("doi"),
           "venue":((w.get("primary_location") or {}).get("source") or {}).get("display_name"),"abstract":inv(w.get("abstract_inverted_index"))[:1400]})
    time.sleep(0.5)
json.dump(out,open(sys.argv[1],"w"),indent=0); print(len(out))
