import json,urllib.request,urllib.parse,time,sys
Q={"fine-grained plant identification":["plant species identification deep learning","plantclef","plant organ classification images"],
"herbarium":["herbarium specimen deep learning","herbarium digitization machine learning trait extraction"],
"disease":["plant disease detection deep learning field images","plant disease few-shot domain shift"],
"phenology":["plant phenology deep learning images","flowering time detection citizen science images"],
"traits":["leaf trait prediction images deep learning","plant functional traits citizen science images prediction"],
"root/3D":["root phenotyping deep learning","plant 3D point cloud phenotyping deep learning"],
"remote sensing":["tree species classification remote sensing deep learning","hyperspectral plant species deep learning"],
"citizen science":["citizen science plant observations machine learning bias","iNaturalist plant images deep learning"],
"self-supervised bio":["self-supervised learning plant images","foundation model plant biology"],
"genomics":["plant genomics deep learning language model","plant single-cell transcriptomics deep learning"],
"pollination/ecology":["plant pollinator interaction deep learning images","vegetation plot species cover estimation images"],
}
def inv(ix):
    if not ix: return ""
    pos=sorted((p,w) for w,ps in ix.items() for p in ps); return " ".join(w for _,w in pos)
out={}
for area,qs in Q.items():
    for q in qs:
        u="https://api.openalex.org/works?"+urllib.parse.urlencode({"search":q,"filter":"from_publication_date:2024-01-01,has_abstract:true","per_page":15,"sort":"relevance_score:desc","mailto":"research-client@example.org"})
        for t in range(4):
            try: j=json.load(urllib.request.urlopen(u,timeout=60)); break
            except Exception as e: time.sleep(3*2**t); j={"results":[]}
        for w in j["results"]:
            out.setdefault(w["id"],{"area":area,"title":w["title"],"year":w["publication_year"],"cites":w["cited_by_count"],
               "venue":((w.get("primary_location") or {}).get("source") or {}).get("display_name"),"abstract":inv(w.get("abstract_inverted_index"))[:1500]})
        time.sleep(0.5)
json.dump(out,open(sys.argv[1],"w"),indent=0); print(len(out),{a:sum(1 for v in out.values() if v["area"]==a) for a in Q})
