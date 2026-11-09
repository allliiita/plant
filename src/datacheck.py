import modal
app=modal.App("plantid-datacheck")
@app.function(image=modal.Image.debian_slim(python_version="3.11"),timeout=900)
def check():
    import urllib.request,urllib.parse,json,time
    out={}
    def get(u,t=60): return urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"research-client"}),timeout=t)
    for q in ["global spectrum of plant form and function enhanced species-level trait dataset","AusTraits","FloralTraits"]:
        try:
            j=json.load(get("https://zenodo.org/api/records?"+urllib.parse.urlencode({"q":q,"size":4})))
            out["zenodo:"+q]=[(h["id"],h["metadata"]["title"][:90],h["metadata"].get("access_right"),[(f["key"],f["size"]) for f in h.get("files",[])][:6]) for h in j["hits"]["hits"]]
        except Exception as e: out["zenodo:"+q]=str(e)[:150]
    try:
        j=json.load(get("https://api.gbif.org/v1/occurrence/search?"+urllib.parse.urlencode({"datasetKey":"50c9509d-22c7-4a22-a47d-8c48425ef4a7","kingdomKey":6,"mediaType":"StillImage","limit":3})))
        out["gbif_inat_plant_count"]=j["count"]; urls=[m["identifier"] for r in j["results"] for m in r.get("media",[])][:3]; out["gbif_media_examples"]=urls
        t=time.time(); sz=0
        for u in urls: sz+=len(get(u.replace("original","medium")).read())
        out["img_dl_3_sec"]=round(time.time()-t,2); out["img_bytes"]=sz
        r=json.load(get("https://api.gbif.org/v1/species/match?name=Quercus%20robur")); out["gbif_taxonomy_example"]={k:r.get(k) for k in ["genus","family","order","rank"]}
    except Exception as e: out["gbif"]=str(e)[:200]
    return out
@app.local_entrypoint()
def main():
    print("DC",json.dumps(check.remote(),indent=0)[:6000])
import json
