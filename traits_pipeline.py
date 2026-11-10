"""Trait-identifiability pipeline (prereg v7). Stages T1 select -> T2 fetch -> T3 features -> T4 analysis.
Manifests + gates as in pipeline.py; single container per stage; resumable fetch."""
import modal, json, os, hashlib, time

app = modal.App("plantid-traits")
vol = modal.Volume.from_name("plantid-vol2", create_if_missing=True)
V = "/v"
IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "timm==1.0.20", "huggingface_hub==0.34.4", "numpy==1.26.4",
                    "pandas==2.2.3", "pyarrow==17.0.0", "pillow==10.4.0", "scipy==1.13.1", "open_clip_torch==2.29.0")
       .add_local_python_source("traits_core"))
TRAITS = ["Height", "LeafArea", "SLA", "Leaf_N"]
CFG = {"min_photos": 30, "n_photos": 30, "conf_min": 90, "shard": 500, "threads": 48, "retry_rounds": 3, "min_photos_after_fetch": 20,
       "max_excluded_frac": 0.05, "subset_species": {"smoke": 60, "pilot": 240}, "min_species": {"smoke": 50, "pilot": 200, "full": 2000},
       "alphas": [10 ** (x / 2) for x in range(-2, 9)], "n_boot": {"full": 2000, "pilot": 500, "smoke": 100}, "backbone": "vit_base_patch16_224.augreg_in1k"}
CFG_B = {"mae_vitl": ["timm", "vit_large_patch16_224.mae"], "clip_vitl": ["open_clip", "ViT-L-14", "openai"]}   # separate: base-stage keys unchanged

def _h(o): return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]
def _fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]
def _src(name):
    t = open(__file__).read(); i = t.index(f"\ndef {name}(")
    j = min([x for x in (t.find("\n@app", i + 1), t.find("\ndef ", i + 1)) if x > 0] + [len(t)]); return t[i:j]
def _d(mode, st): p = f"{V}/traits/{mode}/{st}"; os.makedirs(p, exist_ok=True); return p
def _man(mode, st):
    p = f"{_d(mode, st)}/manifest.json"; return json.load(open(p)) if os.path.exists(p) else None
CORE_FUNCS = {"t1_select": ["select_species"], "t2_fetch": ["run_tier", "statuses", "fetch", "_read", "_write"], "t3_features": ["final_photos", "statuses"],
              "t3_bb": [], "t4": ["ridge_fit", "make_folds", "impute", "_cat", "density_features", "fit_density_weights", "apply_density_weights", "conformal_report", "analyze"]}
def _core_src(names):
    import traits_core; t = open(traits_core.__file__).read(); out = []
    for n in names:
        i = t.index(f"\ndef {n}("); j = min([x for x in (t.find("\ndef ", i + 1),) if x > 0] + [len(t)]); out.append(t[i:j])
    return "".join(out)
def _key(name, mode, inputs):
    core = CORE_FUNCS.get(name, CORE_FUNCS["t4"] if name.startswith("t4") else [])
    return {"code": _h(_src(name)), "core": _h(_core_src(core)), "cfg": _h(CFG), "mode": mode, "inputs": inputs}
def _valid(mode, st, key):
    m = _man(mode, st)
    return bool(m) and m["status"] == "PASS" and m["key"] == key and all(os.path.exists(o["path"]) and _fsha(o["path"]) == o["sha"] for o in m["outputs"])
def _in(mode, st):
    m = _man(mode, st); assert m and m["status"] == "PASS", f"upstream {st} not PASS"; return m["key"]["code"] + ":" + ",".join(o["sha"] for o in m["outputs"])
def _gate(n, v, ok): return {"name": n, "value": v, "ok": bool(ok)}
def _write(mode, st, key, outs, gates, extra=None):
    status = "PASS" if all(g["ok"] for g in gates) else "FAIL"
    m = {"stage": st, "key": key, "status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "outputs": [{"path": p, "sha": _fsha(p)} for p in outs], "gates": gates, "extra": extra or {}}
    p = f"{_d(mode, st)}/manifest.json"; json.dump(m, open(p + ".tmp", "w"), indent=1, default=str); os.replace(p + ".tmp", p); vol.commit()
    if status != "PASS": raise RuntimeError(f"{st} GATE FAIL: " + "; ".join(f"{g['name']}={g['value']}" for g in gates if not g["ok"]))

# ------------------------------------------------------------------ T1 select (core.select_species)
@app.function(image=IMG, volumes={V: vol}, timeout=3 * 3600, memory=16384)
def t1_select(mode):
    vol.reload()
    import pandas as pd, urllib.request, urllib.parse, traits_core as core
    from concurrent.futures import ThreadPoolExecutor
    key = _key("t1_select", mode, {"csv": [_fsha(f"{V}/ptn/{s}.csv") for s in ("train", "val")]})
    if _valid(mode, "t1", key): return "SKIP"
    d_all = pd.concat([pd.read_csv(f"{V}/ptn/{s}.csv") for s in ("train", "val")])
    cache = f"{V}/ptn/gbif_match_full.parquet"
    if os.path.exists(cache): gm = pd.read_parquet(cache)
    else:
        def m(nm):
            for _ in range(4):
                try:
                    r = json.load(urllib.request.urlopen(urllib.request.Request("https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode({"name": nm, "kingdom": "Plantae"}), headers={"User-Agent": "research-client"}), timeout=30))
                    return {"Species": nm, **{k: r.get(k) for k in ("speciesKey", "canonicalName", "genus", "family", "order", "rank", "confidence", "matchType", "status")}}
                except Exception: time.sleep(3)
            return {"Species": nm}
        with ThreadPoolExecutor(4) as ex: gm = pd.DataFrame(list(ex.map(m, sorted(d_all.Species.unique()))))
        gm.to_parquet(cache); vol.commit()
    sel, info = core.select_species(d_all, gm, CFG, mode)
    out = f"{_d(mode, 't1')}/sel.parquet"; sel.to_parquet(out, index=False)
    gates = [_gate("n_species", info["n_species"], info["n_species"] >= CFG["min_species"][mode]), _gate("pid_unique", bool(sel.pid.is_unique), sel.pid.is_unique),
             _gate("labels_positive", True, bool((sel[core.LABELS] > 0).all().all()))]
    _write(mode, "t1", key, [out], gates, info); return "PASS"

# ------------------------------------------------------------------ T2 fetch (core.fetch with network getter)
@app.function(image=IMG, volumes={V: vol}, timeout=8 * 3600, memory=8192)
def t2_fetch(mode):
    vol.reload()
    import pandas as pd, io, glob, urllib.request, urllib.error, random, traits_core as core
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    key = _key("t2_fetch", mode, {"t1": _in(mode, "t1")})
    if _valid(mode, "t2", key): return "SKIP"
    sel = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet"); dd = _d(mode, "t2")
    EXT = ("jpg", "jpeg", "JPG", "png", "JPEG", "PNG")
    def get(pid):
        for t in range(4):
            transient = False
            for ext in EXT:
                try:
                    raw = urllib.request.urlopen(urllib.request.Request(f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{ext}", headers={"User-Agent": "research-client"}), timeout=30).read()
                    im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((384, 384)); o = io.BytesIO(); im.save(o, "JPEG", quality=90); return pid, o.getvalue(), "ok"
                except urllib.error.HTTPError as e:
                    if e.code in (403, 404): continue
                    transient = True; break
                except Exception: transient = True; break
            if not transient: return pid, None, "deleted"
            time.sleep(min(30, 2 * 2 ** t) + random.random())
        return pid, None, "failed"
    def tmap(f, xs):
        with ThreadPoolExecutor(CFG["threads"]) as ex: return list(ex.map(f, xs))
    t0 = time.time(); info = core.fetch(sel, dd, get, CFG, commit=vol.commit, map_fn=tmap); info["seconds"] = round(time.time() - t0, 1)
    shards = sorted(glob.glob(f"{dd}/tier*_shard_*.parquet"))
    gates = [_gate("transient_failures", info["n_failed"], info["n_failed"] == 0), _gate("excluded_species_frac<=0.05", round(info["excluded_frac"], 4), info["excluded_frac"] <= CFG["max_excluded_frac"])]
    _write(mode, "t2", key, shards, gates, info); return "PASS"

# ------------------------------------------------------------------ T3 features (core.final_photos)
@app.function(image=IMG, volumes={V: vol}, gpu="H100", timeout=3 * 3600, memory=65536, cpu=16)
def t3_features(mode):
    vol.reload()
    import torch, timm, numpy as np, pandas as pd, glob, io, pyarrow.parquet as pq, traits_core as core
    from PIL import Image
    torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    key = _key("t3_features", mode, {"t2": _in(mode, "t2"), "organ": _fsha(f"{V}/organ_clf.pt")})
    if _valid(mode, "t3", key): return "SKIP"
    dd = _d(mode, "t2"); sel1 = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet")
    s, dead, nex = core.final_photos(sel1, dd, CFG); want = set(s.pid); B = {}
    for p in sorted(glob.glob(f"{dd}/tier*_shard_*.parquet")):
        t = pq.read_table(p).to_pydict(); B.update({a: b for a, b, c in zip(t["pid"], t["bytes"], t["status"]) if c == "ok" and a in want})
    assert set(B) == want, "photo bytes missing for selected pids"
    net = timm.create_model(CFG["backbone"], pretrained=True, num_classes=0).cuda().eval().float()
    pre = timm.data.create_transform(**timm.data.resolve_data_config({}, model=net))
    class DS(torch.utils.data.Dataset):
        def __len__(self_): return len(s)
        def __getitem__(self_, i): return pre(Image.open(io.BytesIO(B[s.pid[i]])).convert("RGB"))
    F = []
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(DS(), batch_size=512, num_workers=14, shuffle=False): F.append(net(x.cuda()).float().cpu())
    F = torch.cat(F); oc = torch.load(f"{V}/organ_clf.pt")
    P = torch.softmax(((F - oc["mu"]) / oc["sd"]) @ oc["W"] + oc["b"], 1).numpy(); F = F.numpy().astype(np.float32)
    out = f"{_d(mode,'t3')}/feats.npz"
    np.savez(out, pid=s.pid.values, X=F, p_leaf=P[:, oc["classes"].index("leaf")], p_habit=P[:, oc["classes"].index("habit")], dead_species=np.array(dead.index), dead_rate=dead.values)
    gates = [_gate("finite", bool(np.isfinite(F).all()), np.isfinite(F).all()), _gate("n_rows", len(F), len(F) == len(s))]
    _write(mode, "t3", key, [out], gates, {"n_species_kept": int(s.Species.nunique()), "n_species_excluded_lt20": nex, "n_images": len(s)}); return "PASS"

# ------------------------------------------------------------------ T4 analysis (core.analyze), parallel over CV levels
@app.function(image=IMG, volumes={V: vol}, timeout=3 * 3600, memory=65536, cpu=16)
def t4_part(mode, feats_stage, part, v10=False):
    vol.reload()
    import numpy as np, pandas as pd, traits_core as core
    with np.load(f"{_d(mode, feats_stage)}/feats.npz", allow_pickle=True) as z: a = {k: z[k] for k in z.files}
    sel = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet")
    kw = dict(levels=(), do_warmup=True) if part == "warmup" else dict(levels=(part,), do_warmup=False)
    cfg = dict(CFG, density_sd_feature=True) if v10 else CFG
    return core.analyze(a["pid"], a["X"], a["p_leaf"], a["p_habit"], a["dead_species"], a["dead_rate"], sel, cfg, CFG["n_boot"][mode], return_parts=True, **kw)

def _t4_run(mode, feats_stage, st, key, v10=False):
    import numpy as np, pandas as pd, traits_core as core
    t0 = time.time(); parts = list(t4_part.map([mode] * 5, [feats_stage] * 5, ["Species", "genus", "family", "order", "warmup"], [v10] * 5))
    store = {}
    for p in parts[:4]: store.update(p["store"])
    with np.load(f"{_d(mode, feats_stage)}/feats.npz", allow_pickle=True) as z: a = {k: z[k] for k in z.files}
    sel = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet")
    cfg = dict(CFG, density_sd_feature=True) if v10 else CFG
    res, *_ = core.analyze(a["pid"], a["X"], a["p_leaf"], a["p_habit"], a["dead_species"], a["dead_rate"], sel, cfg, CFG["n_boot"][mode],
                           precomputed={"store": store, "warmup": parts[4]["warmup"]})
    out = f"{_d(mode, st)}/results.json"; json.dump(res, open(out, "w"), indent=1, sort_keys=True, default=str)
    w0 = min(v["r2_regress_img"] for v in res["warmup_seen"].values())
    _write(mode, st, key, [out], [_gate("warmup_seen_regression_r2>0", round(w0, 4), w0 > 0)], {"n_species": res["n_species"], "seconds": round(time.time() - t0, 1)}); return "PASS"

@app.function(image=IMG, volumes={V: vol}, timeout=6 * 3600, memory=65536, cpu=8)
def t4_analysis(mode, tag: str = ""):
    vol.reload(); st = f"t4{tag}"
    key = _key("t4_analysis", mode, {"t1": _in(mode, "t1"), "t3": _in(mode, "t3")})
    if _valid(mode, st, key): return "SKIP"
    return _t4_run(mode, "t3", st, key)

def _cmp(a, b):
    """(structure_equal, max_abs_numeric_diff): reproducibility across heterogeneous CPUs (float summation order)"""
    if isinstance(a, dict):
        if set(a) != set(b): return False, float("inf")
        rs = [_cmp(a[k], b[k]) for k in a]; return all(r[0] for r in rs), max([r[1] for r in rs] + [0.0])
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b): return False, float("inf")
        rs = [_cmp(x, y) for x, y in zip(a, b)]; return all(r[0] for r in rs), max([r[1] for r in rs] + [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != a and b != b: return True, 0.0
        return True, abs(a - b)
    return a == b, 0.0

@app.function(image=IMG, volumes={V: vol}, timeout=20 * 3600)
def orchestrate(mode, upto="all"):
    log = []
    for nm, fn in (("t1", t1_select), ("t2", t2_fetch), ("t3", t3_features)):
        r = fn.remote(mode); log.append(f"{nm}:{r}"); print(nm, r, flush=True); vol.reload()
        if upto == nm: return log
    h1, h2 = t4_analysis.spawn(mode), t4_analysis.spawn(mode, "_rep"); r1 = h1.get(); r = h2.get(); vol.reload()
    log.append(f"t4:{r1}"); print("t4", r1, flush=True)
    A_, B_ = (json.load(open(f"{_d(mode, x)}/results.json")) for x in ("t4", "t4_rep")); A_.pop("seconds", None); B_.pop("seconds", None)
    ok, md = _cmp(A_, B_); same = ok and md <= 1e-9
    log.append(f"t4_rep:{r}:identical={same}:maxdiff={md:.2e}:structure={ok}"); print("t4_rep identical", same, md, flush=True)
    return log


# ------------------------------------------------------------------ v8-B: alternative backbones (same photos, same core analysis)
@app.function(image=IMG, volumes={V: vol}, gpu="H100", timeout=3 * 3600, memory=65536, cpu=16)
def t3_bb(mode, bb):
    vol.reload()
    import torch, numpy as np, pandas as pd, glob, io, pyarrow.parquet as pq, traits_core as core
    from PIL import Image
    torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    st = f"t3_{bb}"; key = _key("t3_bb", mode, {"t3": _in(mode, "t3"), "bb": CFG_B[bb]})
    if _valid(mode, st, key): return "SKIP"
    with np.load(f"{_d(mode,'t3')}/feats.npz", allow_pickle=True) as z: base = {k: z[k] for k in z.files}
    want = list(base["pid"]); B = {}
    for p in sorted(glob.glob(f"{_d(mode,'t2')}/tier*_shard_*.parquet")):
        t = pq.read_table(p).to_pydict(); B.update({a: b for a, b, c in zip(t["pid"], t["bytes"], t["status"]) if c == "ok"})
    assert all(w in B for w in want), "photo bytes missing"
    kind, *spec = CFG_B[bb]
    if kind == "timm":
        import timm
        net = timm.create_model(spec[0], pretrained=True, num_classes=0); pre = timm.data.create_transform(**timm.data.resolve_data_config({}, model=net)); fwd = net
    else:
        import open_clip
        net, _, pre = open_clip.create_model_and_transforms(spec[0], pretrained=spec[1]); fwd = net.encode_image
    net = net.cuda().eval().float()
    class DS(torch.utils.data.Dataset):
        def __len__(self_): return len(want)
        def __getitem__(self_, i): return pre(Image.open(io.BytesIO(B[want[i]])).convert("RGB"))
    F = []
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(DS(), batch_size=256, num_workers=14, shuffle=False): F.append(fwd(x.cuda()).float().cpu())
    F = torch.cat(F).numpy().astype(np.float32)
    out = f"{_d(mode, st)}/feats.npz"; np.savez(out, **{**base, "X": F})
    gates = [_gate("finite", bool(np.isfinite(F).all()), np.isfinite(F).all()), _gate("n_rows", len(F), len(F) == len(want)), _gate("nonconstant", float(F.std(0).mean()), F.std(0).mean() > 1e-6)]
    _write(mode, st, key, [out], gates, {"dim": int(F.shape[1]), "n_images": len(F)}); return "PASS"

@app.function(image=IMG, volumes={V: vol}, timeout=6 * 3600, memory=65536, cpu=8)
def t4_bb(mode, bb, tag: str = ""):
    vol.reload(); st = f"t4_{bb}{tag}"
    key = _key("t4_bb", mode, {"t1": _in(mode, "t1"), "t3": _in(mode, f"t3_{bb}")})
    if _valid(mode, st, key): return "SKIP"
    return _t4_run(mode, f"t3_{bb}", st, key)

@app.function(image=IMG, volumes={V: vol}, timeout=20 * 3600)
def orchestrate_b(mode):
    log = []
    for bb in CFG_B:
        r = t3_bb.remote(mode, bb); log.append(f"t3_{bb}:{r}"); vol.reload()
        h1, h2 = t4_bb.spawn(mode, bb), t4_bb.spawn(mode, bb, "_rep"); log.append(f"t4_{bb}:{h1.get()}"); log.append(f"t4rep_{bb}:{h2.get()}"); vol.reload()
        A_, B_ = (json.load(open(f"{_d(mode, 't4_' + bb + x)}/results.json")) for x in ("", "_rep")); ok, md = _cmp(A_, B_)
        log.append(f"rep_{bb}:identical={ok and md <= 1e-9}:maxdiff={md:.2e}"); print("rep", bb, ok, md, flush=True)
    return log

@app.function(image=IMG, volumes={V: vol}, timeout=6 * 3600, memory=65536, cpu=8)
def t4_v10(mode, feats_stage, tag: str = ""):
    vol.reload(); st = f"t4v10_{feats_stage}{tag}"
    key = _key("t4_v10", mode, {"t1": _in(mode, "t1"), "feats": _in(mode, feats_stage)})
    if _valid(mode, st, key): return "SKIP"
    return _t4_run(mode, feats_stage, st, key, v10=True)

@app.function(image=IMG, volumes={V: vol}, timeout=6 * 3600)
def orchestrate_v10(mode):
    fs = ["t3", "t3_mae_vitl", "t3_clip_vitl"]
    hs = [(f, t4_v10.spawn(mode, f), t4_v10.spawn(mode, f, "_rep")) for f in fs]; log = []
    for f, h1, h2 in hs:
        log.append(f"{f}:{h1.get()}/{h2.get()}"); vol.reload()
        A_, B_ = (json.load(open(f"{_d(mode, 't4v10_' + f + x)}/results.json")) for x in ("", "_rep")); ok, md = _cmp(A_, B_)
        log.append(f"rep_{f}:identical={ok and md <= 1e-9}:maxdiff={md:.2e}")
    return log

@app.function(image=IMG, volumes={V: vol}, timeout=600)
def status(mode):
    import glob
    vol.reload(); out = []
    for st in sorted(os.listdir(f"{V}/traits/{mode}")) if os.path.exists(f"{V}/traits/{mode}") else []:
        m = _man(mode, st); out.append(f"{st:7s} " + (f"{m['status']} {m['time']} " + " | ".join(f"{g['name']}={g['value']}" for g in m["gates"]) + f" | {json.dumps(m['extra'], default=str)[:600]}" if m else "INCOMPLETE"))
    out.append("t2 shards on disk: " + str(len(glob.glob(f"{V}/traits/{mode}/t2/tier*_shard_*.parquet"))))
    return "\n".join(out)

@app.local_entrypoint()
def main(mode: str = "smoke", upto: str = "all", show: int = 0):
    if show: print(status.remote(mode)); return
    if upto == "b": print("RUNB", orchestrate_b.remote(mode)); return
    if upto == "v10": print("RUNV10", orchestrate_v10.remote(mode)); return
    print("RUN", orchestrate.remote(mode, upto))
