"""Trait-identifiability pipeline (prereg v7). Stages T1 select -> T2 fetch -> T3 features -> T4 analysis.
Manifests + gates as in pipeline.py; single container per stage; resumable fetch."""
import modal, json, os, hashlib, time

app = modal.App("plantid-traits")
vol = modal.Volume.from_name("plantid-vol2", create_if_missing=True)
V = "/v"
IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "timm==1.0.20", "huggingface_hub==0.34.4", "numpy==1.26.4",
                    "pandas==2.2.3", "pyarrow==17.0.0", "pillow==10.4.0", "scipy==1.13.1"))
TRAITS = ["Height", "LeafArea", "SLA", "Leaf_N"]
CFG = {"min_photos": 30, "n_photos": 30, "n_cand": None, "max_excluded_frac": 0.05, "conf_min": 90, "smoke_species": 60, "shard": 500, "threads": 48, "t1_min_species": 2000,
       "retry_rounds": 3, "min_photos_after_fetch": 20, "alphas": [10 ** (x / 2) for x in range(-2, 9)],
       "n_boot": {"full": 2000, "smoke": 100}, "backbone": "vit_base_patch16_224.augreg_in1k"}

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
def _key(name, mode, inputs): return {"code": _h(_src(name)), "cfg": _h(CFG), "mode": mode, "inputs": inputs}
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

# ------------------------------------------------------------------ T1 select
@app.function(image=IMG, volumes={V: vol}, timeout=3 * 3600, memory=16384)
def t1_select(mode):
    vol.reload()
    import pandas as pd, numpy as np, urllib.request, urllib.parse
    from concurrent.futures import ThreadPoolExecutor
    key = _key("t1_select", mode, {"csv": [_fsha(f"{V}/ptn/{s}.csv") for s in ("train", "val")]})
    if _valid(mode, "t1", key): return "SKIP"
    d = pd.concat([pd.read_csv(f"{V}/ptn/{s}.csv") for s in ("train", "val")])
    d["pid"] = d.Image.str.extract(r"photo_(\d+)")[0]; d_all = d.copy()
    n_nopid = int(d.pid.isna().sum()); d = d[d.pid.notna()].drop_duplicates(["Species", "pid"])
    multi = d.groupby("pid").Species.nunique(); shared = set(multi[multi > 1].index)
    links = d[d.pid.isin(shared)].groupby("pid").Species.apply(lambda x: sorted(set(x))).tolist()
    d = d[~d.pid.isin(shared)]
    n = d.groupby("Species").size(); pool = sorted(n[n >= CFG["min_photos"]].index)
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
        with ThreadPoolExecutor(4) as ex: gm = pd.DataFrame(list(ex.map(m, sorted(d.Species.unique()))))
        gm.to_parquet(cache); vol.commit()
    g = gm[gm.Species.isin(pool)].copy()
    ok = (g["rank"] == "SPECIES") & (g.confidence >= CFG["conf_min"]) & g.genus.notna() & g.family.notna() & g.order.notna()
    g = g[ok]; dup = g.speciesKey.duplicated(keep=False); n_coll = int(dup.sum()); g = g[~dup]
    sp = sorted(g.Species)
    parent = {x: x for x in sp}                                   # union-find over species that shared photos
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for L in links:
        L = [x for x in L if x in parent]
        for x in L[1:]: parent[find(x)] = find(L[0])
    link_group = {x: find(x) for x in sp}
    if mode == "smoke": sp = [sp[i] for i in np.random.default_rng(0).permutation(len(sp))[:CFG["smoke_species"]]]
    lab = d_all.groupby("Species")[[f"{t}_median" for t in TRAITS]].first()     # first non-null per species (some rows carry NaN labels)
    rows = []
    for s in sp:
        x = d[d.Species == s].drop_duplicates("pid").sort_values("pid")
        c = x.iloc[np.random.default_rng(0).permutation(len(x))][["Species", "pid"]].copy(); c["rank"] = np.arange(len(c)); rows.append(c)
    sel = pd.concat(rows).merge(g[["Species", "genus", "family", "order", "speciesKey"]], on="Species").merge(lab, left_on="Species", right_index=True)
    sel["link_group"] = sel.Species.map(link_group)
    sel = sel.sort_values(["Species", "rank"]).reset_index(drop=True)
    out = f"{_d(mode, 't1')}/sel.parquet"; sel.to_parquet(out, index=False)
    gates = [_gate("n_species", sel.Species.nunique(), sel.Species.nunique() >= (50 if mode == "smoke" else CFG["t1_min_species"])),
             _gate("pid_unique", bool(sel.pid.is_unique), sel.pid.is_unique),
             _gate("labels_positive", True, bool((sel[[f"{t}_median" for t in TRAITS]] > 0).all().all())),
             _gate("candidates_per_species>=30", int(sel.groupby("Species").size().min()), sel.groupby("Species").size().min() >= CFG["n_photos"])]
    _write(mode, "t1", key, [out], gates, {"pool": len(pool), "after_tax_filter": int(ok.sum()), "synonym_collisions_dropped": n_coll, "n_species": sel.Species.nunique(), "rows_without_inat_pid": n_nopid, "shared_pids_dropped": len(shared),
                                           "species_in_linked_groups": int(pd.Series(link_group).map(pd.Series(link_group).value_counts()).gt(1).sum()),
                                           "genera": int(sel.genus.nunique()), "families": int(sel.family.nunique()), "orders": int(sel.order.nunique())})
    return "PASS"

# ------------------------------------------------------------------ T2 fetch (resumable shards)
@app.function(image=IMG, volumes={V: vol}, timeout=8 * 3600, memory=8192)
def t2_fetch(mode):
    vol.reload()
    import pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq, io, glob, urllib.request, urllib.error, random
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    key = _key("t2_fetch", mode, {"t1": _in(mode, "t1")})
    if _valid(mode, "t2", key): return "SKIP"
    sel = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet"); S = CFG["shard"]; dd = _d(mode, "t2")
    EXT = ("jpg", "jpeg", "JPG", "png", "JPEG", "PNG")
    def get(pid):
        """returns (pid, bytes|None, status in ok/deleted/failed)"""
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
    def run_tier(name, pids):
        nsh = -(-len(pids) // S)
        for si in range(nsh):
            p = f"{dd}/{name}_shard_{si:05d}.parquet"; chunk = pids[si * S:(si + 1) * S]
            for rnd in range(CFG["retry_rounds"] + 1):
                if os.path.exists(p):
                    t = pq.read_table(p).to_pydict()
                    if t["pid"] != chunk or "status" not in t: os.remove(p); continue
                    bad = [i for i, st_ in enumerate(t["status"]) if st_ == "failed"]
                    if not bad: break
                    with ThreadPoolExecutor(CFG["threads"]) as ex: rr = list(ex.map(get, [chunk[i] for i in bad]))
                    for i, r in zip(bad, rr): t["bytes"][i], t["status"][i] = r[1], r[2]
                else:
                    with ThreadPoolExecutor(CFG["threads"]) as ex: res = list(ex.map(get, chunk))
                    t = {"pid": [r[0] for r in res], "bytes": [r[1] for r in res], "status": [r[2] for r in res]}
                pq.write_table(pa.table(t), p + ".tmp"); os.replace(p + ".tmp", p); vol.commit()
        for p in glob.glob(f"{dd}/{name}_shard_*.parquet"):
            if int(p[-13:-8]) >= nsh: os.remove(p)
    def statuses(prefix="tier"):
        st_ = {}
        for p in sorted(glob.glob(f"{dd}/{prefix}*_shard_*.parquet")):
            t = pq.read_table(p, columns=["pid", "status"]).to_pydict(); st_.update(zip(t["pid"], t["status"]))
        return st_
    for p in glob.glob(f"{dd}/*.parquet"):                       # legacy-format shards (no status column) are stale
        if "status" not in pq.read_schema(p).names: os.remove(p)
    run_tier("tier1", sel[sel["rank"] < CFG["n_photos"]].pid.tolist())
    st_ = statuses("tier1"); n_fail1 = sum(v == "failed" for v in st_.values())
    n_ok = sel[sel.pid.map(st_).eq("ok")].groupby("Species").size().reindex(sel.Species.unique(), fill_value=0)
    short = sorted(n_ok[n_ok < CFG["n_photos"]].index)
    run_tier("tier2", sel[(sel["rank"] >= CFG["n_photos"]) & sel.Species.isin(short)].pid.tolist())
    st_ = statuses(); shards = sorted(glob.glob(f"{dd}/tier*_shard_*.parquet"))
    sel["status"] = sel.pid.map(st_); tried = sel[sel.status.notna()]
    n_failed = int((tried.status == "failed").sum()); dead = float((tried.status == "deleted").mean())
    per = sel[sel.status == "ok"].groupby("Species").size().reindex(sel.Species.unique(), fill_value=0).clip(upper=CFG["n_photos"]); mn = int(per.min())
    t1 = sel[sel["rank"] < CFG["n_photos"]].groupby("Species").agg(dead=("status", lambda x: (x == "deleted").mean()), **{t: (f"{t}_median", "first") for t in TRAITS})
    from scipy.stats import spearmanr
    dead_vs_trait = {t: float(spearmanr(t1.dead, np.log(t1[t])).correlation) for t in TRAITS}
    excl = per[per < CFG["min_photos_after_fetch"]].index; ex_frac = float(len(excl) / len(per))
    lab_ = sel.drop_duplicates("Species").set_index("Species")
    ex_cmp = {t: {"excluded_mean_log": float(np.log(lab_.loc[excl, f"{t}_median"]).mean()) if len(excl) else None,
                  "retained_mean_log": float(np.log(lab_.drop(index=excl)[f"{t}_median"]).mean())} for t in TRAITS}
    gates = [_gate("transient_failures", n_failed, n_failed == 0), _gate("excluded_species_frac<=0.05", round(ex_frac, 4), ex_frac <= CFG["max_excluded_frac"])]
    _write(mode, "t2", key, shards, gates, {"n_excluded_lt20": int(len(excl)), "excluded_vs_retained": ex_cmp, "min_photos_any_species": mn, "dead_rate": round(dead, 4), "n_tried": len(tried), "frac_species_lt30": round(float((per < CFG["n_photos"]).mean()), 4),
                                            "dead_vs_trait_spearman": dead_vs_trait, "tier1_failed_before_retry": n_fail1}); return "PASS"

# ------------------------------------------------------------------ T3 features + organ probabilities
@app.function(image=IMG, volumes={V: vol}, gpu="A10G", timeout=3 * 3600, memory=32768)
def t3_features(mode):
    vol.reload()
    import torch, timm, numpy as np, pandas as pd, glob, io, pyarrow.parquet as pq
    from PIL import Image
    torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    key = _key("t3_features", mode, {"t2": _in(mode, "t2"), "organ": _fsha(f"{V}/organ_clf.pt")})
    if _valid(mode, "t3", key): return "SKIP"
    B, ST = {}, {}
    for p in sorted(glob.glob(f"{_d(mode,'t2')}/tier*_shard_*.parquet")):
        t = pq.read_table(p).to_pydict(); B.update({a: b for a, b, c in zip(t["pid"], t["bytes"], t["status"]) if c == "ok"})
        if "/tier1_" in p: ST.update(zip(t["pid"], t["status"]))
    sel = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet"); sel = sel[sel.pid.isin(B)].sort_values(["Species", "rank"])
    sel = sel.groupby("Species").head(CFG["n_photos"]).reset_index(drop=True)
    cnt = sel.groupby("Species").size(); keep = cnt[cnt >= CFG["min_photos_after_fetch"]].index; n_excl = int((cnt < CFG["min_photos_after_fetch"]).sum())
    sel = sel[sel.Species.isin(keep)].reset_index(drop=True)
    s1 = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet"); s1 = s1[s1["rank"] < CFG["n_photos"]]; s1["st"] = s1.pid.map(ST)
    dead = s1.groupby("Species").st.apply(lambda x: float((x == "deleted").mean())).reindex(sorted(keep))
    net = timm.create_model(CFG["backbone"], pretrained=True, num_classes=0).cuda().eval().float()
    pre = timm.data.create_transform(**timm.data.resolve_data_config({}, model=net))
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(sel)
        def __getitem__(s, i): return pre(Image.open(io.BytesIO(B[sel.pid[i]])).convert("RGB"))
    F = []
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(DS(), batch_size=128, num_workers=4, shuffle=False): F.append(net(x.cuda()).float().cpu())
    F = torch.cat(F); oc = torch.load(f"{V}/organ_clf.pt")
    P = torch.softmax(((F - oc["mu"]) / oc["sd"]) @ oc["W"] + oc["b"], 1).numpy(); F = F.numpy().astype(np.float32)
    out = f"{_d(mode,'t3')}/feats.npz"; np.savez(out, pid=sel.pid.values, X=F, dead_species=np.array(dead.index), dead_rate=dead.values, p_leaf=P[:, oc["classes"].index("leaf")], p_habit=P[:, oc["classes"].index("habit")])
    gates = [_gate("finite", bool(np.isfinite(F).all()), np.isfinite(F).all()), _gate("n_rows", len(F), len(F) == len(sel))]
    _write(mode, "t3", key, [out], gates, {"n_species_kept": int(len(keep)), "n_species_excluded_lt20": n_excl, "dim": int(F.shape[1]), "mean_p_leaf": float(P[:, 0].mean())}); return "PASS"

# ------------------------------------------------------------------ T4 analysis
def _ridge_fit(X, Y, alphas):
    """returns dict alpha -> (W, b); X standardized outside"""
    import numpy as np
    xm, ym = X.mean(0), Y.mean(0); Xc, Yc = X - xm, Y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); UtY = U.T @ Yc
    return {a: (Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY), ym - xm @ (Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY))) for a in alphas}

@app.function(image=IMG, volumes={V: vol}, timeout=6 * 3600, memory=65536, cpu=8)
def t4_analysis(mode, tag: str = ""):
    vol.reload()
    import numpy as np, pandas as pd
    st = f"t4{tag}"
    key = _key("t4_analysis", mode, {"t1": _in(mode, "t1"), "t3": _in(mode, "t3"), "ridge": _h(_src("_ridge_fit"))})
    if _valid(mode, st, key): return "SKIP"
    with np.load(f"{_d(mode,'t3')}/feats.npz", allow_pickle=True) as z: pid, X, pl, ph, dsp, drate = z["pid"], z["X"].astype(np.float64), z["p_leaf"], z["p_habit"], z["dead_species"], z["dead_rate"]
    assert len(set(pid)) == len(pid), "duplicate pids in features"
    sel = pd.read_parquet(f"{_d(mode,'t1')}/sel.parquet").set_index("pid").loc[pid].reset_index()
    sp = sorted(sel.Species.unique()); si = sel.Species.map({s: i for i, s in enumerate(sp)}).values
    tax = sel.drop_duplicates("Species").set_index("Species").loc[sp]
    Y = np.log(tax[[f"{t}_median" for t in TRAITS]].values)          # species x trait
    Yimg = Y[si]; NB = CFG["n_boot"]["full" if mode == "full" else "smoke"]; A = CFG["alphas"]
    def folds(level, seed=0):
        """groups = taxonomic level value merged with link groups (species that shared photos never split across folds)"""
        lab = tax[level].values if level != "Species" else np.array(sp); par = {}
        def f_(x):
            while par.setdefault(x, x) != x: par[x] = par[par[x]]; x = par[x]
            return x
        for a_, b_ in zip(("L:" + pd.Series(lab).astype(str)).values, ("G:" + tax.link_group.astype(str)).values): par[f_(a_)] = f_(b_)
        g = np.array([f_("L:" + str(x)) for x in lab]); u = np.array(sorted(set(g)))
        perm = u[np.random.default_rng(seed).permutation(len(u))]; m = {x: i % 5 for i, x in enumerate(perm)}; return np.array([m[x] for x in g]), g
    def impute(tr_mask):
        trt = tax[tr_mask]; pred = np.full(Y.shape, np.nan); used = np.array(["global"] * len(sp), dtype=object)
        for lv in ("genus", "family", "order"):
            gg = pd.DataFrame(Y[tr_mask], index=trt[lv].values).groupby(level=0); mm = gg.mean(); cnt = gg.size(); mm = mm[cnt >= 2]
            v = mm.reindex(tax[lv].values).values; fill = np.isnan(pred[:, 0]) & ~np.isnan(v[:, 0])
            pred[fill] = v[fill]; used[fill & (used == "global")] = lv
        g = np.isnan(pred[:, 0]); pred[g] = Y[tr_mask].mean(0); return pred, used
    def species_mean(P_img, w=None):
        if w is None: w = np.ones(len(P_img))
        num = np.zeros((len(sp), P_img.shape[1])); den = np.zeros(len(sp)); np.add.at(num, si, P_img * w[:, None]); np.add.at(den, si, w); return num / np.maximum(den, 1e-12)[:, None]
    def fit_predict(trS, teS, level, target_sp):
        """nested CV ridge on images of training species; returns image-level predictions for test images. target_sp: species x k targets"""
        tri = np.isin(si, np.where(trS)[0]); tei = np.isin(si, np.where(teS)[0])
        mu, sd = X[tri].mean(0), X[tri].std(0) + 1e-6; Xs = (X - mu) / sd
        inner, _ = folds(level, seed=1); inner = np.where(trS, inner, -1)
        err = {a: 0.0 for a in A}
        for f in range(5):
            itr = trS & (inner != f); ite = trS & (inner == f)
            if ite.sum() == 0 or itr.sum() == 0: continue
            mi = np.isin(si, np.where(itr)[0]); W = _ridge_fit(Xs[mi], target_sp[si[mi]], A)
            mt = np.isin(si, np.where(ite)[0])
            for a, (w, b) in W.items():
                pim = np.zeros((len(Xs), target_sp.shape[1])); pim[mt] = Xs[mt] @ w + b
                pspc = species_mean(pim, mt.astype(float))[ite]; err[a] += ((pspc - target_sp[ite]) ** 2).sum()
        a_best = min(err, key=err.get); W = _ridge_fit(Xs[tri], target_sp[si[tri]], [a_best])[a_best]
        pim = np.zeros((len(Xs), target_sp.shape[1])); pim[tei] = Xs[tei] @ W[0] + W[1]; return pim, a_best
    res = {"n_species": len(sp), "n_images": len(X)}; store = {}
    for level in ("Species", "genus", "family", "order"):
        fo, _ = folds(level); P_imp = np.zeros_like(Y); used = np.empty(len(sp), dtype=object); P_imgimg = np.zeros((len(X), Y.shape[1])); P_inc = np.zeros((len(X), Y.shape[1])); alphas = []
        for f in range(5):
            te = fo == f; tr = ~te
            pi, u = impute(tr); P_imp[te] = pi[te]; used[te] = u[te]
            pim, a = fit_predict(tr, te, level, Y); P_imgimg += pim; alphas.append(a)
            inner, _ = folds(level, seed=1); resid = np.zeros_like(Y)
            for k in range(5):                                  # training residuals from inner-fold imputation (species never imputes itself)
                ik = tr & (inner == k)
                if ik.sum() == 0: continue
                pk, _ = impute(tr & (inner != k)); resid[ik] = Y[ik] - pk[ik]
            pr, _ = fit_predict(tr, te, level, resid); P_inc += pr
        store[level] = {"imp": P_imp, "img": species_mean(P_imgimg), "inc": P_imp + species_mean(P_inc), "used": used, "alphas": alphas,
                        "img_wleaf": species_mean(P_imgimg, pl), "img_whabit": species_mean(P_imgimg, ph)}
        # top-10 by organ prob
        for nm, pr_ in (("img_top_leaf", pl), ("img_top_habit", ph)):
            w = np.zeros(len(X))
            for s_ in range(len(sp)):
                idx = np.where(si == s_)[0]; w[idx[np.argsort(-pr_[idx])[:10]]] = 1.0
            store[level][nm] = species_mean(P_imgimg, w)
    gen = tax.genus.values; ug = np.array(sorted(set(gen))); gidx = np.searchsorted(ug, gen)
    rng = np.random.default_rng(0); BW = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(NB)]
    def r2(pred, mask, j): y = Y[mask, j]; return 1 - ((y - pred[mask, j]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    def boot_diff(pa, pb, mask, j):
        y = Y[:, j]; ea = (y - pa[:, j]) ** 2; eb = (y - pb[:, j]) ** 2; out = []
        for w in BW:
            ws = w[gidx] * mask; my = (ws * y).sum() / ws.sum(); sst = (ws * (y - my) ** 2).sum()
            out.append(((ws * eb).sum() - (ws * ea).sum()) / sst)
        d0 = r2(pa, mask, j) - r2(pb, mask, j); out = np.array(out)
        return [float(d0), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())]
    for level, S in store.items():
        R = {}
        strata = {"pooled": np.ones(len(sp), bool)}
        if level == "Species": strata.update({"genus_imputed": S["used"] == "genus", "family_imputed": S["used"] == "family", "order_or_global": np.isin(S["used"], ["order", "global"])})
        for sn, m in strata.items():
            if m.sum() < 10: continue
            for j, t in enumerate(TRAITS):
                e = {"n": int(m.sum()), "r2_imp": float(r2(S["imp"], m, j)), "r2_img": float(r2(S["img"], m, j)), "r2_inc": float(r2(S["inc"], m, j)),
                     "d_img_minus_imp": boot_diff(S["img"], S["imp"], m, j), "d_inc_minus_imp": boot_diff(S["inc"], S["imp"], m, j)}
                if sn in ("pooled", "genus_imputed"):
                    wkey = "img_whabit" if t == "Height" else "img_wleaf"; tkey = "img_top_habit" if t == "Height" else "img_top_leaf"
                    e["d_weighted_minus_unweighted"] = boot_diff(S[wkey], S["img"], m, j); e["d_top10_minus_unweighted"] = boot_diff(S[tkey], S["img"], m, j)
                R[f"{sn}|{t}"] = e
        R["imputation_level_counts"] = pd.Series(S["used"]).value_counts().to_dict(); R["alphas"] = S["alphas"]; res[level] = R
    # warm-up: seen species (image-level 5-fold within species), nearest-centroid species lookup vs direct regression
    rng2 = np.random.default_rng(0); ff = np.zeros(len(X), int)
    for s_ in range(len(sp)):
        idx = np.where(si == s_)[0]; ff[idx] = rng2.permutation(len(idx)) % 5
    Pl, Pr = np.zeros((len(X), Y.shape[1])), np.zeros((len(X), Y.shape[1]))
    for f in range(5):
        tr = ff != f; te = ~tr; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
        C = np.zeros((len(sp), X.shape[1])); np.add.at(C, si[tr], Xs[tr]); C /= np.bincount(si[tr], minlength=len(sp))[:, None]
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True); Q = Xs[te] / np.linalg.norm(Xs[te], axis=1, keepdims=True)
        Pl[te] = Y[(Q @ Cn.T).argmax(1)]
        w, b = _ridge_fit(Xs[tr], Yimg[tr], [100.0])[100.0]; Pr[te] = Xs[te] @ w + b
    res["warmup_seen"] = {t: {"r2_lookup_img": float(1 - ((Yimg[:, j] - Pl[:, j]) ** 2).sum() / ((Yimg[:, j] - Yimg[:, j].mean()) ** 2).sum()),
                              "r2_regress_img": float(1 - ((Yimg[:, j] - Pr[:, j]) ** 2).sum() / ((Yimg[:, j] - Yimg[:, j].mean()) ** 2).sum())} for j, t in enumerate(TRAITS)}
    # sensitivity (v7.3): primary stratum restricted to species with tier-1 dead rate <= median (same out-of-fold predictions)
    dr = pd.Series(drate, index=dsp).reindex(sp).values; low = dr <= np.nanmedian(dr); S0 = store["Species"]; gmask = S0["used"] == "genus"
    res["sensitivity_low_dead"] = {t: {"n": int((gmask & low).sum()), "d_img_minus_imp": boot_diff(S0["img"], S0["imp"], gmask & low, j)} for j, t in enumerate(TRAITS)}
    from scipy.stats import spearmanr
    res["dead_vs_trait_spearman_full"] = {t: float(spearmanr(dr, Y[:, j]).correlation) for j, t in enumerate(TRAITS)}
    # primary test: Species-level CV, genus-imputed stratum, Holm over 4 traits (one-sided bootstrap p)
    pv = {t: res["Species"][f"genus_imputed|{t}"]["d_img_minus_imp"][3] for t in TRAITS if f"genus_imputed|{t}" in res["Species"]}
    order_ = sorted(pv, key=pv.get); verdict = {}; stop = False
    for i, t in enumerate(order_):
        thr = 0.05 / (len(order_) - i); sig = (not stop) and pv[t] <= thr; stop = stop or not sig
        verdict[t] = {"p_boot": pv[t], "holm_threshold": thr, "significant": bool(sig)}
    if verdict.get("Leaf_N", {}).get("significant"): verdict["Leaf_N"]["flag"] = "anomalous_inspect"
    if verdict.get("Height", {}).get("significant"): verdict["Height"]["flag"] = "mandatory_leakage_check_before_interpretation"
    res["primary_verdict"] = verdict
    out = f"{_d(mode, st)}/results.json"; json.dump(res, open(out, "w"), indent=1, sort_keys=True, default=str)
    w0 = min(v["r2_regress_img"] for v in res["warmup_seen"].values())
    gates = [_gate("warmup_seen_regression_r2>0", round(w0, 4), w0 > 0)]
    _write(mode, st, key, [out], gates, {"n_species": len(sp)}); return "PASS"

@app.function(image=IMG, volumes={V: vol}, timeout=20 * 3600)
def orchestrate(mode, upto="t4"):
    log = []
    for nm, fn in (("t1", t1_select), ("t2", t2_fetch), ("t3", t3_features), ("t4", t4_analysis)):
        r = fn.remote(mode); log.append(f"{nm}:{r}"); print(nm, r, flush=True); vol.reload()
        if upto == nm: return log
    r = t4_analysis.remote(mode, "_rep"); vol.reload()
    same = _fsha(f"{_d(mode,'t4')}/results.json") == _fsha(f"{_d(mode,'t4_rep')}/results.json"); log.append(f"t4_rep:{r}:identical={same}"); print("t4_rep identical", same, flush=True)
    return log

@app.function(image=IMG, volumes={V: vol}, timeout=600)
def status(mode):
    vol.reload(); out = []
    for st in sorted(os.listdir(f"{V}/traits/{mode}")) if os.path.exists(f"{V}/traits/{mode}") else []:
        m = _man(mode, st); out.append(f"{st:6s} " + (f"{m['status']} {m['time']} " + " | ".join(f"{g['name']}={g['value']}" for g in m["gates"]) + f" | {m['extra']}" if m else "INCOMPLETE"))
    return "\n".join(out)

@app.local_entrypoint()
def main(mode: str = "smoke", upto: str = "t4", show: int = 0):
    if show: print(status.remote(mode)); return
    print("RUN", orchestrate.remote(mode, upto))
