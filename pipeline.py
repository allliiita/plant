"""Reliable, resumable pipeline: S0 env -> S1 select -> S2 fetch -> S3 features -> S4 probes.
Every stage writes runs/<mode>/<stage>/manifest.json (inputs, code hash, config hash, outputs sha256, gates, status).
A stage is skipped iff a PASS manifest exists with identical code/config/input hashes. Gates fail loudly.
Single container at a time; no parallel fan-out."""
import modal, json, os, hashlib, inspect, time

app = modal.App("plantid-pipe")
vol = modal.Volume.from_name("plantid-vol", create_if_missing=True)
V = "/v"
IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "open_clip_torch==2.29.0", "timm==1.0.20",
                    "huggingface_hub==0.34.4", "numpy==1.26.4", "pandas==2.2.3", "pyarrow==17.0.0",
                    "pillow==10.4.0", "scikit-learn==1.5.2")
       .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}))

CFG = {
    "meta_csv": "pc24_meta.csv",                 # PlantCLEF2024 single-plant metadata (cached on volume)
    "bio_species": "bio_species.json",
    "organs": ["flower", "leaf", "bark", "fruit", "habit"],
    "pool_min_obs": 20, "n_species": 300, "n_train": 20, "n_test": 8,
    "smoke": {"n_species": 5, "n_train": 4, "n_test": 2},
    "fetch_threads": 4, "shard": {"full": 500, "smoke": 20},
    "fetch_min_success": 0.995, "cell_min_frac": 0.9,
    "models": {"bioclip": ["oc", "hf-hub:imageomics/bioclip", None], "dinov3_b": ["timm", "vit_base_patch16_dinov3.lvd1689m", None],
               "bioclip2": ["oc", "hf-hub:imageomics/bioclip-2", None], "dinov3_l": ["timm", "vit_large_patch16_dinov3.lvd1689m", None]},
    "pairs": {"B": ["bioclip", "dinov3_b"], "L": ["bioclip2", "dinov3_l"]},
    "k_grid": {"full": [1, 3, 10, 20], "smoke": [1, 3]},
    "n_boot": {"full": 1000, "smoke": 100},
}

# ---------------------------------------------------------------- manifest utilities
def _h(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
def _fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]
def _dir(mode, stage): d = f"{V}/runs/{mode}/{stage}"; os.makedirs(d, exist_ok=True); return d
def _man_path(mode, stage): return f"{V}/runs/{mode}/{stage}/manifest.json"
def _read_man(mode, stage):
    p = _man_path(mode, stage)
    return json.load(open(p)) if os.path.exists(p) else None
def _src(fn):
    name = fn if isinstance(fn, str) else getattr(fn, "__name__", None) or fn.tag
    text = open(__file__).read(); i = text.index(f"\ndef {name}(")
    j = min([x for x in (text.find("\n@app", i + 1), text.find("\ndef ", i + 1), text.find("\n# ---", i + 1)) if x > 0] + [len(text)])
    return text[i:j]
def _key(stage_fn, mode, inputs):
    return {"code": _h(_src(stage_fn)), "cfg": _h(CFG), "mode": mode, "inputs": inputs}
def _valid(mode, stage, key):
    m = _read_man(mode, stage)
    if not m or m.get("status") != "PASS" or m.get("key") != key: return False
    return all(os.path.exists(o["path"]) and _fsha(o["path"]) == o["sha"] for o in m["outputs"])
def _write_man(mode, stage, key, outputs, gates, extra=None):
    status = "PASS" if all(g["ok"] for g in gates) else "FAIL"
    m = {"stage": stage, "key": key, "status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "outputs": [{"path": p, "sha": _fsha(p)} for p in outputs], "gates": gates, "extra": extra or {}}
    p = _man_path(mode, stage); tmp = p + ".tmp"; json.dump(m, open(tmp, "w"), indent=1, default=str); os.replace(tmp, p); vol.commit()
    if status != "PASS": raise RuntimeError(f"{stage} GATE FAIL: " + "; ".join(g["name"] + "=" + str(g["value"]) for g in gates if not g["ok"]))
    return m
def _gate(name, value, ok): return {"name": name, "value": value, "ok": bool(ok)}
def _in(mode, stage): m = _read_man(mode, stage); assert m and m["status"] == "PASS", f"upstream {stage} not PASS"; return m["key"]["code"] + ":" + ",".join(o["sha"] for o in m["outputs"])

# ---------------------------------------------------------------- S0 environment
@app.function(image=IMG, volumes={V: vol}, gpu="A10G", timeout=900)
def s0_env(mode):
    vol.reload()
    import torch, open_clip, timm, numpy, pandas, PIL, sys
    key = _key("s0_env", mode, {})
    if _valid(mode, "s0", key): return "SKIP"
    vers = {"python": sys.version.split()[0], "torch": torch.__version__, "open_clip": open_clip.__version__, "timm": timm.__version__,
            "numpy": numpy.__version__, "pandas": pandas.__version__, "pillow": PIL.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    src = [f"{V}/{CFG['meta_csv']}", f"{V}/{CFG['bio_species']}"]
    out = f"{_dir(mode,'s0')}/env.json"; json.dump({"versions": vers, "sources": {p: _fsha(p) for p in src}}, open(out, "w"), indent=1)
    gates = [_gate("cuda", vers["cuda"], vers["cuda"]), _gate("sources_exist", [os.path.exists(p) for p in src], all(os.path.exists(p) for p in src))]
    _write_man(mode, "s0", key, [out], gates, vers); return "PASS"

# ---------------------------------------------------------------- S1 selection (author-disjoint, one image per obs)
@app.function(image=IMG, volumes={V: vol}, timeout=3600, memory=32768, cpu=4)
def s1_select(mode):
    vol.reload()
    import pandas as pd, numpy as np
    key = _key("s1_select", mode, {"s0": _in(mode, "s0")})
    if _valid(mode, "s1", key): return "SKIP"
    m = pd.read_csv(f"{V}/{CFG['meta_csv']}", sep=';', usecols=['image_name', 'organ', 'species_id', 'obs_id', 'author', 'species', 'dataset', 'url'], dtype=str)
    m = m[m.organ.isin(CFG["organs"])].copy(); m['author'] = m.author.fillna('NA')
    m['bin'] = m.species.str.extract(r'^([A-Z][a-z]+ [a-z\-]+)')[0]
    pn = m[m.dataset == 'plantnet'].drop_duplicates('obs_id')
    c = pn.groupby(['species_id', 'organ']).size().unstack(fill_value=0)
    for o in CFG["organs"]: c[o] = c[o] if o in c else 0
    t = CFG["pool_min_obs"]
    pool = c[(c.flower >= t) & (c.leaf >= t) & (c.bark >= t) & ((c.fruit >= t) | (c.habit >= t))].index
    NTR, NTE = CFG["n_train"], CFG["n_test"]
    def split_species(g):
        orgs = [o for o in CFG["organs"] if (g.organ == o).sum() >= NTR + NTE]
        if not {'flower', 'leaf', 'bark'} <= set(orgs) or not ({'fruit', 'habit'} & set(orgs)): return None
        au = g.author.value_counts().sort_values(); au = au.sample(frac=1, random_state=0).sort_values(kind='stable')
        test = set()
        for a in au.index:
            need = [o for o in orgs if ((g.organ == o) & g.author.isin(test)).sum() < NTE]
            if not need: break
            if ((g.organ.isin(need)) & (g.author == a)).any(): test.add(a)
        ok = [o for o in orgs if ((g.organ == o) & g.author.isin(test)).sum() >= NTE and ((g.organ == o) & ~g.author.isin(test)).sum() >= NTR]
        if not ({'flower', 'leaf', 'bark'} <= set(ok) and ({'fruit', 'habit'} & set(ok))): return None
        return test, ok
    bio = set(json.load(open(f"{V}/{CFG['bio_species']}")))
    feas = {}
    for sid, g in pn[pn.species_id.isin(pool)].groupby('species_id'):
        r = split_species(g)
        if r: feas[sid] = r
    bins = pn.drop_duplicates('species_id').set_index('species_id').bin
    fl = list(feas); ov = [s for s in fl if bins[s] in bio]; rest = [s for s in fl if s not in set(ov)]
    np.random.default_rng(0).shuffle(rest); chosen = (ov + rest)[:CFG["n_species"]]
    ntr, nte = NTR, NTE
    if mode == "smoke": chosen = chosen[:CFG["smoke"]["n_species"]]; ntr, nte = CFG["smoke"]["n_train"], CFG["smoke"]["n_test"]
    rows = []
    for sid in chosen:
        test, ok = feas[sid]; g = pn[pn.species_id == sid]
        for o in ok:
            rows += [g[(g.organ == o) & g.author.isin(test)].sample(NTE, random_state=0).head(nte).assign(split='test'),
                     g[(g.organ == o) & ~g.author.isin(test)].sample(NTR, random_state=0).head(ntr).assign(split='train')]
    sel = pd.concat(rows).reset_index(drop=True)
    sel['dl'] = sel.url.str.replace('/image/o/', '/image/m/')
    sel = sel[['image_name', 'species_id', 'species', 'organ', 'split', 'author', 'obs_id', 'dl']].sort_values(['species_id', 'organ', 'split', 'image_name']).reset_index(drop=True)
    cell = sel.groupby(['species_id', 'organ', 'split']).size()
    leak = sum(len(set(g[g.split == 'train'].author) & set(g[g.split == 'test'].author)) for _, g in sel.groupby('species_id'))
    exp = {"train": ntr, "test": nte}
    gates = [_gate("n_species", int(sel.species_id.nunique()), sel.species_id.nunique() == len(chosen) and len(chosen) >= (5 if mode == "smoke" else 250)),
             _gate("cell_sizes_exact", int((cell != [exp[s] for s in cell.index.get_level_values('split')]).sum()), all(cell.values == [exp[s] for s in cell.index.get_level_values('split')])),
             _gate("author_leak", leak, leak == 0), _gate("dup_image", int(sel.image_name.duplicated().sum()), not sel.image_name.duplicated().any()),
             _gate("dup_obs", int(sel.obs_id.duplicated().sum()), not sel.obs_id.duplicated().any())]
    out = f"{_dir(mode,'s1')}/sel.parquet"; sel.to_parquet(out, index=False)
    _write_man(mode, "s1", key, [out], gates, {"pool": len(pool), "feasible": len(feas), "n_img": len(sel), "bio_overlap": len([s for s in chosen if s in set(ov)])})
    return "PASS"

# ---------------------------------------------------------------- S2 fetch (resumable shards; reuses legacy cache by image_name after re-validation)
@app.function(image=IMG, volumes={V: vol}, timeout=6 * 3600, memory=8192, cpu=2)
def s2_fetch(mode, crash_after_shards: int = -1):
    vol.reload()
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, io, glob, urllib.request, urllib.error, random
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    key = _key("s2_fetch", mode, {"s1": _in(mode, "s1")})
    if _valid(mode, "s2", key): return "SKIP"
    d = _dir(mode, "s2"); sel = pd.read_parquet(f"{_dir(mode,'s1')}/sel.parquet")
    names = sel.image_name.tolist(); dl = dict(zip(sel.image_name, sel.dl)); S = CFG["shard"][mode]
    legacy = {}
    if os.path.exists(f"{V}/pc_sel.parquet"):                     # legacy cache: idx -> image_name
        old = pd.read_parquet(f"{V}/pc_sel.parquet"); want = set(names)
        idx2name = {i: n for i, n in zip(old.index, old.image_name) if n in want}
        for f in sorted(glob.glob(f"{V}/pcimg/*.parquet")):
            t = pq.read_table(f).to_pydict()
            for i, b, s in zip(t['idx'], t['bytes'], t['short']):
                if i in idx2name: legacy[idx2name[i]] = (b, s)
    def norm(raw):
        im = Image.open(io.BytesIO(raw)).convert('RGB'); w, h = im.size; im.thumbnail((800, 800)); o = io.BytesIO(); im.save(o, 'JPEG', quality=90); return o.getvalue(), min(w, h)
    def get(n):
        if n in legacy:
            b, s = legacy[n]; Image.open(io.BytesIO(b)).convert('RGB'); return n, b, s, "cache"
        for t in range(6):
            try:
                raw = urllib.request.urlopen(urllib.request.Request(dl[n], headers={'User-Agent': 'research-client'}), timeout=30).read()
                b, s = norm(raw); return n, b, s, "net"
            except urllib.error.HTTPError as e:
                if e.code == 404: return n, None, 0, "404"
                time.sleep(min(60, 3 * 2 ** t) + random.random())
            except Exception: time.sleep(2 * 2 ** t)
        return n, None, 0, "fail"
    stats = {"cache": 0, "net": 0, "404": 0, "fail": 0}; written = 0
    for si in range(0, len(names), S):
        p = f"{d}/shard_{si // S:05d}.parquet"
        if os.path.exists(p):                                    # resume: shard already complete
            t = pq.read_table(p, columns=['image_name', 'status']).to_pydict()
            for s in t['status']: stats[s] += 1
            continue
        chunk = names[si:si + S]
        with ThreadPoolExecutor(CFG["fetch_threads"]) as ex: res = list(ex.map(get, chunk))
        tbl = pa.table({'image_name': [r[0] for r in res], 'bytes': [r[1] for r in res], 'short': [r[2] for r in res], 'status': [r[3] for r in res]})
        pq.write_table(tbl, p + ".tmp"); os.replace(p + ".tmp", p); vol.commit()
        for r in res: stats[r[3]] += 1
        written += 1
        if crash_after_shards >= 0 and written > crash_after_shards: os._exit(1)   # fault injection: hard crash, no cleanup
    shards = sorted(glob.glob(f"{d}/shard_*.parquet"))
    ok_names = set()
    for p in shards:
        t = pq.read_table(p, columns=['image_name', 'bytes']).to_pydict()
        ok_names |= {n for n, b in zip(t['image_name'], t['bytes']) if b is not None}
    succ = len(ok_names) / len(names)
    sel['ok'] = sel.image_name.isin(ok_names)
    exp = {"train": CFG["n_train"] if mode == "full" else CFG["smoke"]["n_train"], "test": CFG["n_test"] if mode == "full" else CFG["smoke"]["n_test"]}
    cf = sel.groupby(['species_id', 'organ', 'split']).ok.sum().reset_index(); cf['min'] = cf.split.map(exp) * CFG["cell_min_frac"]
    bad_cells = int((cf.ok < cf['min']).sum())
    gates = [_gate("success_rate", round(succ, 5), succ >= CFG["fetch_min_success"]), _gate("cells_below_90pct", bad_cells, bad_cells == 0),
             _gate("all_names_covered", len(shards) * S >= len(names), len(shards) == -(-len(names) // S))]
    _write_man(mode, "s2", key, shards, gates, stats); return "PASS"

# ---------------------------------------------------------------- S3 features (fp32, per-model manifests)
@app.function(image=IMG, volumes={V: vol}, gpu="A10G", timeout=4 * 3600, memory=32768, cpu=4)
def s3_features(mode, model, sample_n: int = 0, tag: str = ""):
    vol.reload()
    import torch, numpy as np, pandas as pd, io, glob, pyarrow.parquet as pq
    from PIL import Image
    torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    stage = f"s3_{model}{tag}"
    key = _key("s3_features", mode, {"s2": _in(mode, "s2"), "model": model, "sample_n": sample_n})
    if _valid(mode, stage, key): return "SKIP"
    kind, mid, pt = CFG["models"][model]; dev = "cuda"
    if kind == "oc":
        import open_clip
        net, _, pre = open_clip.create_model_and_transforms(mid, pretrained=pt) if pt else open_clip.create_model_and_transforms(mid)
        blocks = net.visual.transformer.resblocks; npre = 1
    else:
        import timm
        net = timm.create_model(mid, pretrained=True, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=net); cfg['input_size'] = (3, 224, 224); pre = timm.data.create_transform(**cfg)
        blocks = net.blocks; npre = net.num_prefix_tokens
    net = net.to(dev).eval().float()
    L = len(blocks); idx = sorted(set([L // 4 - 1, L // 2 - 1, 3 * L // 4 - 1, L - 1])); acts = {}
    for i in idx: blocks[i].register_forward_hook(lambda m, inp, out, i=i: acts.__setitem__(i, out))
    B = {}
    for f in sorted(glob.glob(f"{_dir(mode,'s2')}/shard_*.parquet")):
        t = pq.read_table(f, columns=['image_name', 'bytes']).to_pydict(); B.update({n: b for n, b in zip(t['image_name'], t['bytes']) if b is not None})
    sel = pd.read_parquet(f"{_dir(mode,'s1')}/sel.parquet"); sel = sel[sel.image_name.isin(B)].reset_index(drop=True)
    if sample_n: sel = sel.iloc[:sample_n]
    names = sel.image_name.tolist()
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(names)
        def __getitem__(s, k): return pre(Image.open(io.BytesIO(B[names[k]])).convert("RGB"))
    feats = {}
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(DS(), batch_size=64, num_workers=4, shuffle=False):
            x = x.to(dev)
            fin = net.encode_image(x) if kind == "oc" else net.forward_head(net.forward_features(x), pre_logits=True)
            rec = {"final": fin}
            for i in idx:
                a = acts[i]
                if a.shape[0] != x.shape[0]: a = a.permute(1, 0, 2)
                rec[f"L{i+1}_cls"] = a[:, 0]; rec[f"L{i+1}_pm"] = a[:, npre:].mean(1)
            for k, v in rec.items(): feats.setdefault(k, []).append(v.float().cpu().numpy())
    feats = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    finite = all(np.isfinite(v).all() for v in feats.values())
    stds = {k: float(v.std(0).mean()) for k, v in feats.items()}
    # mini decodability probe: within-organ (flower) train->test, nearest-class-mean on final feature
    acc, chance = None, None
    fz = feats["final"]
    if not sample_n:
        m = sel.reset_index(drop=True); fl = (m.organ == 'flower').values
        sp = sorted(m.species_id.unique())[:20 if mode == "full" else 5]
        msk = fl & m.species_id.isin(sp).values; tr = msk & (m.split == 'train').values; te = msk & (m.split == 'test').values
        z = fz / np.linalg.norm(fz, axis=1, keepdims=True)
        mu = np.stack([z[tr & (m.species_id == s).values].mean(0) for s in sp])
        acc = float((np.array(sp)[(z[te] @ mu.T).argmax(1)] == m.species_id.values[te]).mean()); chance = 1 / len(sp)
    out = f"{_dir(mode, stage)}/feats.npz"
    np.savez(out, names=np.array(names), **feats)
    gates = [_gate("finite", finite, finite), _gate("nonzero_std", min(stds.values()), min(stds.values()) > 1e-6),
             _gate("n_rows", len(names), len(names) == len(sel))]
    if acc is not None:
        fac = 3 if mode == "full" else 2
        gates.append(_gate(f"mini_probe_acc_vs_{fac}x_chance", [round(acc, 3), round(fac * chance, 3)], acc >= fac * chance))
    _write_man(mode, stage, key, [out], gates, {"layers": [i + 1 for i in idx], "dim": int(fz.shape[1]), "std_final": stds["final"]}); return "PASS"

# ---------------------------------------------------------------- S4 probes (P1 + M1 round 1), deterministic
@app.function(image=IMG, volumes={V: vol}, gpu="T4", timeout=4 * 3600, memory=32768)
def s4_probes(mode, pair, tag: str = ""):
    vol.reload()
    import torch, numpy as np, pandas as pd
    torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    A, D = CFG["pairs"][pair]; stage = f"s4_{pair}{tag}"
    key = _key("s4_probes", mode, {"A": _in(mode, f"s3_{A}"), "D": _in(mode, f"s3_{D}"), "s1": _in(mode, "s1"), "lr": _h(_src("lr_fit"))})
    if _valid(mode, stage, key): return "SKIP"
    def load(n):
        Z = np.load(f"{_dir(mode, 's3_'+n)}/feats.npz"); Ls = sorted({k for k in Z.files if k.endswith('_cls')}, key=lambda s: int(s[1:].split('_')[0]))
        return Z['names'], Z['final'], Z[Ls[-2]]
    na, fa, aa = load(A); nd, fd, ad = load(D)
    assert (na == nd).all(), "row order mismatch between models"
    sel = pd.read_parquet(f"{_dir(mode,'s1')}/sel.parquet").set_index('image_name').loc[na].reset_index()
    M = sel.organ.isin(['flower', 'leaf']).values; m = sel[M].reset_index(drop=True)
    F = {'A': fa[M], 'D': fd[M], 'AD': np.hstack([fa[M], fd[M]]), 'Anull': np.hstack([fa[M], aa[M]]), 'Dnull': np.hstack([fd[M], ad[M]])}
    tr = (m.split == 'train').values; y = m.species_id.values; corr = {}
    for k in CFG["k_grid"][mode]:
        nmax = CFG["n_train"] if mode == "full" else CFG["smoke"]["n_train"]
        for seed in (range(3) if k < nmax else [0]):
            for o1 in ['flower', 'leaf']:
                pool = m[tr & (m.organ == o1).values]
                idx = np.sort(pool.groupby('species_id').sample(n=k, random_state=seed).index.values) if k < nmax else pool.index.values
                lams = (1e-2,) if k < 10 else (1e-4, 1e-3, 1e-2)
                for o2 in ['flower', 'leaf']:
                    te = np.where(~tr & (m.organ == o2).values)[0]; P = {}
                    for f, X in F.items():
                        cl, pr = lr_fit(X[idx], y[idx], X[te], lams=lams, seed=seed)
                        P[f] = pr; corr.setdefault((k, o1, o2, f), []).append(cl[pr.argmax(1)] == y[te])
                    corr.setdefault((k, o1, o2, 'late'), []).append(cl[(P['A'] + P['D']).argmax(1)] == y[te])
    rng_seed = 0; NB = CFG["n_boot"][mode]
    def bd(c1, c2):
        r = np.random.default_rng(rng_seed); d = [c1[i].mean() - c2[i].mean() for i in (r.integers(0, len(c1), len(c1)) for _ in range(NB))]
        return [float(c1.mean() - c2.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    rows = []
    for (k, o1, o2, f) in sorted(x for x in corr if x[3] == 'A'):
        g = lambda f: np.mean(corr[(k, o1, o2, f)], 0)
        a, d_, ad_, an, dn, lt = g('A'), g('D'), g('AD'), g('Anull'), g('Dnull'), g('late'); best = a if a.mean() >= d_.mean() else d_
        ca = np.array(corr[(k, o1, o2, 'A')]); cd = np.array(corr[(k, o1, o2, 'D')])
        rows.append(dict(pair=pair, k=k, train=o1, test=o2, within=o1 == o2, accA=float(a.mean()), accD=float(d_.mean()), accAD=float(ad_.mean()),
                         d_concat=bd(ad_, best), d_late=bd(lt, best), null_A=bd(an, a), null_D=bd(dn, d_), weak=bool(d_.mean() < a.mean() - 0.15),
                         only_one=float((ca ^ cd).mean()), oracle=float((ca | cd).mean()), n_test=int(len(a))))
    chance = 1 / m.species_id.nunique()
    out = f"{_dir(mode, stage)}/rows.json"; json.dump(rows, open(out, "w"), indent=1, sort_keys=True)
    accs = [r[x] for r in rows for x in ('accA', 'accD')]
    kmax = max(r['k'] for r in rows); fac = 3 if mode == "full" else 2
    easy = [r[x] for r in rows if r['within'] and r['k'] == kmax for x in ('accA', 'accD')]
    gates = [_gate("all_finite", bool(np.isfinite(accs).all()), np.isfinite(accs).all()),
             _gate(f"easy_setting_acc_vs_{fac}x_chance", [round(min(easy), 4), round(fac * chance, 4)], min(easy) > fac * chance + 1e-9)]
    _write_man(mode, stage, key, [out], gates, {"n_rows": len(rows)}); return "PASS"

def lr_fit(Xtr, ytr, Xte, lams=(1e-4, 1e-3, 1e-2), iters=150, seed=0):
    """standardised multinomial logistic regression (full-batch L-BFGS); lambda picked on a seeded 20% holdout. Returns (classes, probs)."""
    import torch, numpy as np
    dev = "cuda"; Xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev); Xte = torch.tensor(Xte, dtype=torch.float32, device=dev)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    cls, yt = np.unique(ytr, return_inverse=True); yt = torch.tensor(yt, device=dev); C = len(cls)
    def fit(X, y, lam):
        W = torch.zeros(X.shape[1], C, device=dev, requires_grad=True); b = torch.zeros(C, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
        def cl():
            opt.zero_grad(); l = torch.nn.functional.cross_entropy(X @ W + b, y) + lam * (W ** 2).sum(); l.backward(); return l
        opt.step(cl); return W.detach(), b.detach()
    lam = lams[0]
    if len(lams) > 1:
        p = torch.randperm(len(yt), generator=torch.Generator().manual_seed(seed)).to(dev); nv = max(1, len(p) // 5); va, tr = p[:nv], p[nv:]; best = None
        for l_ in lams:
            W, b = fit(Xtr[tr], yt[tr], l_); a = ((Xtr[va] @ W + b).argmax(1) == yt[va]).float().mean().item()
            if best is None or a > best[0]: best = (a, l_)
        lam = best[1]
    W, b = fit(Xtr, yt, lam); return cls, torch.softmax(Xte @ W + b, 1).cpu().numpy()

# ---------------------------------------------------------------- orchestration (runs remotely; survives local disconnect)
@app.function(image=IMG, volumes={V: vol}, timeout=20 * 3600)
def orchestrate(mode, upto="s4", crash_after_shards: int = -1):
    log = []
    def step(name, fn, *a, **kw):
        r = fn.remote(*a, **kw); log.append(f"{name}:{r}"); print(name, r, flush=True); vol.reload()
    step("s0", s0_env, mode)
    if upto == "s0": return log
    step("s1", s1_select, mode)
    if upto == "s1": return log
    step("s2", s2_fetch, mode, crash_after_shards)
    if upto == "s2": return log
    for mdl in CFG["models"]: step(f"s3_{mdl}", s3_features, mode, mdl)
    if upto == "s3": return log
    for p in CFG["pairs"]: step(f"s4_{p}", s4_probes, mode, p)
    return log

@app.function(image=IMG, volumes={V: vol}, timeout=600)
def status(mode):
    vol.reload(); out = []
    root = f"{V}/runs/{mode}"
    for st in sorted(os.listdir(root)) if os.path.exists(root) else []:
        m = _read_man(mode, st)
        if m: out.append(f"{st:16s} {m['status']:4s} {m['time']}  " + " | ".join(f"{g['name']}={g['value']}" for g in m['gates']))
        else: out.append(f"{st:16s} INCOMPLETE")
    return "\n".join(out)

@app.local_entrypoint()
def main(mode: str = "smoke", upto: str = "s4", crash_after_shards: int = -1, show: int = 0):
    if show: print(status.remote(mode)); return
    print(orchestrate.remote(mode, upto, crash_after_shards))
