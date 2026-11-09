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
    "frozen_split": {"path": "pc_sel.parquet", "sha": "56d73f6bb69b898c"},
    "k_grid": {"full": [1, 3, 10, 20], "smoke": [1, 3]},
    "n_boot": {"full": 1000, "smoke": 100},
}

# ---------------------------------------------------------------- manifest utilities
def base(mode): return "full" if mode.startswith("full") else "smoke"
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
def _select(mode):
    """pure, deterministic selection; must not depend on hash order"""
    import pandas as pd, numpy as np
    m = pd.read_csv(f"{V}/{CFG['meta_csv']}", sep=';', usecols=['image_name', 'organ', 'species_id', 'obs_id', 'author', 'species', 'dataset', 'url'], dtype=str)
    m = m[m.organ.isin(CFG["organs"])].copy(); m['author'] = m.author.fillna('NA')
    m['bin'] = m.species.str.extract(r'^([A-Z][a-z]+ [a-z\-]+)')[0]
    m = m.sort_values('image_name', kind='stable')
    pn = m[m.dataset == 'plantnet'].drop_duplicates('obs_id')
    c = pn.groupby(['species_id', 'organ']).size().unstack(fill_value=0)
    for o in CFG["organs"]: c[o] = c[o] if o in c else 0
    t = CFG["pool_min_obs"]
    pool = sorted(c[(c.flower >= t) & (c.leaf >= t) & (c.bark >= t) & ((c.fruit >= t) | (c.habit >= t))].index)
    NTR, NTE = CFG["n_train"], CFG["n_test"]
    def split_species(sid, g):
        orgs = [o for o in CFG["organs"] if (g.organ == o).sum() >= NTR + NTE]
        if not {'flower', 'leaf', 'bark'} <= set(orgs) or not ({'fruit', 'habit'} & set(orgs)): return None
        cnt = g.author.value_counts(); names = sorted(cnt.index)
        perm = np.random.default_rng(int(sid)).permutation(len(names))
        order = sorted([names[i] for i in perm], key=lambda a: cnt[a])        # stable: seeded shuffle, then by count
        test = set()
        for a in order:
            need = [o for o in orgs if ((g.organ == o) & g.author.isin(test)).sum() < NTE]
            if not need: break
            if ((g.organ.isin(need)) & (g.author == a)).any(): test.add(a)
        ok = [o for o in orgs if ((g.organ == o) & g.author.isin(test)).sum() >= NTE and ((g.organ == o) & ~g.author.isin(test)).sum() >= NTR]
        if not ({'flower', 'leaf', 'bark'} <= set(ok) and ({'fruit', 'habit'} & set(ok))): return None
        return test, ok
    bio = set(json.load(open(f"{V}/{CFG['bio_species']}")))
    feas = {}
    for sid, g in pn[pn.species_id.isin(pool)].groupby('species_id', sort=True):
        r = split_species(sid, g)
        if r: feas[sid] = r
    bins = pn.drop_duplicates('species_id').set_index('species_id').bin
    fl = sorted(feas); ov = [s for s in fl if bins[s] in bio]; rest = [s for s in fl if s not in set(ov)]
    rest = [rest[i] for i in np.random.default_rng(0).permutation(len(rest))]; chosen = (ov + rest)[:CFG["n_species"]]
    ntr, nte = NTR, NTE
    if base(mode) == "smoke": chosen = chosen[:CFG["smoke"]["n_species"]]; ntr, nte = CFG["smoke"]["n_train"], CFG["smoke"]["n_test"]
    rows = []
    for sid in chosen:
        test, ok = feas[sid]; g = pn[pn.species_id == sid]
        for o in ok:
            rows += [g[(g.organ == o) & g.author.isin(test)].sample(NTE, random_state=0).head(nte).assign(split='test'),
                     g[(g.organ == o) & ~g.author.isin(test)].sample(NTR, random_state=0).head(ntr).assign(split='train')]
    sel = pd.concat(rows)
    sel['dl'] = sel.url.str.replace('/image/o/', '/image/m/')
    sel = sel[['image_name', 'species_id', 'species', 'organ', 'split', 'author', 'obs_id', 'dl']].sort_values(['species_id', 'organ', 'split', 'image_name']).reset_index(drop=True)
    return sel, {"pool": len(pool), "feasible": len(feas), "n_img": len(sel), "bio_overlap": len([s for s in chosen if s in set(ov)]), "n_chosen": len(chosen), "ntr": ntr, "nte": nte}

def _df_hash(df): return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]

@app.function(image=IMG, volumes={V: vol}, timeout=3600, memory=32768, cpu=4)
def s1_select(mode):
    vol.reload()
    import subprocess, sys
    key = _key("s1_select", mode, {"s0": _in(mode, "s0"), "sel": _h(_src("_select"))})
    if _valid(mode, "s1", key): return "SKIP"
    frozen = CFG.get("frozen_split") if base(mode) == "full" else None
    if frozen:                                   # pinned split artifact: reproducibility by sha pin, validated by the same gates
        import pandas as pd
        fp = f"{V}/{frozen['path']}"; fsha = _fsha(fp)
        sel = pd.read_parquet(fp); sel = sel[sel.src == 'plantnet'][['image_name', 'species_id', 'species', 'organ', 'split', 'author', 'obs_id', 'dl']]
        sel = sel.sort_values(['species_id', 'organ', 'split', 'image_name']).reset_index(drop=True)
        info = {"n_chosen": int(sel.species_id.nunique()), "ntr": CFG["n_train"], "nte": CFG["n_test"], "n_img": len(sel), "frozen_sha": fsha}
        h1 = _df_hash(sel); h2 = [h1] if fsha == frozen["sha"] else ["SHA_MISMATCH:" + fsha]
    else:
        sel, info = _select(mode); h1 = _df_hash(sel)
        env = dict(os.environ, PYTHONHASHSEED="12345")
        code = f"import sys; sys.path.insert(0,'/root'); import pipeline as p; s,_=p._select({mode!r}); print(p._df_hash(s))"
        h2 = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd="/root").stdout.strip().splitlines()[-1:]
    cell = sel.groupby(['species_id', 'organ', 'split']).size()
    leak = sum(len(set(g[g.split == 'train'].author) & set(g[g.split == 'test'].author)) for _, g in sel.groupby('species_id'))
    exp = {"train": info["ntr"], "test": info["nte"]}; want = [exp[s] for s in cell.index.get_level_values('split')]
    gates = [_gate("n_species", info["n_chosen"], sel.species_id.nunique() == info["n_chosen"] and info["n_chosen"] >= (5 if base(mode) == "smoke" else 250)),
             _gate("cell_sizes_exact", int((cell.values != want).sum()), all(cell.values == want)),
             _gate("author_leak", leak, leak == 0), _gate("dup_image", int(sel.image_name.duplicated().sum()), not sel.image_name.duplicated().any()),
             _gate("dup_obs", int(sel.obs_id.duplicated().sum()), not sel.obs_id.duplicated().any()),
             _gate("frozen_sha_pin" if frozen else "hashseed_invariant", [h1, h2[0] if h2 else None], bool(h2) and h2[0] == h1)]
    out = f"{_dir(mode,'s1')}/sel.parquet"; sel.to_parquet(out, index=False)
    _write_man(mode, "s1", key, [out], gates, {**info, "sel_hash": h1}); return "PASS"

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
    names = sel.image_name.tolist(); dl = dict(zip(sel.image_name, sel.dl)); S = CFG["shard"][base(mode)]
    legacy = {}
    if os.path.exists(f"{V}/pc_sel.parquet") and not mode.endswith("_nocache"):                     # legacy cache: idx -> image_name
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
        chunk = names[si:si + S]
        if os.path.exists(p):                                    # resume only if shard belongs to THIS input slice
            t = pq.read_table(p, columns=['image_name', 'status']).to_pydict()
            if t['image_name'] == chunk:
                for s in t['status']: stats[s] += 1
                continue
            os.remove(p); stats.setdefault("stale_shards_rewritten", 0); stats["stale_shards_rewritten"] += 1
        with ThreadPoolExecutor(CFG["fetch_threads"]) as ex: res = list(ex.map(get, chunk))
        tbl = pa.table({'image_name': [r[0] for r in res], 'bytes': [r[1] for r in res], 'short': [r[2] for r in res], 'status': [r[3] for r in res]})
        pq.write_table(tbl, p + ".tmp"); os.replace(p + ".tmp", p); vol.commit()
        for r in res: stats[r[3]] += 1
        written += 1
        if crash_after_shards >= 0 and written > crash_after_shards: os._exit(1)   # fault injection: hard crash, no cleanup
    nsh = -(-len(names) // S)
    for p in sorted(glob.glob(f"{d}/shard_*.parquet")):           # remove shards beyond this input's range
        if int(p[-13:-8]) >= nsh: os.remove(p)
    shards = sorted(glob.glob(f"{d}/shard_*.parquet"))
    ok_names = set()
    for p in shards:
        t = pq.read_table(p, columns=['image_name', 'bytes']).to_pydict()
        ok_names |= {n for n, b in zip(t['image_name'], t['bytes']) if b is not None}
    succ = len(ok_names) / len(names)
    sel['ok'] = sel.image_name.isin(ok_names)
    exp = {"train": CFG["n_train"] if base(mode) == "full" else CFG["smoke"]["n_train"], "test": CFG["n_test"] if base(mode) == "full" else CFG["smoke"]["n_test"]}
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
        sp = sorted(m.species_id.unique())[:20 if base(mode) == "full" else 5]
        msk = fl & m.species_id.isin(sp).values; tr = msk & (m.split == 'train').values; te = msk & (m.split == 'test').values
        z = fz / np.linalg.norm(fz, axis=1, keepdims=True)
        mu = np.stack([z[tr & (m.species_id == s).values].mean(0) for s in sp])
        acc = float((np.array(sp)[(z[te] @ mu.T).argmax(1)] == m.species_id.values[te]).mean()); chance = 1 / len(sp)
    out = f"{_dir(mode, stage)}/feats.npz"
    np.savez(out, names=np.array(names), **feats)
    gates = [_gate("finite", finite, finite), _gate("nonzero_std", min(stds.values()), min(stds.values()) > 1e-6),
             _gate("n_rows", len(names), len(names) == len(sel))]
    if acc is not None:
        fac = 3 if base(mode) == "full" else 2
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
    for k in CFG["k_grid"][base(mode)]:
        nmax = CFG["n_train"] if base(mode) == "full" else CFG["smoke"]["n_train"]
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
    rng_seed = 0; NB = CFG["n_boot"][base(mode)]
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
    kmax = max(r['k'] for r in rows); fac = 3 if base(mode) == "full" else 2
    easy = [r[x] for r in rows if r['within'] and r['k'] == kmax for x in ('accA', 'accD')]
    gates = [_gate("all_finite", bool(np.isfinite(accs).all()), np.isfinite(accs).all()),
             _gate(f"easy_setting_acc_vs_{fac}x_chance", [round(min(easy), 4), round(fac * chance, 4)], min(easy) > fac * chance + 1e-9)]
    _write_man(mode, stage, key, [out], gates, {"n_rows": len(rows)}); return "PASS"

@app.function(image=IMG, volumes={V: vol}, gpu="T4", timeout=6 * 3600, memory=32768)
def s5_b2(mode, pair, tag: str = ""):
    """B2: same protocol as s4_probes on rare organs; test restricted to species seen in the training organ."""
    vol.reload()
    import torch, numpy as np, pandas as pd
    torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    A, D = CFG["pairs"][pair]; stage = f"s5_b2_{pair}{tag}"; ORG = ['bark', 'fruit', 'habit']
    key = _key("s5_b2", mode, {"A": _in(mode, f"s3_{A}"), "D": _in(mode, f"s3_{D}"), "s1": _in(mode, "s1"), "lr": _h(_src("lr_fit"))})
    if _valid(mode, stage, key): return "SKIP"
    def load(n):
        Z = np.load(f"{_dir(mode, 's3_'+n)}/feats.npz"); Ls = sorted({k for k in Z.files if k.endswith('_cls')}, key=lambda s: int(s[1:].split('_')[0]))
        return Z['names'], Z['final'], Z[Ls[-2]]
    na, fa, aa = load(A); nd, fd, ad = load(D)
    assert (na == nd).all(), "row order mismatch between models"
    sel = pd.read_parquet(f"{_dir(mode,'s1')}/sel.parquet").set_index('image_name').loc[na].reset_index()
    M = sel.organ.isin(ORG).values; m = sel[M].reset_index(drop=True)
    F = {'A': fa[M], 'D': fd[M], 'AD': np.hstack([fa[M], fd[M]]), 'Anull': np.hstack([fa[M], aa[M]]), 'Dnull': np.hstack([fd[M], ad[M]])}
    tr = (m.split == 'train').values; y = m.species_id.values; corr = {}
    nmax = CFG["n_train"] if base(mode) == "full" else CFG["smoke"]["n_train"]
    for k in CFG["k_grid"][base(mode)]:
        for seed in (range(3) if k < nmax else [0]):
            for o1 in ORG:
                pool = m[tr & (m.organ == o1).values]
                if len(pool) == 0: continue
                idx = np.sort(pool.groupby('species_id').sample(n=k, random_state=seed).index.values) if k < nmax else pool.index.values
                lams = (1e-2,) if k < 10 else (1e-4, 1e-3, 1e-2); seen = set(pool.species_id)
                for o2 in ORG:
                    te = np.where(~tr & (m.organ == o2).values & m.species_id.isin(seen).values)[0]
                    if len(te) == 0: continue
                    P = {}
                    for f, X in F.items():
                        cl, pr = lr_fit(X[idx], y[idx], X[te], lams=lams, seed=seed)
                        P[f] = pr; corr.setdefault((k, o1, o2, f), []).append(cl[pr.argmax(1)] == y[te])
                    corr.setdefault((k, o1, o2, 'late'), []).append(cl[(P['A'] + P['D']).argmax(1)] == y[te])
    NB = CFG["n_boot"][base(mode)]
    def bd(c1, c2):
        r = np.random.default_rng(0); d = [c1[i].mean() - c2[i].mean() for i in (r.integers(0, len(c1), len(c1)) for _ in range(NB))]
        return [float(c1.mean() - c2.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    rows = []
    for (k, o1, o2, f) in sorted(x for x in corr if x[3] == 'A'):
        g = lambda f: np.mean(corr[(k, o1, o2, f)], 0)
        a, d_, ad_, an, dn, lt = g('A'), g('D'), g('AD'), g('Anull'), g('Dnull'), g('late'); best = a if a.mean() >= d_.mean() else d_
        ca = np.array(corr[(k, o1, o2, 'A')]); cd = np.array(corr[(k, o1, o2, 'D')])
        rows.append(dict(pair=pair, k=k, train=o1, test=o2, within=o1 == o2, accA=float(a.mean()), accD=float(d_.mean()), accAD=float(ad_.mean()),
                         d_concat=bd(ad_, best), d_late=bd(lt, best), null_A=bd(an, a), null_D=bd(dn, d_), weak=bool(d_.mean() < a.mean() - 0.15),
                         only_one=float((ca ^ cd).mean()), oracle=float((ca | cd).mean()), n_test=int(len(a)), n_species=int(len(set(y[tr & (m.organ == o1).values])))))
    out = f"{_dir(mode, stage)}/rows.json"; json.dump(rows, open(out, "w"), indent=1, sort_keys=True)
    accs = [r[x] for r in rows for x in ('accA', 'accD')]; kmax = max(r['k'] for r in rows); fac = 3 if base(mode) == "full" else 2
    easy = [(r[x], 1 / r['n_species']) for r in rows if r['within'] and r['k'] == kmax for x in ('accA', 'accD')]
    gates = [_gate("all_finite", bool(np.isfinite(accs).all()), np.isfinite(accs).all()),
             _gate(f"easy_setting_acc_vs_{fac}x_chance", [round(min(a for a, _ in easy), 4)], all(a > fac * c for a, c in easy))]
    _write_man(mode, stage, key, [out], gates, {"n_rows": len(rows)}); return "PASS"

def _strat_split(species, fam):
    """deterministic family-stratified SEEN/UNSEEN split + stratified VAL from SEEN"""
    import numpy as np
    sp = sorted(species); fams = {}
    cnt = {}
    for x in sp: cnt[fam[x]] = cnt.get(fam[x], 0) + 1
    for x in sp: fams.setdefault(fam[x] if cnt[fam[x]] > 1 else "_other", []).append(x)
    seen, unseen, order = [], [], {}
    for f in sorted(fams):
        g = sorted(fams[f]); g = [g[i] for i in np.random.default_rng(0).permutation(len(g))]
        for j, x in enumerate(g):
            order[x] = j
            if j < len(g) - len(g) % 2: (seen if j % 2 == 0 else unseen).append(x)
            else: (seen if len(seen) <= len(unseen) else unseen).append(x)
    sk = sorted(seen, key=lambda x: (fam[x], order[x]))
    val = sk[::5][:max(1, len(seen) // 5)]
    return seen, unseen, val

@app.function(image=IMG, volumes={V: vol}, gpu="T4", timeout=6 * 3600, memory=32768)
def s6_adapter(mode, model, smoke: int = 0, tag: str = ""):
    vol.reload()
    import torch, numpy as np, pandas as pd, itertools
    torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    stage = f"s6_{model}{'_smoke' if smoke else ''}{tag}"
    key = _key("s6_adapter", mode, {"f": _in(mode, f"s3_{model}"), "s1": _in(mode, "s1"), "smoke": smoke, "split": _h(_src("_strat_split")), "lr": _h(_src("lr_fit"))})
    if _valid(mode, stage, key): return "SKIP"
    dev = "cuda"; ORG = CFG["organs"]
    with np.load(f"{_dir(mode, 's3_'+model)}/feats.npz") as Z: names, X = Z['names'], Z['final'].astype(np.float64)
    sel = pd.read_parquet(f"{_dir(mode,'s1')}/sel.parquet").set_index('image_name').loc[names].reset_index()
    fm = pd.read_csv(f"{V}/{CFG['meta_csv']}", sep=';', usecols=['species_id', 'family', 'genus'], dtype=str).drop_duplicates('species_id').set_index('species_id')
    allsp = sorted(sel.species_id.unique())
    if smoke: allsp = allsp[:20]
    fam = fm.family.to_dict(); gen = fm.genus.to_dict()
    seen, unseen, val = _strat_split(allsp, fam); fit = sorted(set(seen) - set(val))
    tvd = 0.5 * sum(abs(pd.Series([fam[x] for x in seen]).value_counts(normalize=True).get(f, 0) - pd.Series([fam[x] for x in unseen]).value_counts(normalize=True).get(f, 0)) for f in set(fam[x] for x in allsp))
    tr = (sel.split == 'train').values; org = sel.organ.values; sp = sel.species_id.values
    S_tr = np.isin(sp, seen) & tr                      # everything fit on SEEN train images only
    mu, sd = X[S_tr].mean(0), X[S_tr].std(0) + 1e-6; Xs = (X - mu) / sd
    # ---- training-free transforms
    om = {o: Xs[S_tr & (org == o)].mean(0) for o in ORG}
    M1 = Xs - np.stack([om[o] for o in org])
    Xc = Xs[S_tr] - Xs[S_tr].mean(0); C = Xc.T @ Xc / len(Xc); C += 1e-3 * np.trace(C) / len(C) * np.eye(len(C))
    ev, Ve = np.linalg.eigh(C); W = Ve @ np.diag(ev ** -.5) @ Ve.T; Wp = Ve @ np.diag(ev ** .5) @ Ve.T
    Zo = np.stack([(org[S_tr] == o).astype(float) for o in ORG], 1); Zo -= Zo.mean(0)
    Q, Rr, _ = np.linalg.svd(W @ (Xc.T @ Zo / len(Xc)), full_matrices=False); Q = Q[:, Rr > 1e-8 * Rr.max()]
    M2 = Xs - (Xs - Xs[S_tr].mean(0)) @ W @ Q @ Q.T @ Wp
    # ---- adapters
    def train_adapter(cross, lr, epochs, temp, mlp, seed=0):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        d = Xs.shape[1]
        net = (torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256)) if mlp else torch.nn.Linear(d, 256)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        idx_fit = np.where(np.isin(sp, fit) & tr)[0]; by_sp = {s: idx_fit[sp[idx_fit] == s] for s in fit}
        Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
        P_, K_ = min(64, len(fit)), 8; steps = max(1, len(idx_fit) // (P_ * K_))
        for ep in range(epochs):
            for _ in range(steps):
                ss = rng.choice(fit, P_, replace=False)
                b = np.concatenate([rng.choice(by_sp[s], K_, replace=len(by_sp[s]) < K_) for s in ss])
                z = torch.nn.functional.normalize(net(Xt[b]), dim=1)
                sim = z @ z.T / temp; eye = torch.eye(len(b), device=dev, dtype=torch.bool)
                sim = sim.masked_fill(eye, -1e9); logp = sim - torch.logsumexp(sim, 1, keepdim=True)
                ys = torch.tensor(pd.factorize(sp[b])[0], device=dev); os_ = torch.tensor(pd.factorize(org[b])[0], device=dev)
                same = (ys[:, None] == ys[None]) & ~eye
                def sup(pos):
                    n = pos.sum(1); ok = n > 0
                    return -((logp * pos).sum(1)[ok] / n[ok]).mean()
                loss = sup(same) if not cross else 0.5 * sup(same & (os_[:, None] != os_[None])) + 0.5 * sup(same)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): return net(Xt).cpu().numpy().astype(np.float64)
    IX = {kk: np.sort(v) for kk, v in sel.groupby(['species_id', 'organ', 'split']).indices.items()}
    def evaluate(Zf, species, ks, seeds_lt, clf="proto", pairs=None):
        """returns dict pair -> (per-species correct sums, counts) averaged over seeds, per k"""
        Zn = Zf / np.linalg.norm(Zf, axis=1, keepdims=True); out = {}; spset = set(species)
        for k in ks:
            seeds = range(seeds_lt) if k < CFG["n_train"] else [0]
            for A, B in (pairs or itertools.product(ORG, ORG)):
                cls = sorted(c for c in spset if (c, A, 'train') in IX and (c, B, 'test') in IX)
                if len(cls) < 2: continue
                q = np.concatenate([IX[(c, B, 'test')] for c in cls]); acc = np.zeros(len(q))
                for sd_ in seeds:
                    r = np.random.default_rng(sd_)
                    sup_idx = [r.choice(IX[(c, A, 'train')], min(k, len(IX[(c, A, 'train')])), replace=False) for c in cls]
                    if clf == "proto":
                        Pm = np.stack([Zn[i].mean(0) for i in sup_idx]); Pm /= np.linalg.norm(Pm, axis=1, keepdims=True)
                        pred = np.array(cls)[(Zn[q] @ Pm.T).argmax(1)]
                    else:
                        si = np.concatenate(sup_idx); cl, pr = lr_fit(Zn[si], sp[si], Zn[q], lams=(1e-2,), seed=sd_); pred = cl[pr.argmax(1)]
                    acc += (pred == sp[q]) / len(seeds)
                df = pd.DataFrame({'s': sp[q], 'c': acc}).groupby('s').c.agg(['sum', 'count'])
                out[(k, A, B)] = df
        return out
    def macro(res, k, cross=True):
        v = [df['sum'].sum() / df['count'].sum() for (kk, A, B), df in res.items() if kk == k and ((A != B) == cross)]
        return float(np.mean(v)) if v else float('nan')
    # ---- hyper-parameter selection on VAL species (cross-organ, k=5, proto)
    grid = [(lr, ep, t) for lr in (1e-3, 3e-3) for ep in (50, 200) for t in (0.05, 0.1)]
    if smoke: grid = [(1e-3, 5, 0.1)]
    xpairs = [(A, B) for A in ORG for B in ORG if A != B]
    sel_log = {}; Zs = {"M0": Xs, "M1": M1, "M2": M2}
    for name, cross, mlp in [("M3", True, False), ("M3a", False, False), ("M3_mlp", True, True), ("M3a_mlp", False, True)]:
        best = None
        for hp in grid:
            Zh = train_adapter(cross, *hp, mlp)
            v = macro(evaluate(Zh, val, [5], 3, pairs=xpairs), 5)
            sel_log[f"{name}|{hp}"] = v
            if best is None or v > best[0]: best = (v, hp, Zh)
        Zs[name] = best[2]; sel_log[f"{name}|chosen"] = [best[1], best[0]]
    # ---- evaluation on UNSEEN (and genus-disjoint subset)
    ks = [1, 5, 20]; NB = 1000 if not smoke else 50
    seen_gen = {gen[x] for x in seen}; gdis = [x for x in unseen if gen[x] not in seen_gen]
    results = {"split": {"n_seen": len(seen), "n_unseen": len(unseen), "n_val": len(val), "family_tvd": tvd, "n_genus_disjoint": len(gdis)}, "hp": sel_log}
    for subset_name, subset in [("unseen", unseen)] + ([("genus_disjoint", gdis)] if len(gdis) >= 40 else []):
        R = {(m, "proto"): evaluate(Zs[m], subset, ks, 5, "proto") for m in Zs}
        if subset_name == "unseen":                      # secondary classifier: k=5, core methods only (cost control, pre-declared as secondary)
            for m in ("M0", "M1", "M3", "M3a"): R[(m, "lr")] = evaluate(Zs[m], subset, [5], 5, "lr")
        spl = sorted(subset); rng = np.random.default_rng(0); W_ = [np.bincount(rng.integers(0, len(spl), len(spl)), minlength=len(spl)) for _ in range(NB)]
        def boot(res, k, cross):
            keys = [kk for kk in res if kk[0] == k and ((kk[1] != kk[2]) == cross)]
            Sm = np.zeros((len(spl), len(keys))); Cn = np.zeros_like(Sm); pos = {s: i for i, s in enumerate(spl)}
            for j, kk in enumerate(keys):
                df = res[kk]; ii = [pos[s_] for s_ in df.index]
                Sm[ii, j] = df['sum'].values; Cn[ii, j] = df['count'].values
            pt = np.mean(Sm.sum(0) / Cn.sum(0)); bs = np.array([np.mean((w @ Sm) / np.maximum(w @ Cn, 1e-9)) for w in W_])
            return pt, bs
        out = {}
        for (m, clf), res in R.items():
            for k in sorted({kk[0] for kk in res}):
                for cross in (True, False):
                    pt, bs = boot(res, k, cross); out[(m, clf, k, cross)] = (pt, bs)
        summ = {}
        for (m, clf, k, cross), (pt, bs) in out.items():
            summ[f"{m}|{clf}|k{k}|{'cross' if cross else 'within'}"] = [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        for a, b in [("M3", "M1"), ("M3", "M3a"), ("M3", "M0"), ("M3", "M2"), ("M1", "M0"), ("M2", "M0"), ("M3a", "M1"), ("M3_mlp", "M1"), ("M3_mlp", "M3a_mlp")]:
            for clf in (["proto", "lr"] if subset_name == "unseen" else ["proto"]):
                for k in ks:
                    for cross in (True, False):
                        if (a, clf, k, cross) not in out or (b, clf, k, cross) not in out: continue
                        d = out[(a, clf, k, cross)][1] - out[(b, clf, k, cross)][1]; pt = out[(a, clf, k, cross)][0] - out[(b, clf, k, cross)][0]
                        summ[f"{a}-{b}|{clf}|k{k}|{'cross' if cross else 'within'}"] = [pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        # per target organ (k=5 proto cross)
        for m in ("M3", "M1"):
            res = R[(m, "proto")]
            for B in ORG:
                v = [df['sum'].sum() / df['count'].sum() for (kk, A, BB), df in res.items() if kk == 5 and BB == B and A != B]
                summ[f"{m}|proto|k5|target_{B}"] = float(np.mean(v))
        results[subset_name] = summ
    out_p = f"{_dir(mode, stage)}/results.json"; json.dump(results, open(out_p, "w"), indent=1, sort_keys=True)
    vals = [v[0] for v in results["unseen"].values() if isinstance(v, list)]
    m0w = results["unseen"]["M0|proto|k5|within"][0]; nchance = 1 / len(unseen)
    gates = [_gate("seen_unseen_disjoint", len(set(seen) & set(unseen)), not set(seen) & set(unseen)),
             _gate("val_fit_disjoint", len(set(val) & set(fit)), not set(val) & set(fit)),
             _gate("family_tvd<=0.10", round(tvd, 4), smoke or tvd <= 0.10),
             _gate("finite", bool(np.isfinite(vals).all()), np.isfinite(vals).all()),
             _gate("M0_within_k5_vs_3x_chance", [round(m0w, 4), round(3 * nchance, 4)], m0w > 3 * nchance)]
    _write_man(mode, stage, key, [out_p], gates, results["split"]); return "PASS"

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

@app.function(image=IMG, volumes={V: vol}, timeout=600)
def compare_s2(mode_a, mode_b):
    vol.reload(); import glob, pyarrow.parquet as pq
    def h(mode):
        out = {}
        for p in sorted(glob.glob(f"{_dir(mode,'s2')}/shard_*.parquet")):
            t = pq.read_table(p).to_pydict()
            for n, b, st in zip(t['image_name'], t['bytes'], t['status']): out[n] = (hashlib.sha256(b).hexdigest() if b else None, st)
        return out
    a, b = h(mode_a), h(mode_b)
    return {"n_a": len(a), "n_b": len(b), "same_keys": set(a) == set(b), "byte_identical": sum(a[k][0] == b.get(k, (None,))[0] for k in a),
            "status_a": {s: sum(v[1] == s for v in a.values()) for s in ("cache", "net", "404", "fail")}, "status_b": {s: sum(v[1] == s for v in b.values()) for s in ("cache", "net", "404", "fail")},
            "man_a": (_read_man(mode_a, "s2") or {}).get("status"), "man_b": (_read_man(mode_b, "s2") or {}).get("status")}

@app.function(image=IMG, volumes={V: vol}, timeout=6 * 3600)
def reliability(mode="full"):
    import numpy as np
    out = {}
    for p in CFG["pairs"]:                                    # R3: rerun S4 into a separate stage, compare bytes
        s4_probes.remote(mode, p, "_r3"); vol.reload()
        a, b = f"{_dir(mode, 's4_'+p)}/rows.json", f"{_dir(mode, 's4_'+p+'_r3')}/rows.json"
        out[f"R3_{p}_identical"] = _fsha(a) == _fsha(b)
    for mdl in ["bioclip", "dinov3_l"]:                      # R4: extract 500 imgs twice, compare to each other and to full run
        for t in ("_r4a", "_r4b"): s3_features.remote(mode, mdl, 500, t)
        vol.reload()
        def ld(p):
            with np.load(p) as z: return {k: z[k] for k in z.files}        # copy + close: open files block vol.reload()
        A = ld(f"{_dir(mode, 's3_'+mdl+'_r4a')}/feats.npz"); B = ld(f"{_dir(mode, 's3_'+mdl+'_r4b')}/feats.npz"); F = ld(f"{_dir(mode, 's3_'+mdl)}/feats.npz")
        out[f"R4_{mdl}_maxdiff_rerun"] = float(max(np.abs(A[k] - B[k]).max() for k in A if k != "names"))
        out[f"R4_{mdl}_maxdiff_vs_full"] = float(max(np.abs(A[k] - F[k][:500]).max() for k in A if k != "names"))
        out[f"R4_{mdl}_finite"] = bool(all(np.isfinite(A[k]).all() for k in A if k != "names"))
    json.dump(out, open(f"{V}/runs/{mode}/reliability.json", "w"), indent=1); vol.commit(); return out

@app.function(image=IMG, volumes={V: vol}, timeout=900, memory=16384)
def split_check(mode="full"):
    vol.reload(); import pandas as pd
    sel = pd.read_parquet(f"{_dir(mode,'s1')}/sel.parquet")
    fm = pd.read_csv(f"{V}/{CFG['meta_csv']}", sep=';', usecols=['species_id', 'family', 'genus'], dtype=str).drop_duplicates('species_id').set_index('species_id')
    allsp = sorted(sel.species_id.unique()); fam = fm.family.to_dict()
    seen, unseen, val = _strat_split(allsp, fam)
    fs = pd.Series([fam[x] for x in seen]).value_counts(normalize=True); fu = pd.Series([fam[x] for x in unseen]).value_counts(normalize=True)
    tvd = 0.5 * sum(abs(fs.get(f, 0) - fu.get(f, 0)) for f in set(fs.index) | set(fu.index))
    vc = pd.Series([fam[x] for x in allsp]).value_counts()
    st = lambda x: fam[x] if vc[fam[x]] > 1 else "_other"
    ss = pd.Series([st(x) for x in seen]).value_counts(normalize=True); su = pd.Series([st(x) for x in unseen]).value_counts(normalize=True)
    tvd_st = 0.5 * sum(abs(ss.get(f, 0) - su.get(f, 0)) for f in set(ss.index) | set(su.index))
    return {"tvd_strata": tvd_st, "n_fam": len(vc), "singleton_fams": int((vc == 1).sum()), "tvd": tvd, "n_seen": len(seen), "n_unseen": len(unseen)}

@app.local_entrypoint()
def main(mode: str = "smoke", upto: str = "s4", crash_after_shards: int = -1, show: int = 0, cmp: str = ""):
    if show: print(status.remote(mode)); return
    if cmp: print(json.dumps(compare_s2.remote(mode, cmp))); return
    if upto == "splitcheck": print("SPLIT", json.dumps(split_check.remote(mode))); return
    if upto.startswith("s6"):
        smoke = int(upto.endswith("smoke"))
        for mdl in (["dinov3_b", "bioclip"] if smoke else ["dinov3_b", "bioclip", "dinov3_l", "bioclip2"]):
            print("S6", mdl, s6_adapter.remote(mode, mdl, smoke), flush=True)
        return
    if upto == "b2":
        for p in CFG["pairs"]: print("B2", p, s5_b2.remote(mode, p))
        return
    if upto == "rel": print("REL", json.dumps(reliability.remote(mode))); return
    print(orchestrate.remote(mode, upto, crash_after_shards))
