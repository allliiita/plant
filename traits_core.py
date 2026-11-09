"""Pure, offline-testable core of the trait pipeline (no Modal, no network).
traits_pipeline.py wraps these with Modal I/O; tests/test_traits_core.py exercises them with fakes."""
import os, json, glob, numpy as np, pandas as pd

TRAITS = ["Height", "LeafArea", "SLA", "Leaf_N"]
LABELS = [f"{t}_median" for t in TRAITS]

# ------------------------------------------------------------------ selection
def select_species(d_all, gm, cfg, mode="full"):
    """d_all: PlantTraitNet rows (Species, Image, *_median). gm: GBIF match table. Returns (sel, info)."""
    d = d_all.copy(); d["pid"] = d.Image.str.extract(r"photo_(\d+)")[0]
    lab = d.groupby("Species")[LABELS].first()                       # first non-null per species
    n_nopid = int(d.pid.isna().sum()); d = d[d.pid.notna()].drop_duplicates(["Species", "pid"])
    multi = d.groupby("pid").Species.nunique(); shared = set(multi[multi > 1].index)
    links = d[d.pid.isin(shared)].groupby("pid").Species.apply(lambda x: sorted(set(x))).tolist()
    d = d[~d.pid.isin(shared)]
    n = d.groupby("Species").size(); pool = sorted(n[n >= cfg["min_photos"]].index)
    g = gm[gm.Species.isin(pool)].copy()
    ok = (g["rank"] == "SPECIES") & (g.confidence >= cfg["conf_min"]) & g.genus.notna() & g.family.notna() & g.order.notna()
    g = g[ok]; dup = g.speciesKey.duplicated(keep=False); n_coll = int(dup.sum()); g = g[~dup]
    g = g[g.Species.isin(lab.dropna().index)]
    sp = sorted(g.Species); parent = {x: x for x in sp}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for L in links:
        L = [x for x in L if x in parent]
        for x in L[1:]: parent[find(x)] = find(L[0])
    link_group = {x: find(x) for x in sp}
    if mode != "full": sp = [sp[i] for i in np.random.default_rng(0).permutation(len(sp))[:cfg["subset_species"][mode]]]
    rows = []
    for s in sp:
        x = d[d.Species == s].sort_values("pid")
        c = x.iloc[np.random.default_rng(0).permutation(len(x))][["Species", "pid"]].copy(); c["rank"] = np.arange(len(c)); rows.append(c)
    sel = pd.concat(rows).merge(g[["Species", "genus", "family", "order", "speciesKey"]], on="Species").merge(lab, left_on="Species", right_index=True)
    sel["link_group"] = sel.Species.map(link_group)
    sel = sel.sort_values(["Species", "rank"]).reset_index(drop=True)
    gs = pd.Series(link_group).value_counts()
    info = {"pool": len(pool), "after_tax_filter": int(ok.sum()), "synonym_collisions_dropped": n_coll, "n_species": int(sel.Species.nunique()),
            "rows_without_inat_pid": n_nopid, "shared_pids_dropped": len(shared), "species_in_linked_groups": int(gs[gs > 1].sum()),
            "genera": int(sel.genus.nunique()), "families": int(sel.family.nunique()), "orders": int(sel.order.nunique())}
    return sel, info

# ------------------------------------------------------------------ fetch (shards on a directory; getter injected)
def _read(p):
    import pyarrow.parquet as pq
    return pq.read_table(p).to_pydict()
def _write(p, t):
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.table(t), p + ".tmp"); os.replace(p + ".tmp", p)

def run_tier(dd, name, pids, getter, cfg, commit=lambda: None, map_fn=None, crash_after_shards=None):
    """map_fn(getter, list)->list (thread pool in production). Each shard holds pid/bytes/status; failed entries are retried up to cfg['retry_rounds']."""
    map_fn = map_fn or (lambda f, xs: [f(x) for x in xs])
    S = cfg["shard"]; nsh = -(-len(pids) // S); written = 0
    for si in range(nsh):
        p = f"{dd}/{name}_shard_{si:05d}.parquet"; chunk = pids[si * S:(si + 1) * S]
        for _ in range(cfg["retry_rounds"] + 1):
            if os.path.exists(p):
                t = _read(p)
                if t["pid"] != chunk or "status" not in t: os.remove(p); continue
                bad = [i for i, s in enumerate(t["status"]) if s == "failed"]
                if not bad: break
                rr = map_fn(getter, [chunk[i] for i in bad])
                for i, r in zip(bad, rr): t["bytes"][i], t["status"][i] = r[1], r[2]
            else:
                res = map_fn(getter, chunk); t = {"pid": [r[0] for r in res], "bytes": [r[1] for r in res], "status": [r[2] for r in res]}
            _write(p, t); commit(); written += 1
            if crash_after_shards is not None and written >= crash_after_shards: raise KeyboardInterrupt("injected crash")
    for p in glob.glob(f"{dd}/{name}_shard_*.parquet"):
        if int(p[-13:-8]) >= nsh: os.remove(p)

def statuses(dd, prefix="tier"):
    st = {}
    for p in sorted(glob.glob(f"{dd}/{prefix}*_shard_*.parquet")):
        t = _read(p); st.update(zip(t["pid"], t["status"]))
    return st

def fetch(sel, dd, getter, cfg, commit=lambda: None, map_fn=None, crash_after_shards=None):
    """Tier 1: ranks < n_photos. Tier 2: all remaining photos of species short after tier 1 (decided from tier-1 statuses only)."""
    import pyarrow.parquet as pq
    os.makedirs(dd, exist_ok=True)
    for p in glob.glob(f"{dd}/*.parquet"):
        if "status" not in pq.read_schema(p).names: os.remove(p)
    N = cfg["n_photos"]
    run_tier(dd, "tier1", sel[sel["rank"] < N].pid.tolist(), getter, cfg, commit, map_fn, crash_after_shards)
    st1 = statuses(dd, "tier1"); n_fail1 = sum(v == "failed" for v in st1.values())
    n_ok = sel[sel.pid.map(st1).eq("ok")].groupby("Species").size().reindex(sel.Species.unique(), fill_value=0)
    short = sorted(n_ok[n_ok < N].index)
    run_tier(dd, "tier2", sel[(sel["rank"] >= N) & sel.Species.isin(short)].pid.tolist(), getter, cfg, commit, map_fn)
    st = statuses(dd); s2 = sel.assign(status=sel.pid.map(st)); tried = s2[s2.status.notna()]
    per = s2[s2.status == "ok"].groupby("Species").size().reindex(sel.Species.unique(), fill_value=0).clip(upper=N)
    excl = per[per < cfg["min_photos_after_fetch"]].index
    lab = sel.drop_duplicates("Species").set_index("Species")
    t1 = s2[s2["rank"] < N].groupby("Species").status.apply(lambda x: float((x == "deleted").mean()))
    from scipy.stats import spearmanr
    info = {"n_failed": int((tried.status == "failed").sum()), "tier1_failed_before_retry": n_fail1, "dead_rate": float((tried.status == "deleted").mean()),
            "n_tried": int(len(tried)), "frac_species_lt30": float((per < N).mean()), "n_excluded_lt20": int(len(excl)), "excluded_frac": float(len(excl) / len(per)),
            "min_photos_any_species": int(per.min()),
            "excluded_vs_retained": {t: {"excluded_mean_log": float(np.log(lab.loc[excl, f"{t}_median"]).mean()) if len(excl) else None,
                                         "retained_mean_log": float(np.log(lab.drop(index=excl)[f"{t}_median"]).mean())} for t in TRAITS},
            "dead_vs_trait_spearman": {t: float(spearmanr(t1.values, np.log(lab.loc[t1.index, f"{t}_median"].values)).correlation) for t in TRAITS}}
    return info

def final_photos(sel, dd, cfg):
    """first n_photos ok photos by rank; drop species with < min_photos_after_fetch; tier-1 dead rate per kept species."""
    st = statuses(dd); st1 = statuses(dd, "tier1")
    s = sel[sel.pid.map(st).eq("ok")].sort_values(["Species", "rank"]).groupby("Species").head(cfg["n_photos"])
    cnt = s.groupby("Species").size(); keep = sorted(cnt[cnt >= cfg["min_photos_after_fetch"]].index)
    s = s[s.Species.isin(keep)].reset_index(drop=True)
    t1 = sel[sel["rank"] < cfg["n_photos"]].assign(st=lambda x: x.pid.map(st1))
    dead = t1.groupby("Species").st.apply(lambda x: float((x == "deleted").mean())).reindex(keep)
    return s, dead, int((cnt < cfg["min_photos_after_fetch"]).sum())

# ------------------------------------------------------------------ analysis
def ridge_fit(X, Y, alphas):
    xm, ym = X.mean(0), Y.mean(0); Xc, Yc = X - xm, Y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); UtY = U.T @ Yc; out = {}
    for a in alphas:
        W = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY); out[a] = (W, ym - xm @ W)
    return out

def make_folds(tax, sp, level, seed=0):
    """taxonomic level merged with link groups; species that shared photos never split."""
    lab = tax[level].values if level != "Species" else np.array(sp); par = {}
    def f_(x):
        while par.setdefault(x, x) != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a_, b_ in zip(["L:" + str(x) for x in lab], ["G:" + str(x) for x in tax.link_group]): par[f_(a_)] = f_(b_)
    g = np.array([f_("L:" + str(x)) for x in lab]); u = np.array(sorted(set(g)))
    perm = u[np.random.default_rng(seed).permutation(len(u))]; m = {x: i % 5 for i, x in enumerate(perm)}
    return np.array([m[x] for x in g])

def impute(Y, tax, tr_mask):
    """genus -> family -> order (each needs >=2 TRAINING species) -> global mean of training species. Uses only rows in tr_mask."""
    pred = np.full(Y.shape, np.nan); used = np.array(["global"] * len(Y), dtype=object)
    for lv in ("genus", "family", "order"):
        gg = pd.DataFrame(Y[tr_mask], index=tax[lv].values[tr_mask]).groupby(level=0); mm = gg.mean(); mm = mm[gg.size() >= 2]
        v = mm.reindex(tax[lv].values).values; fill = np.isnan(pred[:, 0]) & ~np.isnan(v[:, 0])
        pred[fill] = v[fill]; used[fill] = lv
    g = np.isnan(pred[:, 0]); pred[g] = Y[tr_mask].mean(0) if tr_mask.sum() else 0.0; return pred, used

def analyze(pid, X, pl, ph, dead_species, dead_rate, sel, cfg, n_boot):
    assert len(set(pid)) == len(pid), "duplicate pids in features"
    X = X.astype(np.float64)
    sel = sel.set_index("pid").loc[pid].reset_index()
    sp = sorted(sel.Species.unique()); si = sel.Species.map({s: i for i, s in enumerate(sp)}).values
    tax = sel.drop_duplicates("Species").set_index("Species").loc[sp]
    Y = np.log(tax[LABELS].values); Yimg = Y[si]; A = cfg["alphas"]
    def species_mean(P, w=None):
        w = np.ones(len(P)) if w is None else w
        num = np.zeros((len(sp), P.shape[1])); den = np.zeros(len(sp)); np.add.at(num, si, P * w[:, None]); np.add.at(den, si, w)
        return num / np.maximum(den, 1e-12)[:, None]
    def fit_predict(trS, teS, level, target, Zsp=None):
        """Zsp: optional species-level covariates appended to image features (stacking with imputation)."""
        tri = trS[si]; tei = teS[si]; XX = X if Zsp is None else np.hstack([X, Zsp[si]])
        mu, sd = XX[tri].mean(0), XX[tri].std(0) + 1e-6; Xs = (XX - mu) / sd
        inner = np.where(trS, make_folds(tax, sp, level, seed=1), -1); err = {a: 0.0 for a in A}
        for f in range(5):
            itr = trS & (inner != f); ite = trS & (inner == f)
            if ite.sum() == 0 or itr.sum() == 0: continue
            mi = itr[si]; mt = ite[si]; W = ridge_fit(Xs[mi], target[si[mi]], A)
            for a, (w, b) in W.items():
                pim = np.zeros((len(Xs), target.shape[1])); pim[mt] = Xs[mt] @ w + b
                err[a] += ((species_mean(pim, mt.astype(float))[ite] - target[ite]) ** 2).sum()
        a_best = min(err, key=err.get); w, b = ridge_fit(Xs[tri], target[si[tri]], [a_best])[a_best]
        pim = np.zeros((len(Xs), target.shape[1])); pim[tei] = Xs[tei] @ w + b; return pim, a_best
    res = {"n_species": len(sp), "n_images": int(len(X))}; store = {}
    for level in ("Species", "genus", "family", "order"):
        fo = make_folds(tax, sp, level); P_imp = np.zeros_like(Y); used = np.empty(len(sp), dtype=object)
        P_img = np.zeros((len(X), Y.shape[1])); P_inc = np.zeros_like(P_img); alphas = []
        for f in range(5):
            te = fo == f; tr = ~te
            pi, u = impute(Y, tax, tr); P_imp[te] = pi[te]; used[te] = u[te]
            pim, a = fit_predict(tr, te, level, Y); P_img += pim; alphas.append(a)
            # incremental (stacking): image features + imputation as covariate; training species get inner-fold imputation (never their own label)
            inner = make_folds(tax, sp, level, seed=1); Zsp = np.zeros_like(Y); Zsp[te] = pi[te]
            for k in range(5):
                ik = tr & (inner == k)
                if ik.sum() == 0: continue
                if (tr & (inner != k)).sum() == 0: Zsp[ik] = 0.0; continue          # no inner-training species: covariate carries no information
                pk, _ = impute(Y, tax, tr & (inner != k)); Zsp[ik] = pk[ik]
            pr, _ = fit_predict(tr, te, level, Y, Zsp); P_inc += pr
        S = {"imp": P_imp, "img": species_mean(P_img), "inc": species_mean(P_inc), "used": used, "alphas": alphas,
             "img_wleaf": species_mean(P_img, pl), "img_whabit": species_mean(P_img, ph)}
        for nm, pr_ in (("img_top_leaf", pl), ("img_top_habit", ph)):
            w = np.zeros(len(X))
            for s_ in range(len(sp)):
                idx = np.where(si == s_)[0]; w[idx[np.argsort(-pr_[idx], kind="stable")[:10]]] = 1.0
            S[nm] = species_mean(P_img, w)
        store[level] = S
    gen = tax.genus.values; ug = np.array(sorted(set(gen))); gidx = np.searchsorted(ug, gen)
    rng = np.random.default_rng(0); BW = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(n_boot)]
    def r2(p, m, j): y = Y[m, j]; return float(1 - ((y - p[m, j]) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    def boot_diff(pa, pb, m, j):
        y = Y[:, j]; ea = (y - pa[:, j]) ** 2; eb = (y - pb[:, j]) ** 2; out = []
        for w in BW:
            ws = w[gidx] * m
            if ws.sum() == 0: continue
            my = (ws * y).sum() / ws.sum(); out.append(((ws * eb).sum() - (ws * ea).sum()) / (ws * (y - my) ** 2).sum())
        out = np.array(out); return [r2(pa, m, j) - r2(pb, m, j), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())]
    for level, S in store.items():
        R = {}; strata = {"pooled": np.ones(len(sp), bool)}
        if level == "Species": strata.update({"genus_imputed": S["used"] == "genus", "family_imputed": S["used"] == "family", "order_or_global": np.isin(S["used"], ["order", "global"])})
        for sn, m in strata.items():
            if m.sum() < 10: continue
            for j, t in enumerate(TRAITS):
                e = {"n": int(m.sum()), "r2_imp": r2(S["imp"], m, j), "r2_img": r2(S["img"], m, j), "r2_inc": r2(S["inc"], m, j),
                     "d_img_minus_imp": boot_diff(S["img"], S["imp"], m, j), "d_inc_minus_imp": boot_diff(S["inc"], S["imp"], m, j)}
                if sn in ("pooled", "genus_imputed"):
                    wk = "img_whabit" if t == "Height" else "img_wleaf"; tk = "img_top_habit" if t == "Height" else "img_top_leaf"
                    e["d_weighted_minus_unweighted"] = boot_diff(S[wk], S["img"], m, j); e["d_top10_minus_unweighted"] = boot_diff(S[tk], S["img"], m, j)
                R[f"{sn}|{t}"] = e
        R["imputation_level_counts"] = {k: int(v) for k, v in pd.Series(S["used"]).value_counts().items()}; R["alphas"] = S["alphas"]; res[level] = R
    # warm-up (seen species): nearest-centroid lookup vs direct regression, image level
    rng2 = np.random.default_rng(0); ff = np.zeros(len(X), int)
    for s_ in range(len(sp)):
        idx = np.where(si == s_)[0]; ff[idx] = rng2.permutation(len(idx)) % 5
    Pl, Pr = np.zeros((len(X), Y.shape[1])), np.zeros((len(X), Y.shape[1]))
    for f in range(5):
        tr = ff != f; te = ~tr; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
        C = np.zeros((len(sp), X.shape[1])); np.add.at(C, si[tr], Xs[tr]); C /= np.maximum(np.bincount(si[tr], minlength=len(sp)), 1)[:, None]
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12); Q = Xs[te] / (np.linalg.norm(Xs[te], axis=1, keepdims=True) + 1e-12)
        Pl[te] = Y[(Q @ Cn.T).argmax(1)]; w, b = ridge_fit(Xs[tr], Yimg[tr], [100.0])[100.0]; Pr[te] = Xs[te] @ w + b
    sst = ((Yimg - Yimg.mean(0)) ** 2).sum(0)
    res["warmup_seen"] = {t: {"r2_lookup_img": float(1 - ((Yimg[:, j] - Pl[:, j]) ** 2).sum() / sst[j]), "r2_regress_img": float(1 - ((Yimg[:, j] - Pr[:, j]) ** 2).sum() / sst[j])} for j, t in enumerate(TRAITS)}
    # selection-bias sensitivity
    dr = pd.Series(dead_rate, index=dead_species).reindex(sp).values; S0 = store["Species"]; gmask = S0["used"] == "genus"; low = dr <= np.nanmedian(dr)
    res["sensitivity_low_dead"] = {t: {"n": int((gmask & low).sum()), "d_img_minus_imp": boot_diff(S0["img"], S0["imp"], gmask & low, j)} for j, t in enumerate(TRAITS)}
    from scipy.stats import spearmanr
    res["dead_vs_trait_spearman_full"] = {t: float(spearmanr(dr, Y[:, j]).correlation) for j, t in enumerate(TRAITS)}
    # internal Holm (used only to trigger checks; reporting uses ΔR², CI, MDE)
    pv = {t: res["Species"][f"genus_imputed|{t}"]["d_img_minus_imp"][3] for t in TRAITS if f"genus_imputed|{t}" in res["Species"]}
    verdict, stop = {}, False
    for i, t in enumerate(sorted(pv, key=lambda k: (pv[k], k))):
        thr = 0.05 / (len(pv) - i); sig = (not stop) and pv[t] <= thr; stop = stop or not sig
        verdict[t] = {"p_boot": pv[t], "holm_threshold": thr, "triggers_checks": bool(sig)}
    if verdict.get("Leaf_N", {}).get("triggers_checks"): verdict["Leaf_N"]["flag"] = "anomalous_inspect"
    if verdict.get("Height", {}).get("triggers_checks"): verdict["Height"]["flag"] = "mandatory_leakage_check_before_interpretation"
    res["internal_checks"] = verdict
    return res, store, tax, sp
