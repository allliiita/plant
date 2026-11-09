"""Offline, deterministic tests for traits_core ($0). Run: python3 tests/test_traits_core.py
Each test takes the module under test, so the mutation check can run the same suite on a mutated copy."""
import sys, os, json, tempfile, types, hashlib, subprocess, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
CFG = {"min_photos": 30, "n_photos": 30, "conf_min": 90, "shard": 50, "retry_rounds": 3, "min_photos_after_fetch": 20,
       "alphas": [10 ** (x / 2) for x in range(-2, 9)], "subset_species": {"smoke": 4}}
T = ["Height", "LeafArea", "SLA", "Leaf_N"]

# ------------------------------------------------------------------ synthetic fixtures
def fake_csv():
    rows = []
    def add(sp, pids, lab):
        for p in pids: rows.append({"Species": sp, "Image": f"x/{sp}/obs_1_photo_{p}.jpg", **{f"{t}_median": v for t, v in zip(T, lab)}})
    add("Aa a", range(1000, 1040), [1, 2, 3, 4]); add("Aa b", range(2000, 2035), [2, 3, 4, 5])
    add("Bb c", range(3000, 3032), [1, 1, 1, 1]); add("Bb d", range(4000, 4033), [3, 3, 3, 3])
    add("Cc e", range(5000, 5031), [2, 2, 2, 2])                       # 31 photos; 2 shared below -> 29 -> excluded
    add("Cc f", list(range(6000, 6031)) + [5000, 5001], [4, 4, 4, 4])   # shares 2 photos with Cc e -> linked
    add("Dd g", range(7000, 7040), [5, 5, 5, 5]); add("Dd h", range(7500, 7540), [6, 6, 6, 6])   # synonym collision pair
    d = pd.DataFrame(rows)
    nan_first = pd.DataFrame([{"Species": "Bb c", "Image": "x/Bb c/abc-uuid.jpg", **{f"{t}_median": np.nan for t in T}}])
    d = pd.concat([nan_first, d], ignore_index=True)                 # NaN label row first + row without iNat pid
    gm = pd.DataFrame([{"Species": s, "speciesKey": k, "genus": s.split()[0], "family": "F" + s[0], "order": "O", "rank": "SPECIES", "confidence": 99}
                       for s, k in [("Aa a", 1), ("Aa b", 2), ("Bb c", 3), ("Bb d", 4), ("Cc e", 5), ("Cc f", 6), ("Dd g", 7), ("Dd h", 7)]])
    return d, gm

def synth_analysis(kind, n_genus=60, per_genus=5, imgs=8, seed=0):
    rng = np.random.default_rng(seed); rows = []; feats = []
    gemb = rng.normal(size=(n_genus, 24)); geff = rng.normal(0, 1.0, size=(n_genus, 4))
    for g in range(n_genus):
        for s in range(per_genus):
            y = geff[g] + rng.normal(0, 0.6, size=4); name = f"G{g} s{s}"
            for i in range(imgs):
                pid = f"{g}_{s}_{i}"
                rows.append({"pid": pid, "Species": name, "genus": f"G{g}", "family": f"F{g // 4}", "order": f"O{g // 20}", "link_group": name,
                             **{f"{t}_median": float(np.exp(v)) for t, v in zip(T, y)}})
                if kind == "taxonomy_only": x = np.r_[gemb[g] + rng.normal(0, 0.3, 24), rng.normal(size=8)]
                elif kind == "complementary": x = np.r_[(y - geff[g]) + rng.normal(0, 0.1, 4), rng.normal(size=28)]   # species deviation only, no genus info
                else: x = np.r_[y + rng.normal(0, 0.15, 4), gemb[g] + rng.normal(0, 0.3, 24), rng.normal(size=4)]
                feats.append(x)
    sel = pd.DataFrame(rows); X = np.array(feats); n = len(sel)
    sp = sorted(sel.Species.unique())
    return sel.pid.values, X, np.full(n, 0.5), np.full(n, 0.5), np.array(sp), np.zeros(len(sp)), sel

class FakeGetter:
    """deterministic: 'deleted' for pid%10==0 and for most photos of clustered species; 'failed' on first 2 calls for pid%7==3; one pid always fails if persistent."""
    def __init__(self, clustered=(), persistent=None): self.calls = {}; self.clustered = set(clustered); self.persistent = persistent
    def __call__(self, pid):
        c = self.calls[pid] = self.calls.get(pid, 0) + 1; p = int(pid)
        if self.persistent is not None and p == self.persistent: return pid, None, "failed"
        if p % 7 == 3 and c <= 2: return pid, None, "failed"
        if p % 10 == 0 or (p // 1000 in self.clustered and p % 3 != 0): return pid, None, "deleted"
        return pid, hashlib.sha256(pid.encode()).digest(), "ok"

def fetch_sel(n_sp=6, n_ph=45):
    rows = [{"Species": f"S{k}", "pid": str((k + 1) * 1000 + r), "rank": r, **{f"{t}_median": float(k + 1) for t in T}} for k in range(n_sp) for r in range(n_ph)]
    return pd.DataFrame(rows)

def snapshot(C, sel, dd):
    s, dead, nex = C.final_photos(sel, dd, CFG)
    return json.dumps({"st": C.statuses(dd), "final": s.pid.tolist(), "dead": dead.round(6).to_dict(), "nex": nex}, sort_keys=True)

# ------------------------------------------------------------------ tests
def t_select_rules(C):
    d, gm = fake_csv(); sel, info = C.select_species(d, gm, CFG)
    sps = set(sel.Species)
    assert "Dd g" not in sps and "Dd h" not in sps, "synonym collision not dropped"
    assert not {"5000", "5001"} & set(sel.pid), "shared photos kept"
    assert "Cc e" not in sps, "species below 30 after removing shared photos kept"
    assert sel.pid.notna().all() and sel.pid.is_unique, "NaN/duplicate pid in selection"
    assert sel[sel.Species == "Bb c"]["Height_median"].iloc[0] == 1, "label not first non-null"
    assert info["shared_pids_dropped"] == 2 and info["rows_without_inat_pid"] == 1
    assert (sel.groupby("Species")["rank"].apply(lambda r: list(r) == list(range(len(r))))).all(), "ranks not contiguous"

def t_select_hashseed(C):
    code = ("import sys,json;sys.path.insert(0,%r);sys.path.insert(0,%r);import test_traits_core as T,traits_core as C;d,g=T.fake_csv();s,_=C.select_species(d,g,T.CFG);print(s.to_csv(index=False))"
            % (ROOT, os.path.join(ROOT, "tests")))
    outs = {subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=h), capture_output=True, text=True).stdout for h in ("0", "12345")}
    assert len(outs) == 1 and len(next(iter(outs))) > 100, "selection depends on hash seed"

def t_fetch_resume_identical(C):
    sel = fetch_sel()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        C.fetch(sel, a, FakeGetter(clustered={2}), CFG); clean = snapshot(C, sel, a)
        g = FakeGetter(clustered={2})
        for k in (1, 2, 3):
            try: C.fetch(sel, b, g, CFG, crash_after_shards=k)
            except KeyboardInterrupt: pass
        C.fetch(sel, b, g, CFG)
        assert snapshot(C, sel, b) == clean, "resume after crashes differs from clean run"

def t_fetch_transient_retried(C):
    sel = fetch_sel()
    with tempfile.TemporaryDirectory() as a:
        info = C.fetch(sel, a, FakeGetter(), CFG)
        assert info["n_failed"] == 0, "transient failures not retried"
        assert info["tier1_failed_before_retry"] == 0
    with tempfile.TemporaryDirectory() as a:
        info = C.fetch(sel, a, FakeGetter(persistent=1005), CFG)
        assert info["n_failed"] == 1, "persistent failure not reported (gate would miss it)"

def t_fetch_stale_and_legacy(C):
    import pyarrow as pa, pyarrow.parquet as pq
    sel = fetch_sel()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        C.fetch(sel, a, FakeGetter(clustered={2}), CFG); clean = snapshot(C, sel, a)
        pq.write_table(pa.table({"pid": ["999"], "bytes": [b"x"]}), f"{b}/shard_00000.parquet")                 # legacy format
        stale = [str(3000 + r) for r in range(30, 45)] * 2                                                       # 30 'ok' entries for short species S2
        pq.write_table(pa.table({"pid": stale, "bytes": [b"x"] * len(stale), "status": ["ok"] * len(stale)}), f"{b}/tier2_shard_00007.parquet")  # stale tier2
        pq.write_table(pa.table({"pid": ["1"], "bytes": [b"x"], "status": ["ok"]}), f"{b}/tier2_shard_00099.parquet")
        C.fetch(sel, b, FakeGetter(clustered={2}), CFG)
        assert not os.path.exists(f"{b}/shard_00000.parquet"), "legacy shard kept"
        assert snapshot(C, sel, b) == clean, "stale shards changed the outcome"

def t_exclusion(C):
    sel = fetch_sel()
    with tempfile.TemporaryDirectory() as a:
        info = C.fetch(sel, a, FakeGetter(clustered={3}), CFG); s, dead, nex = C.final_photos(sel, a, CFG)
        cnt = s.groupby("Species").size()
        assert (cnt >= CFG["min_photos_after_fetch"]).all() and (cnt <= CFG["n_photos"]).all(), "photo count rules violated"
        assert "S2" not in set(s.Species) and nex == info["n_excluded_lt20"] == 1, "species with <20 photos not excluded"
        ranks = s.groupby("Species")["rank"].apply(lambda r: list(r) == sorted(r))
        assert ranks.all(), "not first-by-rank"

def t_impute_no_leak(C):
    pid, X, pl, ph, dsp, dr, sel = synth_analysis("taxonomy_only", n_genus=20)
    tax = sel.drop_duplicates("Species").set_index("Species"); Y = np.log(tax[[f"{t}_median" for t in T]].values)
    fo = C.make_folds(tax, list(tax.index), "Species"); te = fo == 0; tr = ~te
    p1, _ = C.impute(Y, tax, tr); Y2 = Y.copy(); Y2[te] = np.random.default_rng(1).normal(size=Y2[te].shape)
    p2, _ = C.impute(Y2, tax, tr)
    assert np.allclose(p1[te], p2[te]), "imputation of test species depends on test labels"

def t_folds_links(C):
    tax = pd.DataFrame({"genus": ["A", "B", "C", "D"] * 5, "family": ["F1", "F2"] * 10, "order": ["O"] * 20}, index=[f"s{i}" for i in range(20)])
    tax["link_group"] = tax.index; tax.loc["s1", "link_group"] = "s0"; tax.loc["s2", "link_group"] = "s0"   # s0,s1,s2 linked (genera A,B,C)
    for lv in ("Species", "genus", "family", "order"):
        f = C.make_folds(tax, list(tax.index), lv)
        assert f[0] == f[1] == f[2], f"linked species split at level {lv}"
        if lv == "genus":
            for g in "ABCD": assert len(set(f[(tax.genus == g).values])) == 1, "genus split"

def t_known_answer_taxonomy_only(C):
    res, *_ = C.analyze(*synth_analysis("taxonomy_only"), CFG, n_boot=200)
    for t in T:
        d = res["Species"][f"genus_imputed|{t}"]["d_img_minus_imp"]
        assert d[0] < 0.03 and d[2] < 0.06, f"taxonomy-only features beat genus imputation ({t}: {d})"
        di = res["Species"][f"genus_imputed|{t}"]["d_inc_minus_imp"]
        assert di[0] < 0.03, f"taxonomy-only features add beyond imputation ({t}: {di})"

def t_known_answer_signal(C):
    res, *_ = C.analyze(*synth_analysis("signal"), CFG, n_boot=200)
    for t in T:
        d = res["Species"][f"genus_imputed|{t}"]["d_img_minus_imp"]
        assert d[1] > 0.1, f"real trait signal not detected ({t}: {d})"
        di = res["Species"][f"genus_imputed|{t}"]["d_inc_minus_imp"]
        assert di[1] > 0.1, f"incremental model misses real signal ({t}: {di})"

def t_known_answer_complementary(C):
    res, *_ = C.analyze(*synth_analysis("complementary"), CFG, n_boot=200)
    for t in T:
        e = res["Species"][f"genus_imputed|{t}"]
        assert e["d_inc_minus_imp"][1] > 0.1, f"stacking misses complementary signal ({t}: {e['d_inc_minus_imp']})"
        assert e["r2_inc"] > e["r2_img"] + 0.1, f"stacking not better than image-only when images lack genus info ({t})"

def t_analysis_deterministic(C):
    a = json.dumps(C.analyze(*synth_analysis("signal", n_genus=30), CFG, n_boot=100)[0], sort_keys=True, default=str)
    b = json.dumps(C.analyze(*synth_analysis("signal", n_genus=30), CFG, n_boot=100)[0], sort_keys=True, default=str)
    assert a == b, "analysis not deterministic"

def t_duplicate_pid_rejected(C):
    pid, X, pl, ph, dsp, dr, sel = synth_analysis("signal", n_genus=12); pid = pid.copy(); pid[1] = pid[0]
    try: C.analyze(pid, X, pl, ph, dsp, dr, sel, CFG, n_boot=10)
    except AssertionError: return
    raise AssertionError("duplicate pids accepted")

TESTS = [t_select_rules, t_select_hashseed, t_fetch_resume_identical, t_fetch_transient_retried, t_fetch_stale_and_legacy, t_exclusion,
         t_impute_no_leak, t_folds_links, t_known_answer_taxonomy_only, t_known_answer_signal, t_known_answer_complementary, t_analysis_deterministic, t_duplicate_pid_rejected]

def run(C, tests=TESTS, quiet=False):
    fails = []
    for t in tests:
        try: t(C); (None if quiet else print("PASS", t.__name__))
        except Exception as e: fails.append((t.__name__, repr(e)[:160])); (None if quiet else print("FAIL", t.__name__, repr(e)[:160]))
    return fails

# ------------------------------------------------------------------ mutation check: reintroduce known bugs, the suite must fail
MUTATIONS = {
    "M1 keep shared photos": ('    d = d[~d.pid.isin(shared)]\n', '    d = d\n'),
    "M2 label = first row (NaN)": ('    lab = d.groupby("Species")[LABELS].first()', '    lab = d.drop_duplicates("Species").set_index("Species")[LABELS]'),
    "M3 keep rows without pid": ('d = d[d.pid.notna()].drop_duplicates(["Species", "pid"])', 'd = d.drop_duplicates(["Species", "pid"])'),
    "M4 no synonym-collision drop": ('g = g[~dup]', 'g = g'),
    "M5 imputation uses all species": ('gg = pd.DataFrame(Y[tr_mask], index=tax[lv].values[tr_mask])', 'gg = pd.DataFrame(Y, index=tax[lv].values)'),
    "M6 folds ignore link groups": ('par[f_(a_)] = f_(b_)', 'pass'),
    "M7 tier-2 decided from stale statuses": ('st1 = statuses(dd, "tier1"); n_fail1', 'st1 = statuses(dd); n_fail1'),
    "M8 failures never retried": ('bad = [i for i, s in enumerate(t["status"]) if s == "failed"]', 'bad = []'),
    "M9 legacy shards kept": ('if "status" not in pq.read_schema(p).names: os.remove(p)', 'pass'),
    "M10 no <20 exclusion": ('keep = sorted(cnt[cnt >= cfg["min_photos_after_fetch"]].index)', 'keep = sorted(cnt.index)'),
    "M11 duplicate pids accepted": ('assert len(set(pid)) == len(pid), "duplicate pids in features"', 'pass'),
    "M13 incremental without imputation covariate": ('pr, _ = fit_predict(tr, te, level, Y, Zsp); P_inc += pr', 'pr, _ = fit_predict(tr, te, level, Y); P_inc += pr'),
    "M12 transient treated as deleted": ('rr = map_fn(getter, [chunk[i] for i in bad])', 'rr = [(chunk[i], None, "deleted") for i in bad]'),
}

def mutation_check():
    src = open(os.path.join(ROOT, "traits_core.py")).read(); out = {}
    for name, (a, b) in MUTATIONS.items():
        assert src.count(a) == 1, f"mutation anchor not unique/absent: {name}"
        m = types.ModuleType("traits_core_mut"); exec(compile(src.replace(a, b), "traits_core_mut", "exec"), m.__dict__)
        tests = [t for t in TESTS if t is not t_select_hashseed]
        fails = run(m, tests, quiet=True); out[name] = [f[0] for f in fails]
    return out

if __name__ == "__main__":
    import traits_core as C
    fails = run(C); print(f"\nSUITE: {len(TESTS) - len(fails)}/{len(TESTS)} passed")
    if "--mutations" in sys.argv:
        mres = mutation_check(); caught = sum(1 for v in mres.values() if v)
        for k, v in mres.items(): print(("CAUGHT " if v else "MISSED ") + k + (f"  <- {v}" if v else ""))
        print(f"MUTATIONS: {caught}/{len(mres)} caught")
    sys.exit(1 if fails else 0)
