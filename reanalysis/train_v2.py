"""Reanalysis v2 - models, cluster-grouped nested evaluation, metrics with cluster bootstrap.

Implements protocol_v2.json. File-level predictions only; every split is by cluster.
Usage:  python reanalysis/train_v2.py probes|E1|E2|E3|S2|S3|report
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/v2"
OUT = ROOT / "results/v2"
PRED = OUT / "predictions"
OUTER_SEED, INNER_SEED = 20261020, 20261021
MODELS = ["M0_majority", "M1_logistic", "M2_histgb", "M3_cnn", "M4_wingbeat_logistic"]
C_GRID = [0.01, 0.1, 1.0]
LEAF_GRID = [15, 31]
N_BOOT = 2000


# ----------------------------------------------------------------------------- data
def load():
    man = pd.read_csv(OUT / "manifest_files.csv")
    win = pd.read_csv(CACHE / "windows.csv")
    meta = json.loads((CACHE / "cache_meta.json").read_text())
    mel = np.load(CACHE / "logmel.npy", mmap_mode="r")
    tab = np.load(CACHE / "tabular.npy", mmap_mode="r")
    assert len(win) == len(mel) == len(tab) and meta["finite"]
    counts = win.file_index.value_counts()
    man["n_windows"] = man.index.map(counts).fillna(0).astype(int)
    return man, win, mel, tab, meta


def windows_of(win, files):
    return np.flatnonzero(win.file_index.isin(files).to_numpy())


def sample_weights(win_files, file_target):
    """Each file total weight 1; classes rescaled to equal total weight."""
    f = pd.Series(win_files)
    w = 1.0 / f.map(f.value_counts()).to_numpy(float)
    y = file_target[win_files]
    present = np.unique(y)
    mass = {c: w[y == c].sum() for c in present}
    w = w * np.array([len(win_files) / (len(present) * mass[c]) for c in y])
    return w


def aggregate(win_files, proba, files):
    """File score = softmax(mean window log-probabilities)."""
    lp = pd.DataFrame(np.log(np.clip(proba, 1e-9, 1)), columns=[0, 1, 2])
    lp["f"] = win_files
    m = lp.groupby("f")[[0, 1, 2]].mean().reindex(files)
    z = m.to_numpy(); z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def file_bacc(y, p):
    present = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in present]))


def file_balanced_logloss(y, prob):
    ll = -np.log(np.clip(prob[np.arange(len(y)), y], 1e-9, 1))
    return float(np.mean([ll[y == c].mean() for c in np.unique(y)]))


# ----------------------------------------------------------------------------- models
def full_proba(model, X, classes):
    p = np.zeros((len(X), 3))
    p[:, classes] = model.predict_proba(X)
    return p


def fit_tabular(kind, X, y, w, param):
    if kind == "logistic":
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=param, max_iter=3000).fit(sc.transform(X), y, sample_weight=w)
        return lambda Z: full_proba(m, sc.transform(Z), m.classes_)
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=param,
                                       l2_regularization=1.0, early_stopping=False,
                                       random_state=OUTER_SEED).fit(X, y, sample_weight=w)
    return lambda Z: full_proba(m, Z, m.classes_)


def make_cnn():
    import torch
    from torch import nn
    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(1, 16), *block(16, 32), *block(32, 64), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 3))


def fit_cnn(X, y, w, seed, epochs, val=None):
    """val = (Xv, win_files_v, files_v, yv_files) -> early stopping on file-level balanced log loss."""
    import torch
    from torch import nn
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = float(np.mean(X)), float(np.std(X)) + 1e-6
    net = make_cnn()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.from_numpy(((np.asarray(X) - mu) / sd).astype("float32"))[:, None]
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64)); wt = torch.from_numpy(np.asarray(w, dtype="float32"))
    def predict(Z):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(Z), 1024):
                z = torch.from_numpy(((np.asarray(Z[s:s + 1024]) - mu) / sd).astype("float32"))[:, None]
                out.append(torch.softmax(net(z), 1).numpy())
        return np.concatenate(out)
    best, best_ep, best_state, stale, hist = np.inf, 0, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        net.train(); order = rng.permutation(len(Xt))
        for s in range(0, len(order), 128):
            ix = torch.from_numpy(order[s:s + 128])
            opt.zero_grad()
            loss = (nn.functional.cross_entropy(net(Xt[ix]), yt[ix], reduction="none") * wt[ix]).sum() / wt[ix].sum()
            loss.backward(); opt.step()
        if val is not None:
            Xv, wf, fv, yv = val
            vl = file_balanced_logloss(yv, aggregate(wf, predict(Xv), fv)); hist.append(vl)
            if vl < best - 1e-4:
                best, best_ep, stale = vl, ep, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= 6: break
    if val is not None:
        net.load_state_dict(best_state)
    return predict, dict(best_epoch=best_ep if val is not None else epochs, val_history=hist)


# ----------------------------------------------------------------------------- nested fit/predict
def inner_splits(man, files):
    sub = man.loc[files]
    sg = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=INNER_SEED)
    return [(files[a], files[b]) for a, b in sg.split(sub, sub.target, sub.cluster)]


def fit_predict(model, man, win, mel, tab, train_files, test_files, seed):
    """Selects hyper-parameters in inner grouped CV on train_files only; refits; predicts test files."""
    y_file = man.target.to_numpy()
    tr_w, te_w = windows_of(win, train_files), windows_of(win, test_files)
    wf_tr, wf_te = win.file_index.to_numpy()[tr_w], win.file_index.to_numpy()[te_w]
    info = {}
    if model == "M0_majority":
        prior = np.bincount(y_file[train_files], minlength=3) / len(train_files)
        p = np.tile(prior, (len(test_files), 1)); info["prior"] = prior.tolist()
        return p, info
    if model in ("M1_logistic", "M2_histgb", "M4_wingbeat_logistic"):
        cols = slice(80, 89) if model == "M4_wingbeat_logistic" else slice(None)
        kind = "histgb" if model == "M2_histgb" else "logistic"
        grid = LEAF_GRID if kind == "histgb" else C_GRID
        scores = {}
        splits = inner_splits(man, np.asarray(train_files))
        for g in grid:
            s = []
            for a, b in splits:
                aw, bw = windows_of(win, a), windows_of(win, b)
                wa = win.file_index.to_numpy()[aw]
                pr = fit_tabular(kind, np.asarray(tab[aw])[:, cols], y_file[wa], sample_weights(wa, y_file), g)
                pb = aggregate(win.file_index.to_numpy()[bw], pr(np.asarray(tab[bw])[:, cols]), b)
                s.append(file_bacc(y_file[b], pb.argmax(1)))
            scores[g] = float(np.mean(s))
        g = max(grid, key=lambda k: (scores[k], -grid.index(k)))
        pr = fit_tabular(kind, np.asarray(tab[tr_w])[:, cols], y_file[wf_tr], sample_weights(wf_tr, y_file), g)
        info.update(inner_scores=scores, chosen=g)
        return aggregate(wf_te, pr(np.asarray(tab[te_w])[:, cols]), test_files), info
    if model == "M3_cnn":
        import torch
        torch.set_num_threads(int(os.environ.get("CNN_THREADS", "4")))
        a, b = inner_splits(man, np.asarray(train_files))[0]
        aw, bw = windows_of(win, a), windows_of(win, b)
        wa = win.file_index.to_numpy()[aw]
        _, sel = fit_cnn(np.asarray(mel[aw]), y_file[wa], sample_weights(wa, y_file), seed, 40,
                         val=(np.asarray(mel[bw]), win.file_index.to_numpy()[bw], b, y_file[b]))
        ep = max(sel["best_epoch"], 1)
        pr, _ = fit_cnn(np.asarray(mel[tr_w]), y_file[wf_tr], sample_weights(wf_tr, y_file), seed, ep)
        info.update(best_epoch=ep, val_history=sel["val_history"])
        return aggregate(wf_te, pr(np.asarray(mel[te_w])), test_files), info
    raise ValueError(model)


def _job(args):
    name, model, train_files, test_files, seed, tag = args
    man, win, mel, tab, _ = load()
    t = time.time()
    p, info = fit_predict(model, man, win, mel, tab, np.asarray(train_files), np.asarray(test_files), seed)
    df = man.loc[test_files, ["path", "source", "species", "class_name", "target", "cluster", "native_sr", "device"]].copy()
    df[["p0", "p1", "p2"]] = p; df["pred"] = p.argmax(1); df["fold"] = tag; df["model"] = model
    info["seconds"] = round(time.time() - t, 1)
    print("done", name, model, tag, info.get("chosen", info.get("best_epoch", "")), info["seconds"], flush=True)
    return name, model, tag, df, info


def run_jobs(jobs, workers):
    PRED.mkdir(parents=True, exist_ok=True)
    results = {}
    with ProcessPoolExecutor(workers) as ex:
        for name, model, tag, df, info in ex.map(_job, jobs):
            results.setdefault((name, model), []).append((tag, df, info))
    for (name, model), items in results.items():
        df = pd.concat([d for _, d, _ in items])
        df.to_csv(PRED / f"{name}__{model}.csv", index=False)
        (PRED / f"{name}__{model}.info.json").write_text(json.dumps({str(t): i for t, _, i in items}, indent=1))
    return results


def dev_files(man, sources=None):
    m = man[(man.role == "development") & (man.n_windows > 0)]
    if sources is not None:
        m = m[m.source.isin(sources)]
    return m.index.to_numpy()


def outer_folds(man, files):
    sub = man.loc[files]
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    folds = [(files[a], files[b]) for a, b in sg.split(sub, sub.target, sub.cluster)]
    for tr, te in folds:
        assert not set(man.cluster[tr]) & set(man.cluster[te]), "cluster leakage"
        assert (man.target[te] == 2).any(), "outer test fold without Aedes"
    return folds


def grouped_cv_jobs(name, man, files):
    folds = outer_folds(man, files)
    rows = [dict(fold=k, test_files=len(te), test_clusters=int(man.cluster[te].nunique()),
                 test_aedes_clusters=int(man.cluster[te][man.target[te] == 2].nunique())) for k, (tr, te) in enumerate(folds)]
    (OUT / f"{name}_folds.json").write_text(json.dumps(rows, indent=1))
    return [(name, m, tr.tolist(), te.tolist(), OUTER_SEED + 101 * k, k) for k, (tr, te) in enumerate(folds) for m in MODELS]


# ----------------------------------------------------------------------------- metrics
def point_metrics(df):
    y, pred = df.target.to_numpy(), df.pred.to_numpy()
    present = sorted(np.unique(y))
    out = dict(n_files=len(df), n_clusters=int(df.cluster.nunique()), classes=[int(c) for c in present],
               balanced_accuracy=file_bacc(y, pred),
               macro_f1=float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
               confusion=np.bincount(3 * y + pred, minlength=9).reshape(3, 3).tolist())
    a = y == 2
    if a.any():
        out.update(aedes_recall=float((pred[a] == 2).mean()), aedes_n=int(a.sum()),
                   aedes_precision=float((y[pred == 2] == 2).mean()) if (pred == 2).any() else None,
                   aedes_predicted=int((pred == 2).sum()))
        if (~a).any():
            out.update(auroc_aedes=float(roc_auc_score(a, df.p2)), auprc_aedes=float(average_precision_score(a, df.p2)),
                       auprc_chance=float(a.mean()))
    for c, nm in [(1, "other"), (0, "noise")]:
        if (y == c).any():
            out[f"fpr_aedes_in_{nm}"] = float((pred[y == c] == 2).mean())
    if (y == 0).any() and (y > 0).any():
        out["auroc_mosquito_vs_noise"] = float(roc_auc_score(y > 0, 1 - df.p0))
    return out


KEYS = ["balanced_accuracy", "macro_f1", "aedes_recall", "aedes_precision", "fpr_aedes_in_other",
        "fpr_aedes_in_noise", "auroc_aedes", "auprc_aedes", "auroc_mosquito_vs_noise"]


def bootstrap_indices(df, n=N_BOOT, seed=7):
    cl = df.groupby("cluster").target.max()
    strata = {s: cl.index[cl == s].to_numpy() for s in cl.unique()}
    pos = df.groupby("cluster").indices
    rng = np.random.default_rng(seed)
    for _ in range(n):
        pick = np.concatenate([rng.choice(v, len(v)) for v in strata.values()])
        yield np.concatenate([pos[c] for c in pick])


def with_ci(dfs):
    """dfs: {model: df} on identical files -> point, CI, paired differences vs M0 and vs best."""
    names = list(dfs)
    base = dfs[names[0]].reset_index(drop=True)
    for m in names:
        assert (dfs[m].path.to_numpy() == base.path.to_numpy()).all()
    res = {m: dict(point=point_metrics(dfs[m])) for m in names}
    boots = {m: {k: [] for k in KEYS} for m in names}
    undefined = 0
    for ix in bootstrap_indices(base):
        for m in names:
            d = dfs[m].iloc[ix]
            try:
                pm = point_metrics(d)
            except ValueError:
                undefined += 1; pm = {}
            for k in KEYS:
                boots[m][k].append(pm.get(k, np.nan) if pm.get(k) is not None else np.nan)
    for m in names:
        res[m]["ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                          for k, v in boots[m].items() if np.isfinite(v).any()}
        for ref in ["M0_majority"]:
            if m != ref and ref in names:
                res[m][f"diff_vs_{ref}"] = {k: [float(np.nanpercentile(np.array(boots[m][k]) - np.array(boots[ref][k]), q)) for q in (2.5, 50, 97.5)]
                                            for k in ["balanced_accuracy", "auroc_aedes"] if np.isfinite(boots[m][k]).any() and np.isfinite(boots[ref][k]).any()}
    res["_bootstrap"] = dict(n=N_BOOT, undefined_replicates=undefined, unit="cluster, stratified by cluster max class")
    return res


# ----------------------------------------------------------------------------- evaluations
def probes(man, win, tab):
    """Shortcut probes on v2 features (and on the legacy features for comparison)."""
    res = {}
    mos = man[(man.role == "development") & (man.target > 0) & man.source.isin(["dryad", "humbug"]) & (man.n_windows > 0)]
    tasks = {"source_dryad_vs_humbug_mosquitoes": (mos.index.to_numpy(), (mos.source == "humbug").astype(int).to_numpy())}
    d = man[(man.source == "dryad") & (man.target == 1) & man.native_sr.isin([8000, 44100]) & (man.n_windows > 0)]
    tasks["native_8k_vs_44k_dryad_other"] = (d.index.to_numpy(), (d.native_sr == 44100).astype(int).to_numpy())
    for name, (files, lab) in tasks.items():
        sub = man.loc[files]
        lab_s = pd.Series(lab, index=files)
        out = {}
        for kind, param in [("logistic", 0.1), ("histgb", 15)]:
            preds = pd.Series(index=files, dtype=float)
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=OUTER_SEED).split(sub, lab, sub.cluster):
                fa, fb = files[a], files[b]
                aw, bw = windows_of(win, fa), windows_of(win, fb)
                wa = win.file_index.to_numpy()[aw]; wb = win.file_index.to_numpy()[bw]
                ya = lab_s.loc[wa].to_numpy()
                X = np.asarray(tab[aw])
                if kind == "logistic":
                    sc = StandardScaler().fit(X); m = LogisticRegression(C=param, max_iter=3000, class_weight="balanced").fit(sc.transform(X), ya)
                    pw = m.predict_proba(sc.transform(np.asarray(tab[bw])))[:, 1]
                else:
                    m = HistGradientBoostingClassifier(max_leaf_nodes=param, learning_rate=0.05, max_iter=300, class_weight="balanced",
                                                       random_state=OUTER_SEED).fit(X, ya)
                    pw = m.predict_proba(np.asarray(tab[bw]))[:, 1]
                preds.loc[fb] = pd.Series(pw).groupby(wb).mean().reindex(fb).to_numpy()
            yy = lab_s.loc[files].to_numpy(); pp = preds.loc[files].to_numpy()
            out[kind] = dict(balanced_accuracy=file_bacc(yy, (pp > 0.5).astype(int)), auroc=float(roc_auc_score(yy, pp)),
                             n_files=len(files), n_clusters=int(sub.cluster.nunique()), positives=int(yy.sum()))
        res[name] = out
    res["legacy_features"] = legacy_probe(man)
    (OUT / "shortcut_probe.json").write_text(json.dumps(res, indent=1))
    return res


def legacy_probe(man):
    """Same source probe on the cached v1 features (data/cache/reliability_2026_10_01/features.npy)."""
    path = ROOT / "data/cache/reliability_2026_10_01/features.npy"
    if not path.exists():
        return "legacy cache missing"
    seg = pd.read_csv(ROOT / "data/metadata/ternary_segments.csv.gz", low_memory=False)
    F = np.load(path, mmap_mode="r")
    keep = np.isfinite(F).all(1) & seg.path.isin(man.path).to_numpy()
    seg = seg[keep]; X = np.asarray(F[keep])
    info = man.set_index("path").loc[seg.path]
    ok = ((info.role == "development") & (info.target > 0) & info.source.isin(["dryad", "humbug"])).to_numpy()
    seg, X, info = seg[ok], X[ok], info[ok]
    y = (info.source == "humbug").astype(int).to_numpy(); groups = info.cluster.to_numpy(); fpath = seg.path.to_numpy()
    files = pd.Series(y, index=fpath).groupby(level=0).first()
    fclust = pd.Series(groups, index=fpath).groupby(level=0).first()
    preds = pd.Series(index=files.index, dtype=float)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=OUTER_SEED).split(files, files.to_numpy(), fclust.to_numpy()):
        fa, fb = set(files.index[a]), files.index[b]
        ia = np.isin(fpath, list(fa)); ib = np.isin(fpath, fb)
        m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=300, class_weight="balanced",
                                           random_state=OUTER_SEED).fit(X[ia], y[ia])
        preds.loc[fb] = pd.Series(m.predict_proba(X[ib])[:, 1], index=fpath[ib]).groupby(level=0).mean().reindex(fb).to_numpy()
    return dict(task="source_dryad_vs_humbug_mosquitoes", model="histgb", n_files=len(files),
                balanced_accuracy=file_bacc(files.to_numpy(), (preds.to_numpy() > 0.5).astype(int)),
                auroc=float(roc_auc_score(files.to_numpy(), preds.to_numpy())))


def main(stage, workers=5):
    man, win, mel, tab, meta = load()
    OUT.mkdir(parents=True, exist_ok=True)
    if stage == "probes":
        print(json.dumps(probes(man, win, tab), indent=1)); return
    if stage == "E1":
        run_jobs(grouped_cv_jobs("E1", man, dev_files(man)), workers)
    elif stage == "S2":
        run_jobs(grouped_cv_jobs("S2", man, dev_files(man, ["dryad", "humbug"])), workers)
    elif stage == "S3":
        f = dev_files(man, ["humbug"])
        f = f[(man.setting[f] == "Tanzania|Ifakara|tascam|cup").to_numpy() & (man.target[f] > 0).to_numpy()]
        sub = man.loc[f]
        sg = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=OUTER_SEED)
        folds = [(f[a], f[b]) for a, b in sg.split(sub, sub.target, sub.cluster)]
        jobs = [("S3", m, tr.tolist(), te.tolist(), OUTER_SEED + k, k) for k, (tr, te) in enumerate(folds)
                for m in MODELS if (man.target[te] == 2).any()]
        run_jobs(jobs, workers)
    elif stage == "E2":
        noise = dev_files(man, ["pete_noise", "esc50"])
        dry, hum = dev_files(man, ["dryad"]), dev_files(man, ["humbug"])
        jobs = [("E2_dryad_to_humbug", m, np.r_[dry, noise].tolist(), hum.tolist(), OUTER_SEED, 0) for m in MODELS]
        jobs += [("E2_humbug_to_dryad", m, np.r_[hum, noise].tolist(), dry.tolist(), OUTER_SEED, 0) for m in MODELS]
        run_jobs(jobs, workers)
    elif stage == "E3":
        ext = man[(man.role == "external") & (man.n_windows > 0)].index.to_numpy()
        jobs = [("E3_external", m, dev_files(man).tolist(), ext.tolist(), OUTER_SEED, 0) for m in MODELS]
        run_jobs(jobs, workers)
    elif stage == "report":
        report()
    code = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ["reanalysis/train_v2.py", "reanalysis/frontend.py", "reanalysis/protocol_v2.json"]}
    (OUT / f"run_{stage}.json").write_text(json.dumps(dict(stage=stage, code_sha256=code, cache_sha256=meta["sha256"]), indent=1))


def report():
    out = {}
    for name in ["E1", "S2", "S3", "E2_dryad_to_humbug", "E2_humbug_to_dryad", "E3_external"]:
        dfs = {}
        for m in MODELS:
            p = PRED / f"{name}__{m}.csv"
            if p.exists():
                dfs[m] = pd.read_csv(p).sort_values("path").reset_index(drop=True)
        if not dfs:
            continue
        out[name] = with_ci(dfs)
        if name == "E1":  # S1: stratified by native sample rate, no refit
            for sr in [8000, 44100]:
                sub = {m: d[d.native_sr == sr].reset_index(drop=True) for m, d in dfs.items()}
                if (sub["M0_majority"].target == 2).any():
                    out[f"S1_E1_native_{sr}"] = with_ci(sub)
            for src in ["dryad", "humbug"]:
                sub = {m: d[d.source == src].reset_index(drop=True) for m, d in dfs.items()}
                out[f"E1_by_source_{src}"] = with_ci(sub)
        if name == "E3_external":
            for src in ["mosquitosong", "ufrgs"]:
                sub = {m: d[d.source == src].reset_index(drop=True) for m, d in dfs.items()}
                out[f"E3_{src}"] = with_ci(sub)
        print("reported", name, flush=True)
    (OUT / "metrics_v2.json").write_text(json.dumps(out, indent=1, default=float))
    rows = []
    for ev, r in out.items():
        for m, v in r.items():
            if m.startswith("_"): continue
            row = dict(evaluation=ev, model=m, n_files=v["point"]["n_files"], n_clusters=v["point"]["n_clusters"])
            for k in KEYS:
                if k in v["point"] and v["point"][k] is not None:
                    ci = v["ci95"].get(k, [np.nan, np.nan])
                    row[k] = f"{v['point'][k]:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "tabela_resultados_v2.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
