"""v4 diagnostics (protocol_v4.json, evaluations): source probe, group alignment, band occlusion, sex strata.

All on v3 development roles only. Run after `python -m reanalysis.v4_train dev`.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from reanalysis import frontend as fe
from reanalysis import v3_metrics as metrics
from reanalysis import v4_train as v4

PROBE_ORIGINS = ["abuzz", "humbug", "mosquitosong"]
BAND_EDGES = [(0, 300), (300, 600), (600, 1000), (1000, 2000), (2000, 3400)]


def mel_centres():
    edges = fe.mel_to_hz(np.linspace(fe.hz_to_mel(fe.FMIN), fe.hz_to_mel(fe.FMAX), fe.N_MELS + 2))
    return edges[1:-1]


def coral_all(frame, X, seed=v4.SEED):
    """CORAL on all probed rows, per origin (uses origin identity = an unlabeled device batch)."""
    c = v4.Coral().fit(X, seed)
    origins = frame.origin.to_numpy()
    for o in np.unique(origins):
        c.add_origin(o, X[origins == o])
    return c.transform(X, origins)


def source_probe(frame, arrays, seed=v4.SEED):
    use = (frame.target.to_numpy() > 0) & frame.origin.isin(PROBE_ORIGINS).to_numpy()
    sub = frame.loc[use].reset_index(drop=True)
    y = sub.origin.to_numpy()
    reps = {name: arrays[name][use] for name in ["perch_plain", "perch_white", "tabular_white", "band_tabular", "wingbeat"]}
    reps["coral(perch_white)"] = coral_all(sub, arrays["perch_white"][use], seed)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, sub.cluster_merged))
    out = {}
    for name, X in reps.items():
        pred = np.empty(len(y), dtype=object)
        for a, b in folds:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=3000))
            with threadpool_limits(8):
                pred[b] = model.fit(X[a], y[a]).predict(X[b])
        out[name] = dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)), chance=1 / len(PROBE_ORIGINS),
                         n_files=int(len(y)), n_clusters=int(sub.cluster_merged.nunique()))
    return out


def mmd(a, b):
    z = np.r_[a, b]
    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    bw = np.median(sq[sq > 0])
    k = np.exp(-sq / bw)
    n = len(a)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean())


def group_alignment(frame, arrays, seed=v4.SEED, per_origin=500):
    rng = np.random.default_rng(seed)
    use = (frame.target.to_numpy() == 1) & frame.origin.isin(PROBE_ORIGINS).to_numpy()
    sub = frame.loc[use].reset_index(drop=True)
    X = arrays["perch_white"][use]
    reps = {"perch_white_pca128": PCA(128, random_state=seed).fit_transform(X),
            "coral(perch_white)": coral_all(sub, X, seed)}
    pick = {o: rng.choice(np.flatnonzero(sub.origin.eq(o)), min(per_origin, int(sub.origin.eq(o).sum())), replace=False)
            for o in PROBE_ORIGINS}
    out = {}
    for name, Z in reps.items():
        Z = StandardScaler().fit_transform(Z)
        out[name] = {f"{a}~{b}": mmd(Z[pick[a]], Z[pick[b]])
                     for i, a in enumerate(PROBE_ORIGINS) for b in PROBE_ORIGINS[i + 1:]}
    return out


def band_occlusion(frame, arrays):
    centres = mel_centres()
    rows = []
    for corpus in v4.LODO_MOSQUITO:
        directory = v4.OUT / "development" / f"LODO_{corpus}"
        part = json.loads((directory / "partitions.json").read_text())
        fit = frame.index[frame.file_id.isin(part["fit"])].to_numpy()
        test = frame.index[frame.file_id.isin(part["test"])].to_numpy()
        aedes = frame.target.to_numpy()[test] == 2
        for name in ["B1_band_tabular", "C1_aug_band_tabular", "R_tabular_full"]:
            bundle = joblib.load(directory / f"{name}.joblib")["bundle"]
            feature = v4.SPEC[name][0]
            columns = v4.BAND_COLUMNS if feature == "band_tabular" else list(range(89))
            X = arrays[feature]
            base = bundle.scores(X[test])[0]
            for lo, hi in BAND_EDGES:
                bands = [b for b in range(40) if lo <= centres[b] < hi]
                cols = [columns.index(c) for b in bands for c in (b, 40 + b) if c in columns]
                if not cols:
                    continue
                Xo = X[test].copy()
                Xo[:, cols] = X[fit][:, cols].mean(0)
                delta = bundle.scores(Xo)[0] - base
                rows.append(dict(corpus=corpus, model=name, band=f"{lo}-{hi} Hz", n_bands=len(bands),
                                 mean_abs_delta_p_aedes=float(np.abs(delta).mean()),
                                 mean_delta_p_aedes_on_aedes=float(delta[aedes].mean()) if aedes.any() else None))
    return rows


def sex_strata():
    directory = v4.OUT / "development" / "LODO_mosquitosong"
    rows = []
    for name in v4.CANDIDATES:
        p = pd.read_csv(directory / f"{name}.csv")
        p["sex"] = p.path.str.extract(r"_\d+([FM])_")[0]
        for gated in [False, True]:
            for sex, part in p.groupby("sex"):
                m = metrics.point(v4.as_metric_frame(part, gated))
                rows.append(dict(candidate=name, gated=gated, sex=sex, aedes_n=m["aedes_n"], aedes_tp=m["aedes_tp"],
                                 aedes_recall=m["aedes_recall"], other_n=m["other_n"], other_fp=m["other_fp"],
                                 fpr_other=m["fpr_other"], auroc=m["auroc"], aedes_clusters=m["class_clusters"]["2"]))
    return rows


def main():
    v4.check_freeze()
    frame, arrays = v4.load_development()
    out = dict(source_probe=source_probe(frame, arrays), group_alignment=group_alignment(frame, arrays))
    v4.write_json(v4.OUT / "diagnostics.json", out)
    pd.DataFrame(band_occlusion(frame, arrays)).to_csv(v4.OUT / "band_occlusion.csv", index=False)
    pd.DataFrame(sex_strata()).to_csv(v4.OUT / "sex_strata.csv", index=False)
    print(json.dumps(out, indent=2), flush=True)
