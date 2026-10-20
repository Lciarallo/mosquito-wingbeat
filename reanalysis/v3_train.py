"""Train the preregistered v3 continuation with independent cluster calibration.

Commands: dev (LODO + pooled CV), final (freeze, then score once), fewshot.
All heavy input and model files stay in ignored local directories.
"""
import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from reanalysis import v3_metrics as metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_continuation"
CACHE = ROOT / "data/cache/v3_continuation"
CANDIDATES = ["A_perch_plain", "A_perch_white", "B_mmd_plain", "B_mmd_white", "C_wingbeat", "D_tabular"]
CORPORA = ["abuzz", "humbug", "mosquitosong"]
SEED = 20261023
EPOCHS = 20


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load():
    meta = json.loads((CACHE / "meta.json").read_text())
    if not meta.get("complete") or digest(CACHE / "features.npz") != meta["features_sha256"]:
        raise ValueError("Incomplete or changed acoustic feature cache")
    manifest_path = OUT / "manifest.csv"
    if digest(manifest_path) != meta["manifest_sha256"]:
        raise ValueError("Feature/manifest identity mismatch")
    frozen = json.loads((ROOT / "reanalysis/protocol_v3_continuation_freeze.json").read_text())
    for path, sha in frozen["sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    m = pd.read_csv(manifest_path).set_index("file_id")
    with np.load(CACHE / "features.npz") as cache:
        m = m.loc[cache["file_ids"]].reset_index()
        arrays = {key: cache[key].copy() for key in cache.files if key not in ["file_ids", "n_chunks"]}
    m["target"] = m.target.astype(int)
    arrays["wingbeat"] = arrays["tabular_white"][:, 80:89]
    duration = repair_duration_metadata(m)
    arrays["duration"] = np.stack([np.log(duration), np.maximum(0, 1 - duration / 5)], axis=1).astype("float32")
    return m, arrays


def repair_duration_metadata(frame, known=None):
    """Recover missing container durations from existing label-free v2 metadata.

    soundfile cannot probe some MP4/M4A containers. The v2 manifest already
    probed those with ffprobe. No values are estimated from labels or scores.
    This affects only the duration diagnostic, not any acoustic candidate.
    """
    duration = frame.duration_s.to_numpy(float).copy()
    missing = ~np.isfinite(duration) | (duration <= 0)
    if missing.any():
        if known is None:
            source = ROOT / "reanalysis/manifest_files.csv"
            parent_freeze = json.loads((ROOT / "reanalysis/protocol_freeze.json").read_text())
            if digest(source) != parent_freeze["manifest_sha256"]:
                raise ValueError("The frozen source of physical durations changed")
            known = pd.read_csv(source).set_index("path").duration_s
        duration[missing] = frame.path[missing].map(known).to_numpy(float)
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Missing physical duration after the metadata backfill")
    return duration


def balanced_weights(y, multiplier=None):
    w = np.ones(len(y), dtype=float) if multiplier is None else np.asarray(multiplier, dtype=float).copy()
    for c in np.unique(y):
        use = y == c
        w[use] *= len(y) / (len(np.unique(y)) * w[use].sum())
    return w


class MMDNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(dim, 256), nn.ReLU())
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        h = self.hidden(x)
        return self.output(self.dropout(h)).squeeze(1), h


def mmd_penalty(hidden, domain):
    squared = torch.cdist(hidden, hidden).square()
    positive = squared.detach()[squared.detach() > 0]
    bandwidth = positive.median().clamp_min(1e-6) if len(positive) else squared.new_tensor(1.)
    kernel = sum(torch.exp(-squared / (2 * bandwidth * scale)) for scale in [.5, 1., 2.]) / 3
    present = [d for d in domain.unique() if int((domain == d).sum()) >= 2]
    losses = []
    for a, b in itertools.combinations(present, 2):
        ia, ib = domain == a, domain == b
        losses.append(kernel[ia][:, ia].mean() + kernel[ib][:, ib].mean() - 2 * kernel[ia][:, ib].mean())
    return torch.stack(losses).mean() if losses else hidden.sum() * 0


@dataclass
class Head:
    kind: str
    param: float
    scaler: object = None
    model: object = None
    constant: float = None

    def logits(self, X):
        if self.constant is not None:
            return np.full(len(X), self.constant)
        Z = self.scaler.transform(X).astype("float32") if self.scaler is not None else X
        if self.kind == "mlp":
            self.model.eval()
            with torch.no_grad():
                return np.concatenate([self.model(torch.from_numpy(Z[i:i + 1024]))[0].numpy()
                                       for i in range(0, len(Z), 1024)])
        if self.kind == "histgb":
            return logit(np.clip(self.model.predict_proba(Z)[:, 1], 1e-8, 1 - 1e-8))
        return self.model.decision_function(Z)


def fit_head(kind, param, X, y, origins, seed, multiplier=None):
    head = Head(kind, float(param))
    if len(np.unique(y)) < 2:
        head.constant = float(logit(np.clip(np.mean(y), 1e-8, 1 - 1e-8)))
        return head
    weights = balanced_weights(y, multiplier)
    if kind == "histgb":
        head.model = HistGradientBoostingClassifier(max_leaf_nodes=int(param), max_iter=200, learning_rate=.05,
                                                    l2_regularization=1., early_stopping=False, random_state=seed)
        head.model.fit(X, y, sample_weight=weights)
        return head
    head.scaler = StandardScaler().fit(X)
    Z = head.scaler.transform(X).astype("float32")
    if kind == "logistic":
        head.model = LogisticRegression(C=param, max_iter=3000).fit(Z, y, sample_weight=weights)
        return head
    torch.manual_seed(seed)
    head.model = MMDNet(X.shape[1])
    opt = torch.optim.AdamW(head.model.parameters(), lr=.001, weight_decay=.001)
    xt, yt, wt = torch.from_numpy(Z), torch.from_numpy(y.astype("float32")), torch.from_numpy(weights.astype("float32"))
    dt = torch.from_numpy(np.unique(origins, return_inverse=True)[1])
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        head.model.train()
        order = rng.permutation(len(X))
        for start in range(0, len(order), 256):
            ix = torch.from_numpy(order[start:start + 256])
            opt.zero_grad()
            raw, hidden = head.model(xt[ix])
            loss = (nn.functional.binary_cross_entropy_with_logits(raw, yt[ix], reduction="none") * wt[ix]).mean()
            if param:
                loss = loss + param * mmd_penalty(hidden, dt[ix])
            loss.backward()
            opt.step()
    head.model.eval()
    return head


def select_head(kind, X, y, frame, seed):
    grid = [0., .3, 1.] if kind == "mlp" else [15, 31] if kind == "histgb" else [.01, .1, 1.]
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED + 1).split(X, y, frame.cluster_merged))
    scores = []
    for param in grid:
        values = []
        for k, (fit, valid) in enumerate(splits):
            if set(frame.cluster_merged.iloc[fit]) & set(frame.cluster_merged.iloc[valid]):
                raise ValueError("Inner cluster overlap")
            head = fit_head(kind, param, X[fit], y[fit], frame.origin.to_numpy()[fit], seed + k)
            pred = head.logits(X[valid]) >= 0
            values.append(float(balanced_accuracy_score(y[valid], pred)))
        scores.append(dict(param=float(param), score=float(np.mean(values))))
    best = max(scores, key=lambda row: row["score"])["param"]
    head = fit_head(kind, best, X, y, frame.origin.to_numpy(), seed)
    return head, dict(chosen=best, inner_scores=scores)


@dataclass
class ProbabilityMap:
    model: object = None
    constant: float = None

    def apply(self, values):
        if self.constant is not None:
            return np.full(len(values), self.constant)
        return self.model.predict_proba(np.asarray(values).reshape(-1, 1))[:, 1]


def calibrator(values, y):
    if len(np.unique(y)) < 2:
        return ProbabilityMap(constant=float(np.mean(y)))
    model = LogisticRegression(C=1., max_iter=2000).fit(np.asarray(values).reshape(-1, 1), y)
    return ProbabilityMap(model=model)


def threshold(y, score, presence):
    """Optimise only empirical calibration FPRs. >1 disables alerts everywhere."""
    active = presence >= .5
    candidates = np.unique(np.r_[0., score[active], np.nextafter(1., np.inf)])
    for tau in candidates:
        alert = active & (score >= tau)
        valid = True
        for c, cap in [(1, .02), (0, .01)]:
            if (y == c).any() and (alert[y == c]).mean() > cap + 1e-15:
                valid = False
                break
        if valid:
            return float(tau)
    return float(np.nextafter(1., np.inf))


def split_fit_calibration(frame, train, seed):
    """Deterministic cluster split, stratified without inspecting any test score."""
    sub = frame.iloc[train]
    cl = sub.groupby("cluster_merged").target.max()
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in cl.groupby(cl):
        clusters = np.sort(group.index.to_numpy())
        if len(clusters) >= 2:
            count = max(1, int(np.floor(.25 * len(clusters))))
            selected.extend(rng.choice(clusters, count, replace=False))
    is_cal = sub.cluster_merged.isin(selected).to_numpy()
    fit, cal = np.asarray(train)[~is_cal], np.asarray(train)[is_cal]
    for c in [0, 1, 2]:
        if not (frame.target.iloc[fit] == c).any() or not (frame.target.iloc[cal] == c).any():
            raise ValueError(f"Fit/calibration split lacks class {c}; cannot silently relax cluster separation")
    assert not set(frame.cluster_merged.iloc[fit]) & set(frame.cluster_merged.iloc[cal])
    return fit, cal


def candidate_spec(name):
    if name == "C_wingbeat":
        return "wingbeat", "logistic"
    if name == "D_tabular":
        return "tabular_white", "histgb"
    if name == "duration_diagnostic":
        return "duration", "logistic"
    return "perch_white" if name.endswith("white") else "perch_plain", "mlp" if name.startswith("B_") else "logistic"


@dataclass
class Bundle:
    candidate: str
    feature: str
    stage1: Head
    stage2: Head
    presence_map: ProbabilityMap
    aedes_map: ProbabilityMap
    tau: float
    info: dict

    def scores(self, arrays, indices):
        X = arrays[self.feature][indices]
        presence = self.presence_map.apply(self.stage1.logits(X))
        conditional = self.aedes_map.apply(self.stage2.logits(X))
        return presence * conditional, presence

    def predict(self, arrays, indices):
        score, presence = self.scores(arrays, indices)
        pred = np.where(presence < .5, 0, np.where(score >= self.tau, 2, 1))
        return pred, score, presence


def fit_bundle(name, frame, arrays, fit, cal, seed):
    feature, kind = candidate_spec(name)
    X = arrays[feature]
    y = frame.target.to_numpy(int)
    sub = frame.iloc[fit].reset_index(drop=True)
    presence_kind = "histgb" if kind == "histgb" else "logistic"
    first, first_info = select_head(presence_kind, X[fit], (y[fit] > 0).astype(int), sub, seed)
    fit_mosquito = fit[y[fit] > 0]
    second, second_info = select_head(kind, X[fit_mosquito], (y[fit_mosquito] == 2).astype(int),
                                      frame.iloc[fit_mosquito].reset_index(drop=True), seed + 100)
    presence_map = calibrator(first.logits(X[cal]), (y[cal] > 0).astype(int))
    cal_mosquito = cal[y[cal] > 0]
    aedes_map = calibrator(second.logits(X[cal_mosquito]), (y[cal_mosquito] == 2).astype(int))
    bundle = Bundle(name, feature, first, second, presence_map, aedes_map, 1.,
                    dict(stage1=first_info, stage2=second_info, fit_file_ids=frame.file_id.iloc[fit].tolist(),
                         calibration_file_ids=frame.file_id.iloc[cal].tolist(), seed=seed))
    score, presence = bundle.scores(arrays, cal)
    bundle.tau = threshold(y[cal], score, presence)
    bundle.info["threshold"] = bundle.tau
    bundle.info["calibration_counts"] = {str(c): int((y[cal] == c).sum()) for c in [0, 1, 2]}
    return bundle


def prediction_frame(bundle, gate, frame, arrays, test, fold):
    pred, score, presence = bundle.predict(arrays, test)
    ungated = pred.copy()
    if gate is not None and bundle.candidate not in ["C_wingbeat", "duration_diagnostic"]:
        gate_pred, _, _ = gate.predict(arrays, test)
        pred[(pred == 2) != (gate_pred == 2)] = 3
    columns = ["file_id", "path", "target", "origin", "raw_source", "species", "cluster_merged", "role",
               "previously_scored_component", "provenance_note", "duration_s"]
    out = frame.iloc[test][columns].copy()
    out["pred"], out["ungated_pred"], out["score"], out["presence_score"] = pred, ungated, score, presence
    out["threshold"], out["fold"], out["candidate"] = bundle.tau, fold, bundle.candidate
    return out


def ensure_training_identity():
    paths = ["reanalysis/v3_train.py", "reanalysis/v3_metrics.py", "reanalysis/protocol_v3_continuation.json"]
    identity = dict(code_sha256={p: digest(ROOT / p) for p in paths},
                    cache_meta_sha256=digest(CACHE / "meta.json"), manifest_sha256=digest(OUT / "manifest.csv"),
                    package_versions={key: importlib.metadata.version(key) for key in
                                      ["numpy", "scipy", "pandas", "scikit-learn", "torch", "joblib"]})
    path = OUT / "training_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Changed training code/inputs: use an explicit amendment and a new run directory")
    write_json(path, identity)
    return identity


def _outer_run(tag, frame, arrays, train, test, seed, names):
    directory = OUT / "development" / tag
    directory.mkdir(parents=True, exist_ok=True)
    touched = set(frame.cluster_merged.iloc[test])
    train = np.array([i for i in train if frame.cluster_merged.iloc[i] not in touched], dtype=int)
    for key in ["path", "component", "cluster_merged"]:
        if set(frame[key].iloc[train]) & set(frame[key].iloc[test]):
            raise ValueError(f"Outer overlap: {tag}, {key}")
    fit, cal = split_fit_calibration(frame, train, seed)
    write_json(directory / "partitions.json", dict(fit=frame.file_id.iloc[fit].tolist(),
               calibration=frame.file_id.iloc[cal].tolist(), test=frame.file_id.iloc[test].tolist(),
               disjoint_paths_components_clusters=True))
    gate = None
    for name in ["C_wingbeat"] + [n for n in names if n != "C_wingbeat"]:
        path = directory / f"{name}.joblib"
        if path.exists():
            artifact = json.loads((directory / f"{name}.info.json").read_text())
            if digest(path) != artifact["model_sha256"]:
                raise ValueError(f"Changed cached development model: {path}")
            bundle = joblib.load(path)
        else:
            started = time.monotonic()
            bundle = fit_bundle(name, frame, arrays, fit, cal, seed)
            joblib.dump(bundle, path, compress=3)
            write_json(directory / f"{name}.info.json", dict(bundle.info, model_sha256=digest(path)))
            print("FIT", tag, name, "seconds", round(time.monotonic() - started, 1), flush=True)
        if name == "C_wingbeat":
            gate = bundle
        destination = directory / f"{name}.csv"
        if not destination.exists():
            prediction_frame(bundle, gate, frame, arrays, test, tag).to_csv(destination, index=False)
        print("DONE", tag, name, flush=True)


def select_candidate(lodo_points):
    rows = []
    for name in CANDIDATES:
        points = [lodo_points[c][name] for c in CORPORA]
        mean = lambda key, default: float(np.mean([p[key] if p[key] is not None else default for p in points]))
        fprs = [p["fpr_other"] for p in points if p["fpr_other"] is not None]
        noise = [p["fpr_noise"] for p in points if p["fpr_noise"] is not None]
        row = dict(candidate=name, precision=mean("aedes_precision", 0.), recall=mean("aedes_recall", 0.),
                   auroc=mean("auroc", .5), other_fpr=float(np.mean(fprs)) if fprs else None,
                   noise_fpr=float(np.mean(noise)) if noise else None)
        row["eligible"] = bool(row["recall"] >= .2 and fprs and max(fprs) <= .02
                               and (not noise or max(noise) <= .01))
        rows.append(row)
    eligible = [r for r in rows if r["eligible"]]
    chosen = max(eligible, key=lambda r: (r["precision"], r["auroc"])) if eligible else max(rows, key=lambda r: r["auroc"])
    return dict(candidate=chosen["candidate"], eligible=bool(eligible), candidates=rows,
                reason="Highest development precision subject to recall/FPR constraints" if eligible else
                       "No candidate met the development criteria; highest-AUROC descriptive fallback only")


def development(frame, arrays):
    ensure_training_identity()
    dev = np.flatnonzero(frame.role.isin(["train", "noise_train"]))
    names = CANDIDATES + ["duration_diagnostic"]
    for k, corpus in enumerate(CORPORA):
        test = dev[frame.origin.iloc[dev].eq(corpus).to_numpy()]
        train = dev[~frame.origin.iloc[dev].eq(corpus).to_numpy()]
        _outer_run(f"LODO_{corpus}", frame, arrays, train, test, SEED + k, names)
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=20261020).split(
        dev, frame.target.iloc[dev], frame.cluster_merged.iloc[dev])
    for k, (a, b) in enumerate(folds):
        _outer_run(f"pooled_{k}", frame, arrays, dev[a], dev[b], SEED + 10 + k, names)
    points = {c: {} for c in CORPORA}
    all_metrics = {}
    for name in names:
        for corpus in CORPORA:
            d = pd.read_csv(OUT / "development" / f"LODO_{corpus}" / f"{name}.csv")
            all_metrics.setdefault(f"LODO_{corpus}", {})[name] = metrics.with_uncertainty(d)
            points[corpus][name] = all_metrics[f"LODO_{corpus}"][name]["point"]
        pooled = pd.concat([pd.read_csv(OUT / "development" / f"pooled_{k}" / f"{name}.csv") for k in range(5)])
        if not pooled.file_id.is_unique or set(pooled.file_id) != set(frame.file_id.iloc[dev]):
            raise ValueError("Pooled OOF coverage mismatch")
        pooled.to_csv(OUT / f"pooled__{name}.csv", index=False)
        all_metrics.setdefault("pooled", {})[name] = metrics.with_uncertainty(pooled)
    write_json(OUT / "development_metrics.json", all_metrics)
    selection = select_candidate(points)
    write_json(OUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def final(frame, arrays):
    identity = ensure_training_identity()
    selection = json.loads((OUT / "selection.json").read_text())
    path = OUT / "final_metrics.json"
    if path.exists():
        existing_freeze = json.loads((OUT / "final_freeze.json").read_text())
        if selection != existing_freeze["selection"] or identity != existing_freeze["training_identity"]:
            raise ValueError("Final inputs changed after the recorded evaluation")
        print("Final evaluation already recorded; no rescore.", flush=True)
        return
    directory = OUT / "final"
    directory.mkdir(exist_ok=True)
    dev = np.flatnonzero(frame.role.isin(["train", "noise_train"]))
    fit, cal = split_fit_calibration(frame, dev, SEED + 99)
    bundles = {}
    for name in dict.fromkeys(["C_wingbeat", selection["candidate"]]):
        model_path = directory / f"{name}.joblib"
        if model_path.exists():
            bundle = joblib.load(model_path)
        else:
            bundle = fit_bundle(name, frame, arrays, fit, cal, SEED + 99)
            joblib.dump(bundle, model_path, compress=3)
        bundles[name] = bundle
    freeze = dict(selection=selection, training_identity=identity,
                  development_metrics_sha256=digest(OUT / "development_metrics.json"),
                  selection_sha256=digest(OUT / "selection.json"),
                  model_sha256={name: digest(directory / f"{name}.joblib") for name in bundles},
                  fit_file_ids=frame.file_id.iloc[fit].tolist(), calibration_file_ids=frame.file_id.iloc[cal].tolist(),
                  no_final_predictions_inspected=True)
    frozen_path = OUT / "final_freeze.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != freeze:
        raise ValueError("Final frozen configuration changed; refusing a second adaptive evaluation")
    write_json(frozen_path, freeze)
    # FIRST call predicting final rows occurs strictly after the configuration/model freeze.
    test = np.flatnonzero(frame.role.isin(["locked_test", "locked_noise"]))
    pred = prediction_frame(bundles[selection["candidate"]], bundles["C_wingbeat"], frame, arrays, test, "final")
    pred.to_csv(OUT / "final_predictions.csv", index=False)
    mosquito = pred[pred.role.eq("locked_test")]
    noise = pred[pred.role.eq("locked_noise")]
    result = dict(selection=selection, locked_mosquito=metrics.with_uncertainty(mosquito),
                  locked_noise=metrics.with_uncertainty(noise), combined=metrics.with_uncertainty(pred))
    for key, subset in [("previously_scored", mosquito[mosquito.previously_scored_component]),
                        ("not_previously_scored", mosquito[~mosquito.previously_scored_component])]:
        result[key] = metrics.with_uncertainty(subset)
    result["provenance"] = {key: metrics.with_uncertainty(sub) for key, sub in mosquito.groupby("provenance_note")}
    ungated = pred.copy()
    ungated["pred"] = ungated.ungated_pred
    result["ungated_sensitivity_analysis"] = metrics.with_uncertainty(ungated)
    result["success"] = metrics.success(result["locked_mosquito"], result["locked_noise"])
    if not selection["eligible"]:
        result["success"]["statistical_support"] = False
        result["success"]["development_eligible"] = False
    write_json(path, result)
    print(json.dumps(dict(selected=selection["candidate"], success=result["success"],
                          mosquito=result["locked_mosquito"]["point"], noise=result["locked_noise"]["point"]), indent=2), flush=True)


def choose_fewshot(frame, test, k, seed):
    """Only calibration clusters contribute files; every touched cluster is purged."""
    rng = np.random.default_rng(seed)
    chosen = []
    sub = frame.iloc[test]
    for c in sorted(sub.target.unique()):
        rows = sub[sub.target.eq(c)]
        clusters = np.sort(rows.cluster_merged.unique())
        if len(clusters) < 2:
            return None, None, f"class_{c}_has_fewer_than_2_clusters"
        candidates = rng.permutation(clusters)
        selected = []
        # At most half the clusters: keep independent evaluation clusters in every class.
        for cluster in candidates[:max(1, len(clusters) // 2)]:
            selected.append(cluster)
            if rows.cluster_merged.isin(selected).sum() >= k:
                break
        available = rows[rows.cluster_merged.isin(selected)].index.to_numpy()
        if len(available) < k:
            return None, None, f"class_{c}_insufficient_files_without_consuming_test_clusters"
        chosen.extend(rng.choice(available, k, replace=False))
    chosen = np.array(chosen, dtype=int)
    touched = set(frame.cluster_merged.iloc[chosen])
    evaluate = np.array([i for i in test if frame.cluster_merged.iloc[i] not in touched], dtype=int)
    if not (frame.target.iloc[evaluate] == 2).any() or not (frame.target.iloc[evaluate] == 1).any():
        return None, None, "no_disjoint_positive_and_negative_evaluation"
    return chosen, evaluate, None


def adapt_bundle(bundle, frame, arrays, calibration, seed):
    import copy
    adapted = copy.deepcopy(bundle)
    indexed = frame.reset_index().set_index("file_id")["index"]
    fit = indexed.loc[bundle.info["fit_file_ids"]].to_numpy(int)
    augmented = np.r_[fit, calibration]
    multiplier = np.r_[np.ones(len(fit)), np.full(len(calibration), 5.)]
    y = frame.target.to_numpy(int)
    X = arrays[bundle.feature]
    adapted.stage1 = fit_head(bundle.stage1.kind, bundle.stage1.param, X[augmented], (y[augmented] > 0).astype(int),
                              frame.origin.iloc[augmented].to_numpy(), seed, multiplier)
    mosquito = y[augmented] > 0
    second = augmented[mosquito]
    adapted.stage2 = fit_head(bundle.stage2.kind, bundle.stage2.param, X[second], (y[second] == 2).astype(int),
                              frame.origin.iloc[second].to_numpy(), seed + 100, multiplier[mosquito])
    # Source maps and thresholds remain frozen, as declared before adaptation.
    return adapted


def fewshot(frame, arrays):
    ensure_training_identity()
    if not (OUT / "final_metrics.json").exists():
        raise ValueError("Freeze and complete zero-shot final evaluation before running adaptation arms")
    selection = json.loads((OUT / "selection.json").read_text())
    name = selection["candidate"]
    summaries = []
    for corpus in CORPORA + ["final_locked"]:
        directory = OUT / "final" if corpus == "final_locked" else OUT / "development" / f"LODO_{corpus}"
        base = joblib.load(directory / f"{name}.joblib")
        gate = joblib.load(directory / "C_wingbeat.joblib")
        test = np.flatnonzero(frame.role.eq("locked_test")) if corpus == "final_locked" else np.flatnonzero(
            frame.role.isin(["train", "noise_train"]) & frame.origin.eq(corpus))
        for k in [5, 10, 20]:
            for repeat in range(5):
                seed = SEED + repeat
                record = dict(corpus=corpus, k=k, repeat=repeat, seed=seed)
                destination = OUT / "fewshot" / f"{corpus}_k{k}_r{repeat}.csv"
                summary_path = destination.with_suffix(".json")
                if summary_path.exists():
                    summaries.append(json.loads(summary_path.read_text()))
                    continue
                cal, evaluate, reason = choose_fewshot(frame, test, k, seed)
                if reason:
                    record.update(available=False, reason=reason)
                else:
                    fitted = adapt_bundle(base, frame, arrays, cal, seed)
                    fitted_gate = fitted if name == "C_wingbeat" else adapt_bundle(gate, frame, arrays, cal, seed)
                    predictions = prediction_frame(fitted, fitted_gate, frame, arrays, evaluate,
                                                   f"fewshot_{corpus}_k{k}_r{repeat}")
                    destination.parent.mkdir(exist_ok=True)
                    predictions.to_csv(destination, index=False)
                    record.update(available=True, calibration_file_ids=frame.file_id.iloc[cal].tolist(),
                                  evaluation_file_ids=frame.file_id.iloc[evaluate].tolist(),
                                  metrics=metrics.point(predictions),
                                  uncertainty=metrics.with_uncertainty(predictions),
                                  thresholds_and_source_calibrators_unchanged=True)
                write_json(summary_path, record)
                summaries.append(record)
                print("FEWSHOT", corpus, k, repeat, record.get("available"), record.get("reason", ""), flush=True)
    write_json(OUT / "fewshot_summary.json", summaries)


def main(stage):
    with threadpool_limits(limits=1):
        torch.set_num_threads(4)
        frame, arrays = load()
        if stage == "dev":
            development(frame, arrays)
        elif stage == "final":
            final(frame, arrays)
        elif stage == "fewshot":
            fewshot(frame, arrays)


if __name__ == "__main__":
    # Keep saved bundles loadable from an inference script as well as `python -m`.
    import sys
    sys.modules["reanalysis.v3_train"] = sys.modules[__name__]
    for cls in [Head, ProbabilityMap, Bundle, MMDNet]:
        cls.__module__ = "reanalysis.v3_train"
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["dev", "final", "fewshot"])
    main(parser.parse_args().stage)
