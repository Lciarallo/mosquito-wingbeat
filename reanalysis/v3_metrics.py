"""File-level alert metrics with paired, cluster-stratified uncertainty.

Abstentions remain in all relevant denominators. A degenerate bootstrap with
zero observed errors is explicitly not a population false-positive guarantee.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

KEYS = ["aedes_precision", "aedes_recall", "fpr_other", "fpr_noise", "coverage",
        "balanced_accuracy", "auroc", "auprc"]


def _ratio(a, b):
    return float(a / b) if b else None


def point(frame):
    y = frame.target.to_numpy(int)
    pred = frame.pred.to_numpy(int)
    alert = pred == 2
    aedes = y == 2
    other, noise = y == 1, y == 0
    tp = int((alert & aedes).sum())
    out = dict(n_files=len(frame), n_clusters=int(frame.cluster_merged.nunique()),
               aedes_n=int(aedes.sum()), other_n=int(other.sum()), noise_n=int(noise.sum()),
               aedes_tp=tp, other_fp=int((alert & other).sum()), noise_fp=int((alert & noise).sum()),
               alerts=int(alert.sum()), uncertain=int((pred == 3).sum()),
               aedes_precision=_ratio(tp, int(alert.sum())), aedes_recall=_ratio(tp, int(aedes.sum())),
               fpr_other=_ratio(int((alert & other).sum()), int(other.sum())),
               fpr_noise=_ratio(int((alert & noise).sum()), int(noise.sum())),
               prevalence=float(aedes.mean()) if len(y) else None,
               coverage=float((pred != 3).mean()) if len(y) else None,
               balanced_accuracy=float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])) if len(y) else None,
               confusion=np.bincount(y * 4 + pred, minlength=12).reshape(3, 4).tolist(),
               class_clusters={str(c): int(frame.loc[frame.target.eq(c), "cluster_merged"].nunique()) for c in [0, 1, 2]})
    out["auroc"] = float(roc_auc_score(aedes, frame.score)) if aedes.any() and (~aedes).any() else None
    out["auprc"] = float(average_precision_score(aedes, frame.score)) if aedes.any() else None
    return out


def _interval(successes, total):
    if not total:
        return None
    return [float(beta.ppf(.025, successes, total - successes + 1)) if successes else 0.,
            float(beta.ppf(.975, successes + 1, total - successes)) if successes < total else 1.]


def _rank_metrics(y, score, weights):
    """Vectorised weighted AUROC/AP; ties handled as groups, not arbitrary ranks."""
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    w = weights[:, order]
    pos = np.add.reduceat(w * (y[order] == 2), starts, axis=1)
    neg = np.add.reduceat(w * (y[order] != 2), starts, axis=1)
    npos, nneg = pos.sum(1), neg.sum(1)
    auc_num = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = auc_num / (npos * nneg)
        p, n = pos[:, ::-1], neg[:, ::-1]
        precision = np.cumsum(p, axis=1) / np.cumsum(p + n, axis=1)
        ap = np.nansum(precision * p, axis=1) / npos
    return auc, ap


def with_uncertainty(frame, draws=2000, seed=20261025):
    frame = frame.reset_index(drop=True)
    result = {"point": point(frame)}
    if frame.empty:
        result.update(ci95={}, one_sided95={}, undefined_draws={k: draws for k in KEYS})
        return result
    y = frame.target.to_numpy(int)
    pred = frame.pred.to_numpy(int)
    score = frame.score.to_numpy(float)
    labels, cluster_id = np.unique(frame.cluster_merged, return_inverse=True)
    cluster_max = np.array([y[cluster_id == i].max() for i in range(len(labels))])
    strata = [np.flatnonzero(cluster_max == c) for c in np.unique(cluster_max)]
    rng = np.random.default_rng(seed)
    values = {k: [] for k in KEYS}
    alert = pred == 2
    for offset in range(0, draws, 64):
        size = min(64, draws - offset)
        sampled = np.concatenate([rng.choice(s, (size, len(s)), replace=True) for s in strata], axis=1)
        multiplicity = np.zeros((size, len(labels)), dtype=np.int16)
        for k, row in enumerate(sampled):
            multiplicity[k] = np.bincount(row, minlength=len(labels))
        weights = multiplicity[:, cluster_id].astype(float)
        counts = {c: weights[:, y == c].sum(1) for c in [0, 1, 2]}
        positives = weights[:, alert].sum(1)
        tp = weights[:, alert & (y == 2)].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values["aedes_precision"].extend(tp / positives)
            values["aedes_recall"].extend(tp / counts[2])
            values["fpr_other"].extend(weights[:, alert & (y == 1)].sum(1) / counts[1])
            values["fpr_noise"].extend(weights[:, alert & (y == 0)].sum(1) / counts[0])
            values["coverage"].extend(weights[:, pred != 3].sum(1) / weights.sum(1))
            recalls = np.stack([weights[:, (pred == c) & (y == c)].sum(1) / counts[c]
                                for c in np.unique(y)], axis=1)
            values["balanced_accuracy"].extend(np.nanmean(recalls, axis=1))
        auc, ap = _rank_metrics(y, score, weights)
        values["auroc"].extend(auc)
        values["auprc"].extend(ap)
    result["ci95"], result["one_sided95"], result["undefined_draws"] = {}, {}, {}
    for key, raw in values.items():
        raw = np.asarray(raw)
        finite = raw[np.isfinite(raw)]
        result["undefined_draws"][key] = int((~np.isfinite(raw)).sum())
        if len(finite):
            result["ci95"][key] = np.quantile(finite, [.025, .975]).tolist()
            result["one_sided95"][key] = np.quantile(finite, [.05, .95]).tolist()
    p = result["point"]
    result["file_binomial_ci95_descriptive_only"] = {
        "aedes_recall": _interval(p["aedes_tp"], p["aedes_n"]),
        "aedes_precision": _interval(p["aedes_tp"], p["alerts"]),
        "fpr_other": _interval(p["other_fp"], p["other_n"]),
        "fpr_noise": _interval(p["noise_fp"], p["noise_n"]),
    }
    result["bootstrap"] = dict(draws=draws, unit="cluster_merged", stratification="cluster maximum target",
                               fixed_fitted_models=True, includes_training_variability=False)
    result["cautions"] = []
    if p["aedes_n"] and p["class_clusters"]["2"] < 5:
        result["cautions"].append("fewer_than_5_Aedes_clusters")
    for name in ["other", "noise"]:
        if p[f"{name}_n"] and p[f"{name}_fp"] == 0:
            result["cautions"].append(f"zero_{name}_errors_bootstrap_cannot_bound_unseen_errors")
    return result


def success(mosquito, noise):
    """Never confuse point targets or a zero-error bootstrap with reliability."""
    m, n = mosquito["point"], noise["point"]
    observed = bool(m["aedes_recall"] is not None and m["aedes_recall"] >= .2
                    and m["fpr_other"] is not None and m["fpr_other"] <= .02
                    and n["fpr_noise"] is not None and n["fpr_noise"] <= .01)
    ms, ns = mosquito.get("one_sided95", {}), noise.get("one_sided95", {})
    supported = bool(observed and not mosquito.get("cautions") and not noise.get("cautions")
                     and ms.get("aedes_recall", [0])[0] >= .2
                     and ms.get("fpr_other", [1, 1])[1] <= .02
                     and ns.get("fpr_noise", [1, 1])[1] <= .01
                     and ms.get("aedes_precision", [0])[0] > m["prevalence"])
    return dict(observed_targets_met=observed, statistical_support=supported,
                field_validated=False, caution="Point targets and laboratory transfer are distinct from field reliability.")
