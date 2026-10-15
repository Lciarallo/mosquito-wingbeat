"""Arithmetic and grouped uncertainty for the frozen ternary experiment."""
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ternary_protocol import (
    CLASS_NAMES, SEED, digest, json_write, aggregate_groups, apply_policy,
    decision_scores, load_thresholds, verify_freeze,
)


def wilson_ci(k, n, confidence=0.95):
    if (not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer))
            or not 0 <= k <= n or not 0 < confidence < 1):
        raise ValueError("Invalid binomial counts/confidence.")
    result = {"confidence": confidence, "point": None, "ci_lower": None,
              "ci_upper": None, "interval": None, "estimable": False,
              "assumption": "independent_binomial_observations; descriptive_only_for_correlated_groups"}
    if n:
        z = NormalDist().inv_cdf((1 + confidence) / 2)
        p = k / n
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lo, hi = max(0.0, center - radius), min(1.0, center + radius)
        result.update(point=p, ci_lower=float(lo), ci_upper=float(hi),
                      interval=[float(lo), float(hi)], estimable=True)
    if confidence == 0.95:
        result["ci_95"] = result["interval"]
    return result


def canonical_source(value):
    value = str(value).lower()
    return {"dryad": "Dryad", "humbug": "HumBug", "mosquitosong": "MosquitoSong",
            "esc50": "ESC50"}.get(value, value if value not in {"", "nan", "none"} else "Unknown")


def matrix_metrics(cm):
    cm = np.asarray(cm, dtype=float)
    total, support, emitted = cm.sum(), cm.sum(1), cm.sum(0)
    rates = np.divide(np.diag(cm), support, out=np.full(3, np.nan), where=support > 0)
    return {
        "accuracy": float(np.trace(cm) / total) if total else None,
        "balanced_accuracy": float(np.nanmean(rates)) if total else None,
        "aedes_recall": float(cm[2, 2] / support[2]) if support[2] else None,
        "aedes_precision": float(cm[2, 2] / emitted[2]) if emitted[2] else None,
        "noise_to_aedes_fpr": float(cm[0, 2] / support[0]) if support[0] else None,
        "other_to_aedes_fpr": float(cm[1, 2] / support[1]) if support[1] else None,
        "noise_to_mosquito_fpr": float(cm[0, 1:].sum() / support[0]) if support[0] else None,
    }


def evaluate_predictions(y_true, pred, prob=None, sources=None):
    truth, guessed = np.asarray(y_true, dtype=int), np.asarray(pred, dtype=int)
    if truth.ndim != 1 or truth.shape != guessed.shape:
        raise ValueError("Misaligned classification vectors.")
    if not set(np.unique(np.r_[truth, guessed])).issubset({0, 1, 2}):
        raise ValueError("Unexpected target class.")
    cm = confusion_matrix(truth, guessed, labels=[0, 1, 2])
    support, emitted = cm.sum(1), cm.sum(0)
    normalized = np.divide(cm, support[:, None], out=np.zeros((3, 3)), where=support[:, None] > 0)
    metrics = matrix_metrics(cm)
    counts = {
        "aedes_tp": int(cm[2, 2]), "aedes_total": int(support[2]),
        "noise_fp_to_aedes": int(cm[0, 2]), "noise_total": int(support[0]),
        "other_fp_to_aedes": int(cm[1, 2]), "other_total": int(support[1]),
        "noise_fp_to_any_mosquito": int(cm[0, 1:].sum()),
    }
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c])
        den = int(support[c] + emitted[c])
        per_class[name] = {
            "precision": float(tp / emitted[c]) if emitted[c] else None,
            "recall": float(tp / support[c]) if support[c] else None,
            "f1": float(2 * tp / den) if den else None,
            "support": int(support[c]), "emitted": int(emitted[c]),
        }
    metrics.update(
        integer_counts=counts, confusion_matrix_raw=cm.tolist(),
        confusion_matrix_norm=normalized.tolist(), per_class=per_class,
        wilson_ci_95={
            "aedes_recall": wilson_ci(counts["aedes_tp"], counts["aedes_total"]),
            "noise_to_aedes_fpr": wilson_ci(counts["noise_fp_to_aedes"], counts["noise_total"]),
            "other_to_aedes_fpr": wilson_ci(counts["other_fp_to_aedes"], counts["other_total"]),
        },
    )
    if sources is not None:
        sources = np.asarray([canonical_source(s) for s in sources])
        if len(sources) != len(truth):
            raise ValueError("Source metadata is misaligned.")
        breakdown = {}
        for source in sorted(set(sources)):
            mask = sources == source
            sub = evaluate_predictions(truth[mask], guessed[mask])
            n = sub["integer_counts"]
            breakdown[source] = {
                "groups": int(mask.sum()), "metrics": sub,
                "aedes": {"total": n["aedes_total"], "tp": n["aedes_tp"],
                          "recall": sub["wilson_ci_95"]["aedes_recall"]},
                "other_mosquito": {"total": n["other_total"], "fp": n["other_fp_to_aedes"],
                                   "fpr": sub["wilson_ci_95"]["other_to_aedes_fpr"]},
                "no_mosquito": {"total": n["noise_total"], "fp": n["noise_fp_to_aedes"],
                                "fpr": sub["wilson_ci_95"]["noise_to_aedes_fpr"]},
            }
        for field in ["aedes_total", "noise_total", "other_total"]:
            if sum(s["metrics"]["integer_counts"][field] for s in breakdown.values()) != counts[field]:
                raise ValueError("Source denominators do not sum to the global denominator.")
        metrics["source_breakdown"] = breakdown
    return metrics


def cluster_intervals(groups, predictions, replicates=2000):
    """Stratified, paired bootstrap of whole audited PCM blocks, not windows."""
    truth = groups.true_label.to_numpy(dtype=int)
    block_ids, inverse = np.unique(groups.pcm_block.astype(str), return_inverse=True)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((replicates, len(block_ids)), dtype=np.int32)
    strata = {}
    for block in range(len(block_ids)):
        key = tuple(sorted(set(truth[inverse == block])))
        strata.setdefault(key, []).append(block)
    for indices in strata.values():
        n = len(indices)
        weights[:, indices] = rng.multinomial(n, np.full(n, 1 / n), size=replicates)
    result, draws = {}, {}
    for name, pred in predictions.items():
        cm = np.zeros((len(block_ids), 9), dtype=int)
        np.add.at(cm, (inverse, 3 * truth + np.asarray(pred)), 1)
        sampled = (weights @ cm).reshape(replicates, 3, 3)
        columns = {key: [] for key in matrix_metrics(sampled[0])}
        for item in sampled:
            for key, value in matrix_metrics(item).items():
                columns[key].append(np.nan if value is None else value)
        draws[name] = {key: np.asarray(values) for key, values in columns.items()}
        result[name] = {}
        for key, values in draws[name].items():
            valid = values[np.isfinite(values)]
            result[name][key] = {
                "interval": np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
                "valid_replicates": len(valid),
            }
    primary = "activity_gated_log_odds_cascade_event"
    paired = {}
    for reference in ["activity_gated_log_odds_no_temporal_event",
                      "hist_gradient_boosting_constrained_event", "ensemble_constrained_event"]:
        if primary not in draws or reference not in draws:
            continue
        paired[primary + "_minus_" + reference] = {}
        for key in draws[primary]:
            values = draws[primary][key] - draws[reference][key]
            valid = values[np.isfinite(values)]
            paired[primary + "_minus_" + reference][key] = {
                "interval": np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
                "valid_replicates": len(valid),
            }
    return {
        "method": "paired_percentile_bootstrap_stratified_by_block_label_set",
        "unit": "audited_pcm_block", "blocks": len(block_ids), "replicates": replicates,
        "seed": SEED, "stratum_block_counts": {str(k): len(v) for k, v in strata.items()},
        "limits": "Does not establish biological independence or undo prior corpus inspection. "
                  "Singleton strata and rare classes may yield degenerate intervals.",
        "models": result, "paired_differences": paired,
    }


def targets(metrics):
    c = metrics["integer_counts"]
    minimum = (50 * c["aedes_total"] + 63) // 64
    max_noise, max_other = 4 * c["noise_total"] // 426, 20 * c["other_total"] // 1014
    checks = {
        "aedes_recall": bool(c["aedes_total"] and c["aedes_tp"] >= minimum),
        "noise_fpr": bool(c["noise_total"] and c["noise_fp_to_aedes"] <= max_noise),
        "other_fpr": bool(c["other_total"] and c["other_fp_to_aedes"] <= max_other),
    }
    return {"passed_all": all(checks.values()), "checks": checks,
            "observed": c, "required": {"minimum_aedes_tp": minimum,
                                        "maximum_noise_fp": max_noise, "maximum_other_fp": max_other},
            "scope": "sample acceptance criterion; no population reliability claim"}


def verify_evaluation(out):
    out = Path(out)
    freeze = verify_freeze(out)
    record = json.loads((out / "evaluation_complete.json").read_text())
    if record["run_id"] != freeze["run_id"] or record["freeze_sha256"] != digest(out / "freeze.json"):
        raise ValueError("Evaluation belongs to a different freeze.")
    for name, expected in record["outputs_sha256"].items():
        if digest(out / name) != expected:
            raise ValueError("Evaluation output changed: " + name)
    return record


def evaluate_run(out):
    out = Path(out)
    thresholds = load_thresholds(out)
    protocol = json.loads((out / "protocol.json").read_text())
    if (out / "event_level_metrics.json").exists():
        raise FileExistsError("Evaluation metrics already exist; do not overwrite them.")
    frame = pd.read_csv(out / "predictions_oof.csv")
    required = {"row_id", "path", "group", "pcm_block", "domain", "start_s",
                "window_duration_s", "true_label", "fold"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Incomplete prediction metadata.")
    if frame.row_id.duplicated().any() or frame.groupby("pcm_block").fold.nunique().max() != 1:
        raise ValueError("Duplicate predictions or PCM fold leakage.")
    partitions = pd.read_csv(out / "partitions.csv.gz")
    expected = partitions.loc[partitions.role.eq("test"), ["row_id", "fold"]].sort_values("row_id")
    actual = frame[["row_id", "fold"]].sort_values("row_id")
    if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
        raise ValueError("Prediction folds do not match the frozen outer partitions.")
    for k in range(protocol["outer_splits"]):
        item = thresholds["folds"][f"fold_{k}"]
        if set(item["test_row_ids"]) != set(frame.loc[frame.fold.eq(k), "row_id"]):
            raise ValueError("Prediction rows do not match inner-threshold lineage.")
    lr = frame[[f"lr_prob_{c}" for c in range(3)]].to_numpy()
    hgb = frame[[f"hgb_prob_{c}" for c in range(3)]].to_numpy()
    groups = aggregate_groups(frame, lr, hgb, protocol["temporal"])
    predictions = {}
    for name, family in [("logistic_regression_event", "lr"),
                         ("hist_gradient_boosting_event", "hgb"), ("ensemble_event", "ensemble")]:
        predictions[name] = groups[[f"{family}_simple_{c}" for c in range(3)]].to_numpy().argmax(1)
    for name in protocol["policy_names"]:
        pred = np.full(len(groups), -1)
        for k in range(protocol["outer_splits"]):
            mask = groups.fold.eq(k)
            config = thresholds["folds"][f"fold_{k}"]["policies"][name]
            pred[mask] = apply_policy(groups.loc[mask], config)
            p0, score = decision_scores(groups.loc[mask], config)
            groups.loc[mask, name + "_noise_score"] = p0
            groups.loc[mask, name + "_aedes_score"] = score
        if (pred < 0).any():
            raise ValueError("Unassigned group predictions.")
        predictions[name] = pred
    metrics = {}
    for name, pred in predictions.items():
        groups[name + "_pred"] = pred
        metrics[name] = evaluate_predictions(groups.true_label, pred, sources=groups.source)
    uncertainty = cluster_intervals(groups, predictions)
    for name in metrics:
        metrics[name]["pcm_block_bootstrap_95"] = uncertainty["models"][name]
    groups.to_csv(out / "event_predictions.csv", index=False)
    payload = {
        "run_id": protocol["run_id"], "evaluation_unit": "group",
        "event_summary": {"total_groups": len(groups),
                          "groups_per_class": {CLASS_NAMES[c]: int(groups.true_label.eq(c).sum()) for c in range(3)},
                          "pcm_blocks": int(groups.pcm_block.nunique()),
                          "persistence_available_groups": int(groups.persistence_available.sum())},
        "models": metrics, "uncertainty": uncertainty,
        "no_population_error_guarantee": True,
    }
    json_write(out / "event_level_metrics.json", payload)
    json_write(out / "window_metrics.json", {
        name: evaluate_predictions(frame.true_label, frame[f"{prefix}_pred"])
        for name, prefix in [("logistic_regression", "lr"),
                             ("hist_gradient_boosting", "hgb"), ("ensemble", "ensemble")]
    })
    json_write(out / "operational_targets.json", {name: targets(item) for name, item in metrics.items()})
    return payload
