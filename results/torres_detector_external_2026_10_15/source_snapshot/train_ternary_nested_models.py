"""Audited nested ternary training. Historical runs are never overwritten."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ternary_protocol import (
    ROOT, SEGMENTS, FEATURES, KNOWN_MAP, CLASS_NAMES, SEED, TEMPORAL, MODEL_CONFIG,
    now, digest, json_write, load_dataset, audit_pcm, sample_weights, check_partition,
    aggregate_groups, calibrate_policy, apply_policy, verify_freeze,
)

DATA_CSV, DATA_FEAT, COLLISION_MAP_CSV = SEGMENTS, FEATURES, KNOWN_MAP
POLICY_NAMES = [
    "activity_gated_log_odds_cascade_event",
    "activity_gated_log_odds_no_temporal_event",
    "hist_gradient_boosting_constrained_event",
    "ensemble_constrained_event",
]
CODE_FILES = [
    "ternary_protocol.py", "ternary_reporting.py", "train_ternary_nested_models.py",
    "evaluate_ternary_event_pipeline.py", "predict_ternary.py", "fetch_ternary_fresh.py", "fetch_humbug.py",
    "aedes_classical.py", "aedes_binary.py", "bioacoustic_invariance.py",
]


def load_and_filter_data():
    return load_dataset(DATA_CSV, DATA_FEAT, COLLISION_MAP_CSV)


compute_sample_weights = sample_weights


def fit_pair(frame, x, y, gamma, config=MODEL_CONFIG):
    """Only the provided fit partition is visible to weights, scaler and estimators."""
    weights = sample_weights(frame, y, gamma)
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(**config["lr"]).fit(scaler.transform(x), y, sample_weight=weights)
    hgb = HistGradientBoostingClassifier(**config["hgb"]).fit(x, y, sample_weight=weights)
    if not np.array_equal(lr.classes_, [0, 1, 2]) or not np.array_equal(hgb.classes_, [0, 1, 2]):
        raise ValueError("Incomplete model class order.")
    return {"scaler": scaler, "lr": lr, "hgb": hgb, "gamma": gamma,
            "config": config, "hgb_iterations": int(hgb.n_iter_),
            "lr_iterations": lr.n_iter_.astype(int).tolist(),
            "fit_rows": frame.row_id.astype(int).tolist(),
            "fit_pcm_blocks": sorted(frame.pcm_block.unique()),
            "weight_class_mass": np.bincount(y, weights=weights).tolist(),
            "scaler_mean": scaler.mean_.tolist()}


def predict_pair(bundle, x):
    lr = bundle["lr"].predict_proba(bundle["scaler"].transform(x))
    hgb = bundle["hgb"].predict_proba(x)
    for prob in [lr, hgb]:
        if not np.isfinite(prob).all() or (prob < 0).any() or not np.allclose(prob.sum(1), 1):
            raise ValueError("Invalid model probabilities.")
    return lr, hgb


def save_pair(bundle, path):
    joblib.dump(bundle, path, compress=3)
    json_write(Path(path).with_suffix(".json"), {
        key: bundle[key] for key in
        ["gamma", "config", "hgb_iterations", "lr_iterations", "fit_rows",
         "fit_pcm_blocks", "weight_class_mass", "scaler_mean"]
    })


def policy_set(groups):
    return {
        POLICY_NAMES[0]: calibrate_policy(groups, "log_pool", temporal=True),
        POLICY_NAMES[1]: calibrate_policy(groups, "log_pool", temporal=False),
        POLICY_NAMES[2]: calibrate_policy(groups, "hgb", temporal=False),
        POLICY_NAMES[3]: calibrate_policy(groups, "ensemble", temporal=False),
    }


def predict_table(frame, lr, hgb):
    columns = [
        "row_id", "path", "group", "pcm_block", "domain", "start_s", "duration_s",
        "window_duration_s", "declared_duration_s", "duration_origin", "bucket", "true_label",
    ]
    answer = frame[[name for name in columns if name in frame]].copy()
    for name, p in [("lr", lr), ("hgb", hgb), ("ensemble", (lr + hgb) / 2)]:
        for c in range(3):
            answer[f"{name}_prob_{c}"] = p[:, c]
        answer[f"{name}_pred"] = p.argmax(1)
    return answer


def choose_with_inner_cv(frame, x, y, out, context, *, inner_splits=4, gammas=(0.5, 0.75)):
    """Produce and save only inner predictions before choosing any outer policy."""
    folder = Path(out) / "inner" / context
    folder.mkdir(parents=True, exist_ok=False)
    cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
    splits = list(cv.split(x, y, frame.pcm_block))
    partition_rows, audit_rows = [], []
    for fold, (train, valid) in enumerate(splits):
        check = check_partition(frame, train, valid)
        audit_rows.append(dict(inner_fold=fold, **check))
        for role, indices in [("fit", train), ("validation", valid)]:
            rows = frame.iloc[indices][["row_id", "group", "path", "pcm_block"]].copy()
            rows["inner_fold"], rows["role"] = fold, role
            partition_rows.append(rows)
    pd.concat(partition_rows).to_csv(folder / "partitions.csv.gz", index=False)
    json_write(folder / "partition_audit.json", audit_rows)
    best, best_key, candidates = None, None, []
    for gamma in gammas:
        lr, hgb = np.zeros((len(frame), 3)), np.zeros((len(frame), 3))
        prefix = f"gamma_{gamma:.2f}"
        fit_log = []
        for fold, (train, valid) in enumerate(splits):
            fit = frame.iloc[train].reset_index(drop=True)
            model = fit_pair(fit, x[train], y[train], gamma)
            lr[valid], hgb[valid] = predict_pair(model, x[valid])
            model_path = folder / f"{prefix}_fold_{fold}.joblib"
            save_pair(model, model_path)
            fit_log.append({"inner_fold": fold, "model_sha256": digest(model_path),
                            "fit_rows": model["fit_rows"], "hgb_iterations": model["hgb_iterations"]})
            print(context, "gamma", gamma, "inner", fold + 1, "/", inner_splits, flush=True)
        table = predict_table(frame, lr, hgb)
        table.to_csv(folder / f"{prefix}_predictions.csv.gz", index=False)
        groups = aggregate_groups(frame, lr, hgb)
        groups.to_csv(folder / f"{prefix}_groups.csv", index=False)
        policies = policy_set(groups)
        policy = policies[POLICY_NAMES[0]]
        truth = groups.true_label.to_numpy()
        guess = apply_policy(groups, policy)
        tp = int(((truth == 2) & (guess == 2)).sum())
        fp = int(((truth != 2) & (guess == 2)).sum())
        key = (tp, -fp, -policy["temporal_weight"], -policy["blend_weight_lr"], -gamma)
        item = {"gamma": gamma, "selection_key": list(key), "policies": policies,
                "inner_group_count": len(groups), "fits": fit_log}
        candidates.append(item)
        if best_key is None or key > best_key:
            best, best_key = item, key
    selection = {"context": context, "selected_gamma": best["gamma"],
                 "policies": best["policies"], "candidates": candidates,
                 "objective": "group Aedes recall under fixed empirical FPR constraints; "
                              "ties: fewer false calls, simpler temporal/fusion rule, smaller gamma",
                 "selection_has_no_outer_test_predictions": True}
    json_write(folder / "selection.json", selection)
    return selection


def train(out, *, outer_splits=5, inner_splits=4, threads=4):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty experiment directory; completed/partial runs are immutable.")
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name
    json_write(out / "training_started.json", {"run_id": run_id, "started_utc": now()})
    frame, x, y, stats = load_and_filter_data()
    frame, pcm_audit = audit_pcm(frame, out, COLLISION_MAP_CSV)
    raw_sources = pd.read_csv(out / "source_pcm_audit.csv")
    stats["input_sha256"].update(dict(zip(raw_sources.path, raw_sources.file_sha256)))
    frame.to_csv(out / "segments.csv.gz", index=False)
    stats["unique_pcm_blocks"] = int(frame.pcm_block.nunique())
    json_write(out / "dataset_statistics.json", stats)
    code_hashes = {name: digest(ROOT / name) for name in CODE_FILES}
    protocol = {
        "schema_version": 1, "run_id": run_id, "created_utc": now(), "seed": SEED,
        "outer_splits": outer_splits, "inner_splits": inner_splits, "gammas": [0.5, 0.75],
        "model_config": MODEL_CONFIG, "temporal": TEMPORAL, "policy_names": POLICY_NAMES,
        "empirical_calibration_fpr": {"noise": 0.0075, "other": 0.0175},
        "no_population_error_guarantee": True, "evaluation_unit": "original_group",
        "resampling_unit": "audited_pcm_block", "pcm_audit_limits": pcm_audit["limits"],
        "final_model_scope": "all development data; requires new independent evaluation",
        "postfreeze_acquisition": {
            "source": "https://zenodo.org/records/19095788", "target": "Aedes aegypti",
            "archive_index_sha256": digest(ROOT / "tmp/reproductions_2026_10_01/cdmsc_central_full.json"),
            "rule": "first two unused numeric S_1_D_3 clips, >=1.984 real seconds, "
                    "exclude known bytes/declared PCM collisions without using predictions",
            "not_a_new_biological_campaign": True,
        },
        "code_sha256": code_hashes, "input_sha256": stats["input_sha256"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "threads": threads},
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    json_write(out / "protocol.json", protocol)
    outer_cv = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    outer = list(outer_cv.split(x, y, frame.pcm_block))
    partitions, partition_audits, thresholds = [], [], {}
    (out / "models").mkdir()
    with threadpool_limits(limits=threads):
        for k, (fit, test) in enumerate(outer):
            partition_audits.append(dict(outer_fold=k, **check_partition(frame, fit, test)))
            for role, ix in [("fit", fit), ("test", test)]:
                rows = frame.iloc[ix][["row_id", "group", "path", "pcm_block"]].copy()
                rows["fold"], rows["role"] = k, role
                partitions.append(rows)
            selection = choose_with_inner_cv(frame.iloc[fit].reset_index(drop=True), x[fit], y[fit],
                                             out, f"outer_{k}", inner_splits=inner_splits)
            model = fit_pair(frame.iloc[fit], x[fit], y[fit], selection["selected_gamma"])
            save_pair(model, out / "models" / f"outer_{k}.joblib")
            thresholds[f"fold_{k}"] = {
                "selected_gamma": selection["selected_gamma"],
                "train_folds": [j for j in range(outer_splits) if j != k],
                "policies": selection["policies"],
                "selection_sha256": digest(out / "inner" / f"outer_{k}" / "selection.json"),
                "fit_row_ids": frame.iloc[fit].row_id.astype(int).tolist(),
                "test_row_ids": frame.iloc[test].row_id.astype(int).tolist(),
            }
            print("FROZEN CANDIDATE outer", k, "without outer predictions", flush=True)
        final = choose_with_inner_cv(frame, x, y, out, "final", inner_splits=inner_splits)
        model = fit_pair(frame, x, y, final["selected_gamma"])
        model["policies"], model["temporal"] = final["policies"], TEMPORAL
        model["run_id"] = run_id
        save_pair(model, out / "models" / "final.joblib")
    pd.concat(partitions).to_csv(out / "partitions.csv.gz", index=False)
    json_write(out / "partition_audit.json", partition_audits)
    json_write(out / "inner_thresholds.json", {"schema_version": 1, "run_id": run_id, "folds": thresholds})
    json_write(out / "final_selection.json", final)
    if code_hashes != {name: digest(ROOT / name) for name in CODE_FILES}:
        raise ValueError("Code changed during training; freeze refused.")
    artifacts = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    freeze = {
        "schema_version": 1, "run_id": run_id, "created_utc": now(),
        "no_outer_predictions_yet": True, "all_outer_models_and_final_frozen": True,
        "artifacts_sha256": artifacts, "code_sha256": code_hashes,
        "input_sha256": stats["input_sha256"],
    }
    json_write(out / "freeze.json", freeze)
    verify_freeze(out, inputs=True)
    print("TRAINING COMPLETE; all selection/models frozen:", out, flush=True)
    return out


def evaluate(out, *, threads=4):
    out = Path(out).resolve()
    if (out / "evaluation_started.json").exists():
        raise FileExistsError("This evaluation was already started; do not overwrite its results.")
    freeze = verify_freeze(out, inputs=True)
    from ternary_protocol import load_thresholds
    load_thresholds(out)
    protocol = json.loads((out / "protocol.json").read_text())
    frame = pd.read_csv(out / "segments.csv.gz", low_memory=False)
    x = np.asarray(np.load(FEATURES, mmap_mode="r")[frame.row_id.to_numpy(dtype=int)], dtype="float32")
    partitions = pd.read_csv(out / "partitions.csv.gz")
    json_write(out / "evaluation_started.json", {
        "started_utc": now(), "run_id": freeze["run_id"], "freeze_sha256": digest(out / "freeze.json"),
    })
    lr, hgb, folds = np.zeros((len(frame), 3)), np.zeros((len(frame), 3)), np.full(len(frame), -1)
    with threadpool_limits(limits=threads):
        for k in range(protocol["outer_splits"]):
            test_ids = partitions.loc[(partitions.fold == k) & partitions.role.eq("test"), "row_id"]
            ix = np.flatnonzero(frame.row_id.isin(test_ids))
            bundle = joblib.load(out / "models" / f"outer_{k}.joblib")
            if set(bundle["fit_rows"]) & set(test_ids):
                raise ValueError("Persisted model includes outer test rows.")
            if (folds[ix] != -1).any():
                raise ValueError("An outer row was predicted more than once.")
            lr[ix], hgb[ix] = predict_pair(bundle, x[ix])
            folds[ix] = k
    if (folds < 0).any():
        raise ValueError("An outer row has no test prediction.")
    table = predict_table(frame, lr, hgb)
    table["fold"] = folds
    table.to_csv(out / "predictions_oof.csv", index=False)
    from ternary_reporting import evaluate_run
    result = evaluate_run(out)
    verify_freeze(out)
    outputs = ["predictions_oof.csv", "event_predictions.csv", "event_level_metrics.json",
               "window_metrics.json", "operational_targets.json"]
    json_write(out / "evaluation_complete.json", {
        "completed_utc": now(), "run_id": freeze["run_id"],
        "freeze_sha256": digest(out / "freeze.json"), "outer_predictions_after_freeze": True,
        "outputs_sha256": {name: digest(out / name) for name in outputs},
        "historical_data_not_independent_confirmation": True,
    })
    print("EVALUATION COMPLETE:", out, flush=True)
    return result


def run_nested_training(out=None):
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
        out = ROOT / "results" / ("ternary_verified_" + stamp)
    train(out)
    return evaluate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.phase in {"train", "all"}:
        train(args.out, threads=args.threads)
    if args.phase in {"evaluate", "all"}:
        evaluate(args.out, threads=args.threads)


if __name__ == "__main__":
    main()
