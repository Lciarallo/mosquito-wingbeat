"""Evaluate a frozen ternary run, or inspect historical metrics explicitly."""
import argparse
import json
from pathlib import Path

import numpy as np

from ternary_protocol import (
    ROOT, CLASS_NAMES, TEMPORAL, activity_pool, log_pool, persistence, load_thresholds,
)
from ternary_reporting import (
    wilson_ci, canonical_source, evaluate_predictions, evaluate_run, verify_evaluation,
)


def softmax(z):
    z = np.asarray(z, dtype=float)
    if not z.size or not np.isfinite(z).all():
        raise ValueError("Invalid softmax input.")
    p = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def logarithmic_opinion_pool(p_lr, p_hgb, w=0.6, eps=1e-12):
    return log_pool(p_lr, p_hgb, w)


activity_gated_pooling = activity_pool


def audit_denominators(frame):
    if not set(frame.true_label.unique()).issubset({0, 1, 2}):
        raise ValueError("Unexpected class.")
    labels = frame.groupby("group").true_label.max()
    return {
        "n_windows_per_class": {c: int(frame.true_label.eq(c).sum()) for c in range(3)},
        "n_recordings_touching_class": {c: int(frame.loc[frame.true_label.eq(c), "group"].nunique())
                                       for c in range(3)},
        "n_pure_events_per_class": {c: int(labels.eq(c).sum()) for c in range(3)},
        "evaluation_unit": "group; not a claim of independent biological events",
    }, labels


def compute_group_wav_bounded_persistence(sub_df, p2_col="ensemble_prob_2", **settings):
    return persistence(sub_df, p2_col, **(TEMPORAL | settings))["value"]


def get_canonical_source(domain_val, path_val=""):
    # A path is not a substitute for missing source provenance.
    return canonical_source(domain_val)


def calibrate_neyman_pearson_oof(rec_df):
    raise ValueError(
        "Calibration from global outer OOF predictions is disabled. "
        "Generate inner thresholds with train_ternary_nested_models.py."
    )


def latest_verified_run():
    candidates = [p.parent for p in (ROOT / "results").glob("ternary_verified_*/freeze.json")]
    if not candidates:
        raise FileNotFoundError("No verified run exists; train with --out results/ternary_verified_<id>.")
    return max(candidates, key=lambda p: (p / "freeze.json").stat().st_mtime_ns)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--nested", action="store_true", help="Select latest verified run.")
    parser.add_argument("--legacy", action="store_true", help="Read historical metrics without rewriting them.")
    args = parser.parse_args(argv)
    if args.legacy:
        if args.nested:
            parser.error("--legacy and --nested are mutually exclusive.")
        out = args.results_dir or ROOT / "results/ternary_detection_2026_10_14"
        payload = json.loads((out / "event_level_metrics.json").read_text())
        print(json.dumps({"scope": "historical_unverified_read_only", "metrics": payload},
                         ensure_ascii=False, indent=2))
        return payload
    out = args.results_dir or latest_verified_run()
    # Check mandatory thresholds before any possible write or attempt to recover.
    load_thresholds(out)
    if (out / "evaluation_complete.json").exists():
        verify_evaluation(out)
        payload = json.loads((out / "event_level_metrics.json").read_text())
    elif not (out / "evaluation_started.json").exists():
        from train_ternary_nested_models import evaluate
        payload = evaluate(out)
    else:
        raise FileExistsError("Incomplete prior evaluation; inspect its failure instead of overwriting it.")
    for name, item in payload["models"].items():
        counts = item["integer_counts"]
        print(name, counts, "precision", item["aedes_precision"], flush=True)
    return payload


if __name__ == "__main__":
    main()
