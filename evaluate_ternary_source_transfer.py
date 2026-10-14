"""Retrospective evaluation of an explicitly excluded source, without retuning."""
import argparse
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ternary_protocol import (
    ROOT, SEGMENTS, FEATURES, digest, json_write, now, aggregate_groups,
    apply_policy, pcm_signatures, read_audio, verify_freeze,
)
from ternary_reporting import evaluate_predictions, verify_evaluation
from train_ternary_nested_models import predict_pair


def evaluate_source(run, source):
    run = Path(run).resolve()
    freeze = verify_freeze(run, inputs=True)
    verify_evaluation(run)
    protocol = json.loads((run / "protocol.json").read_text())
    base = pd.read_csv(run / "segments.csv.gz", low_memory=False)
    if source in set(base.domain):
        raise ValueError("This source was in development; it cannot be relabelled as a held-out source.")
    table = pd.read_csv(SEGMENTS, low_memory=False)
    raw = np.load(FEATURES, mmap_mode="r")
    quality = table.quality_ok.isin([True, "True", "true", 1, "1"])
    mask = table.domain.eq(source) & quality & np.isfinite(raw).all(axis=1)
    if not mask.any():
        raise ValueError("No valid labelled windows for this source.")
    frame = table.loc[mask].copy()
    frame["row_id"] = np.flatnonzero(mask)
    frame["window_duration_s"] = frame.bucket.map({"long": 0.992, "short": 0.3})
    if frame.window_duration_s.isna().any():
        raise ValueError("Unknown window duration for excluded source.")
    frame["duration_s"] = frame.window_duration_s
    frame["true_label"] = np.where(frame.label.eq("Aedes aegypti"), 2,
                                   np.where(frame.label.eq("noise"), 0, 1))
    frame["pcm_block"] = "source_transfer:" + frame.group
    x = np.asarray(raw[mask.to_numpy()], dtype="float32")
    out = run / ("source_transfer_" + source)
    out.mkdir(exist_ok=False)
    with gzip.open(run / "pcm_signatures.json.gz", "rt") as stream:
        known = {key for keys in json.load(stream).values() for key in keys}
    bytehashes = set(pd.read_csv(run / "source_pcm_audit.csv").file_sha256)
    overlaps = []
    for number, name in enumerate(sorted(frame.path.unique()), 1):
        path = ROOT / name
        overlaps.append({"path": name, "file_sha256": digest(path),
                         "training_byte_duplicate": digest(path) in bytehashes,
                         "training_pcm_matches": len(pcm_signatures(read_audio(path, 8000)) & known)})
        if number % 200 == 0:
            print("SOURCE TRANSFER AUDIT", number, flush=True)
    json_write(out / "overlap_audit.json", {
        "source": source, "files": overlaps,
        "any_detected_overlap": any(r["training_byte_duplicate"] or r["training_pcm_matches"] for r in overlaps),
        "limits": "Declared exact/aligned PCM check; no proof of biological independence.",
    })
    with threadpool_limits(limits=4):
        bundle = joblib.load(run / "models/final.joblib")
        lr, hgb = predict_pair(bundle, x)
        groups = aggregate_groups(frame, lr, hgb, protocol["temporal"])
        metrics = {}
        for name, config in bundle["policies"].items():
            pred = apply_policy(groups, config)
            groups[name + "_pred"] = pred
            metrics[name] = evaluate_predictions(groups.true_label, pred, sources=groups.source)
    groups.to_csv(out / "event_predictions.csv", index=False)
    json_write(out / "metrics.json", {
        "completed_utc": now(), "source": source, "scope": "retrospective_source_transfer",
        "groups": len(groups), "windows": len(frame), "models": metrics,
        "model_refitted": False, "thresholds_retuned": False,
        "familiar_previously_inspected_source": True, "independent_confirmation": False,
        "freeze_sha256": digest(run / "freeze.json"), "evaluation_code_sha256": digest(Path(__file__)),
    })
    verify_freeze(run)
    print("SOURCE TRANSFER COMPLETE", source, flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    evaluate_source(args.run, args.source)


if __name__ == "__main__":
    main()
