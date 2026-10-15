"""Offline inference with a verified ternary development model; no biological certification."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ternary_protocol import (
    CLASS_NAMES, read_audio, aggregate_groups, apply_policy, verify_freeze, json_write,
)
from train_ternary_nested_models import predict_pair, POLICY_NAMES


def audio_features(path):
    from aedes_classical import features
    from bioacoustic_invariance import extract_batch_bioacoustic_features
    wave = read_audio(path, 16000)
    size = 15872 if len(wave) >= 15872 else 4800
    if len(wave) < size:
        raise ValueError("At least 0.3 s of real audio is required; no padding or repetition.")
    n = len(wave) // size
    windows = wave[:n * size].reshape(n, size)
    pcm = np.rint(np.clip(windows, -1, 32767 / 32768) * 32768).astype("<i2").astype("int32")
    quality = (np.abs(pcm).max(1) >= 32) & ((np.abs(pcm) >= 32700).sum(1) * 100 <= size)
    x = np.concatenate([features(windows), extract_batch_bioacoustic_features(windows)], axis=1)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite inference features.")
    frame = pd.DataFrame({
        "row_id": np.arange(n), "path": str(Path(path).resolve()), "group": "input_audio",
        "domain": "inference", "pcm_block": "input_audio",
        "start_s": np.arange(n) * size / 16000, "window_duration_s": size / 16000,
        "duration_s": size / 16000, "quality_ok": quality,
    })
    return frame, x, {"real_seconds": len(wave) / 16000, "windows": n,
                      "ignored_tail_seconds": (len(wave) - n * size) / 16000,
                      "quality_invalid_windows": int((~quality).sum()),
                      "padding_repetition_concatenation": False}


def predict_features(bundle, frame, x):
    lr, hgb = predict_pair(bundle, x)
    groups = aggregate_groups(frame, lr, hgb, bundle["temporal"])
    answer = {name: int(apply_policy(groups, rule)[0]) for name, rule in bundle["policies"].items()}
    return answer, groups, lr, hgb


def predict_audio(run, audio, output=None):
    run = Path(run)
    freeze = verify_freeze(run)
    with threadpool_limits(limits=4):
        bundle = joblib.load(run / "models/final.joblib")
        if bundle.get("run_id") != freeze["run_id"]:
            raise ValueError("Inference model belongs to a different run.")
        frame, x, details = audio_features(audio)
        valid = frame.quality_ok.to_numpy()
        if valid.any():
            classes, groups, lr, hgb = predict_features(bundle, frame.loc[valid], x[valid])
            for prefix, prob in [("lr", lr), ("hgb", hgb)]:
                for c in range(3):
                    frame.loc[valid, f"{prefix}_prob_{c}"] = prob[:, c]
            primary = classes[POLICY_NAMES[0]]
            decision = CLASS_NAMES[primary]
            temporal = {"available": bool(groups.persistence_available.iloc[0]),
                        "conditional_score": (None if pd.isna(groups.persistence_q.iloc[0])
                                              else float(groups.persistence_q.iloc[0]))}
        else:
            classes, primary, decision = {}, None, "invalid_audio"
            temporal = {"available": False, "conditional_score": None}
    summary = {
        "run_id": freeze["run_id"], "audio": str(Path(audio).resolve()),
        "decision": decision, "predicted_class": primary, "policies": classes,
        "temporal": temporal, **details, "experimental": True,
        "biological_confirmation": False, "independent_field_validation": False,
        "reference_denominator_includes_invalid_audio": True,
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        frame.to_csv(output / "window_predictions.csv", index=False)
        json_write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict_audio(args.run, args.audio, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
