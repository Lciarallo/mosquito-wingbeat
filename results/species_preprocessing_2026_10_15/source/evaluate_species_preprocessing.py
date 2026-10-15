"""Retrospective, validation-selected preprocessing comparison for 20 species.

All frontends are applied during fitting as well as evaluation. The original
Torres run is preserved. No candidate is selected using new test predictions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from apply_torres_2025 import train_candidate, validation_loss, weights_for, evaluate
from species_preprocessing import SpeciesFrontend, VARIANTS
from torres_lstm import (SEED, TorresLSTM, aggregate_groups, classification_metrics,
                         digest, predict_probabilities, vote_statistics, selective_metrics, write_json)


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "results/torres_2025_2026_10_15/species"
OUTPUT = ROOT / "results/species_preprocessing_2026_10_15"
SOURCES = ["evaluate_species_preprocessing.py", "species_preprocessing.py",
           "apply_torres_2025.py", "torres_lstm.py"]
TRAINING = dict(epochs=80, lr=.0003, batch_size=128, patience=15, gradient_clip=3)


def read_json(path):
    return json.loads(Path(path).read_text())


def temporal_summary(x):
    return np.concatenate([x.mean(1), x.std(1)], axis=1)


def cached_features(frame, variant, device):
    frontend = SpeciesFrontend(variant).to(device).eval()
    indices = frame.wave_index.to_numpy(dtype=int)
    source_marker = ROOT / "data/cache/large_cnn/manifest.json"
    manifest = read_json(source_marker)
    if manifest["segments_sha256"] != digest(ROOT / "results/expanded/segments.csv"):
        raise ValueError("Waveform cache no longer matches source metadata.")
    identity = dict(indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                    waveform_manifest_sha256=digest(source_marker), frontend=frontend.config,
                    implementation_sha256=digest(ROOT / "species_preprocessing.py"),
                    mel_bank_implementation_sha256=digest(ROOT / "torres_lstm.py"))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/cache/species_preprocessing"
    cache.mkdir(parents=True, exist_ok=True)
    path, marker = cache / f"{key}.npy", cache / f"{key}.json"
    if not path.exists():
        waves = np.load(ROOT / "data/cache/large_cnn/waves.npy", mmap_mode="r")
        values = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(frame), 55, 64))
        with torch.inference_mode():
            for start in range(0, len(frame), 256):
                batch = torch.tensor(np.array(waves[indices[start:start + 256]]), device=device)
                values[start:start + len(batch)] = frontend(batch).cpu().numpy()
        values.flush()
        write_json(marker, dict(identity=identity, feature_sha256=digest(path)))
    metadata = read_json(marker)
    if metadata["identity"] != identity or metadata["feature_sha256"] != digest(path):
        raise ValueError("Preprocessing feature cache verification failed.")
    return np.load(path, mmap_mode="r"), dict(feature_path=str(path.relative_to(ROOT)),
        feature_sha256=metadata["feature_sha256"], **identity)


def save_predictions(directory, name, frame, probabilities, classes):
    groups, target, values = aggregate_groups(frame, probabilities)
    prediction = values.argmax(1)
    result = classification_metrics(target, prediction, classes)
    pd.DataFrame(dict(group=groups, target=target, truth=np.asarray(classes)[target],
                       prediction=prediction, predicted_label=np.asarray(classes)[prediction])).to_csv(
                           directory / f"{name}_recordings.csv", index=False)
    np.savez_compressed(directory / f"{name}_probabilities.npz", groups=groups.astype(str),
                        target=target, probabilities=values)
    write_json(directory / f"{name}_metrics.json", result)
    return result


def prepare_reference(reference, frame, test, classes, device, output):
    freeze, protocol = read_json(reference / "freeze.json"), read_json(reference / "protocol.json")
    if digest(reference / "bilstm.pt") != freeze["checkpoint_sha256"]["bilstm"]:
        raise ValueError("Original checkpoint has changed.")
    feature_path = ROOT / protocol["input_features"]
    if digest(feature_path) != protocol["feature_sha256"]:
        raise ValueError("Original features have changed.")
    if digest(reference / "scaler.npz") != freeze["scaler_sha256"]:
        raise ValueError("Original scaler has changed.")
    if digest(reference / "linear_baseline.joblib") != freeze["baseline_sha256"]:
        raise ValueError("Original linear model has changed.")
    with np.load(reference / "scaler.npz", allow_pickle=False) as scaler:
        values = ((np.load(feature_path, mmap_mode="r")[test] - scaler["mean"]) / scaler["std"]).astype("float32")
    linear = joblib.load(reference / "linear_baseline.joblib")
    linear_metric = save_predictions(output, "original_linear", frame.iloc[test],
                                      linear.predict_proba(temporal_summary(values)), classes)
    checkpoint = torch.load(reference / "bilstm.pt", map_location="cpu", weights_only=True)
    network = TorresLSTM(**checkpoint["config"]).to(device)
    network.load_state_dict(checkpoint["state_dict"])
    neural_metric = save_predictions(output, "original_bilstm", frame.iloc[test],
                                      predict_probabilities(network, values, device), classes)
    expected = read_json(reference / "metrics.json")
    assert linear_metric["confusion_matrix"] == expected["linear_baseline"]["confusion_matrix"]
    original = next(m for m in expected["neural_models"] if m["model"] == "bilstm")
    assert neural_metric["confusion_matrix"] == original["deterministic"]["confusion_matrix"]
    return linear_metric, neural_metric


def verify(output):
    output = Path(output)
    assert read_json(output / "complete.json")["complete"]
    protocol, frozen = read_json(output / "protocol.json"), read_json(output / "freeze.json")
    frame = pd.read_csv(output / "partitions.csv")
    for column in ["group", "pcm_block", "content_sha256"]:
        assert frame.groupby(column).split.nunique().max() == 1, column
    assert digest(output / "partitions.csv") == protocol["reference_partitions_sha256"]
    assert frozen["frozen_before_new_test_inference"]
    assert min(frozen["candidates"], key=lambda c: c["validation_recording_loss"])["id"] == frozen["selected_candidate"]
    neural = [c for c in frozen["candidates"] if c["model"] == "bilstm"]
    assert min(neural, key=lambda c: c["validation_recording_loss"])["id"] == frozen["selected_neural_candidate"]
    for source, sha in protocol["source_sha256"].items():
        assert digest(output / "source" / source) == sha
    metrics = read_json(output / "metrics.json")
    for row in metrics:
        with np.load(output / f"{row['id']}_probabilities.npz", allow_pickle=False) as data:
            assert len(data["target"]) == frame.loc[frame.split.eq("test"), "group"].nunique()
            actual = classification_metrics(data["target"], data["probabilities"].argmax(1), protocol["classes"])
            assert actual["confusion_matrix"] == row["test"]["confusion_matrix"]
            np.testing.assert_allclose(data["probabilities"].sum(-1), 1, atol=1e-5)
            assert set(data["groups"]) == set(frame.loc[frame.split.eq("test"), "group"])
    for variant in VARIANTS:
        directory = output / variant
        freeze = read_json(directory / "freeze.json")
        for name, sha in [("bilstm.pt", freeze["checkpoint_sha256"]["bilstm"]),
                          ("scaler.npz", freeze["scaler_sha256"]),
                          ("linear_baseline.joblib", freeze["baseline_sha256"])]:
            assert digest(directory / name) == sha
        p = read_json(directory / "protocol.json")
        x = np.load(ROOT / p["feature_path"], mmap_mode="r")
        assert digest(ROOT / p["feature_path"]) == p["feature_sha256"]
        tr = frame.split.eq("train").to_numpy()
        with np.load(directory / "scaler.npz", allow_pickle=False) as s:
            np.testing.assert_allclose(s["mean"], x[tr].mean((0, 1), keepdims=True), rtol=1e-5)
            np.testing.assert_allclose(s["std"], x[tr].std((0, 1), keepdims=True).clip(.001), rtol=1e-5)
    selected = frozen["selected_neural_candidate"].removesuffix("_bilstm")
    if selected != "original":
        directory = output / selected
        metric = read_json(directory / "bilstm_metrics.json")
        with np.load(directory / "bilstm_mc_group_probabilities.npz", allow_pickle=False) as data:
            assert len(data["probabilities"]) == 100
            stats = vote_statistics(data["probabilities"])
            for row in metric["selective"]:
                actual = selective_metrics(data["target"], stats["prediction"], stats["uncertainty"], protocol["classes"], row["threshold"])
                assert actual["confusion_with_abstention"] == row["confusion_with_abstention"]
    result = dict(complete=True, candidates_verified=len(metrics), test_groups=190,
        checks=["identical original partitions", "group/PCM/hash isolation", "source snapshots",
                "selection frozen before new test inference", "checkpoint/scaler hashes",
                "normalizers fitted on training rows only", "all saved confusion matrices", "selected neural MC coverage"])
    write_json(output / "verification.json", result)
    return result


def run(output, reference, device):
    if output.exists():
        raise FileExistsError("Existing results are preserved: " + str(output))
    output.mkdir(parents=True)
    (output / "source").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, output / "source" / name)
    torch.set_num_threads(4)
    frame = pd.read_csv(reference / "partitions.csv")
    for column in ["group", "pcm_block", "content_sha256"]:
        if frame.groupby(column).split.nunique().max() != 1:
            raise ValueError("Partition identity crosses splits: " + column)
    shutil.copy2(reference / "partitions.csv", output / "partitions.csv")
    classes = read_json(reference / "protocol.json")["classes"]
    tr, va, te = [np.flatnonzero(frame.split.eq(s)) for s in ["train", "validation", "test"]]
    old_freeze = read_json(reference / "freeze.json")
    candidates = [dict(id="original_linear", variant="original", model="linear",
                       validation_recording_loss=old_freeze["linear_baseline_validation_loss"]),
                  dict(id="original_bilstm", variant="original", model="bilstm",
                       validation_recording_loss=next(c["validation_recording_loss"] for c in old_freeze["candidates"] if c["model"] == "bilstm"))]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
        classes=classes, variants=list(VARIANTS), training=TRAINING,
        reference=str(reference.relative_to(ROOT)), reference_partitions_sha256=digest(reference / "partitions.csv"),
        reference_freeze_sha256=digest(reference / "freeze.json"),
        source_sha256={name: digest(ROOT / name) for name in SOURCES}, device=str(device),
        torch_version=str(torch.__version__), selection="minimum class-balanced recording validation log loss",
        test_already_known_retrospective=True, mosquitosong_used_for_training=False,
        feature_normalization="training band mean/std, std floor 0.001",
        architecture="Bi-LSTM: 2 layers x 256, dropout 0.2; original 32 inputs, new frontends 64",
        split_counts={s: dict(windows=len(g), groups=int(g.group.nunique()), pcm_blocks=int(g.pcm_block.nunique())) for s, g in frame.groupby("split")})
    write_json(output / "protocol.json", protocol)
    stored = {}
    for variant in VARIANTS:
        print(json.dumps(dict(starting=variant)), flush=True)
        directory = output / variant
        directory.mkdir()
        raw, metadata = cached_features(frame, variant, device)
        mean, std = raw[tr].mean((0, 1), keepdims=True), raw[tr].std((0, 1), keepdims=True).clip(.001)
        np.savez(directory / "scaler.npz", mean=mean, std=std)
        x = ((raw - mean) / std).astype("float32")
        metadata.update(task="20-species classification", classes=classes, training=TRAINING,
                        split_counts=protocol["split_counts"], selection=protocol["selection"], seed=SEED)
        write_json(directory / "protocol.json", metadata)
        summary = temporal_summary(x)
        linear = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=SEED))
        linear.fit(summary[tr], frame.target.iloc[tr], logisticregression__sample_weight=weights_for(frame.iloc[tr]))
        loss = validation_loss(frame.iloc[va], linear.predict_proba(summary[va]))
        joblib.dump(linear, directory / "linear_baseline.joblib")
        candidates.append(dict(id=f"{variant}_linear", variant=variant, model="linear", validation_recording_loss=loss))
        model, row = train_candidate("bilstm", frame, x, classes, device, directory, TRAINING)
        candidates.append(dict(id=f"{variant}_bilstm", variant=variant, **row))
        write_json(directory / "freeze.json", dict(selected_neural_model="bilstm",
            checkpoint_sha256={"bilstm": digest(directory / "bilstm.pt")},
            baseline_sha256=digest(directory / "linear_baseline.joblib"), scaler_sha256=digest(directory / "scaler.npz"),
            frozen_before_test_inference=True, candidates=[row]))
        stored[variant] = (x, linear, model)
        write_json(output / "validation_candidates.json", candidates)
    selected = min(candidates, key=lambda c: c["validation_recording_loss"])
    neural = min((c for c in candidates if c["model"] == "bilstm"), key=lambda c: c["validation_recording_loss"])
    frozen = dict(selected_candidate=selected["id"], selected_neural_candidate=neural["id"],
        candidates=candidates, frozen_utc=datetime.now(timezone.utc).isoformat(), frozen_before_new_test_inference=True,
        candidate_freeze_sha256={v: digest(output / v / "freeze.json") for v in VARIANTS})
    write_json(output / "freeze.json", frozen)
    # First evaluation of any new test predictions happens after this freeze.
    original_linear, original_neural = prepare_reference(reference, frame, te, classes, device, output)
    rows = [dict(**candidates[i], test=m) for i, m in enumerate([original_linear, original_neural])]
    for variant, (x, linear, model) in stored.items():
        linear_metric = save_predictions(output, f"{variant}_linear", frame.iloc[te],
                                          linear.predict_proba(temporal_summary(x[te])), classes)
        neural_metric = save_predictions(output, f"{variant}_bilstm", frame.iloc[te],
                                          predict_probabilities(model, x[te], device), classes)
        for name, metric in [("linear", linear_metric), ("bilstm", neural_metric)]:
            row = next(c for c in candidates if c["id"] == f"{variant}_{name}")
            rows.append(dict(**row, test=metric))
        print(json.dumps(dict(variant=variant, linear_accuracy=linear_metric["accuracy"],
                               bilstm_accuracy=neural_metric["accuracy"])), flush=True)
    write_json(output / "metrics.json", rows)
    pd.DataFrame([dict(id=r["id"], model=r["model"], variant=r["variant"],
        validation_loss=r["validation_recording_loss"], **{k:v for k,v in r["test"].items() if k != "confusion_matrix"}) for r in rows]).to_csv(output / "comparison.csv", index=False)
    if neural["variant"] != "original":
        variant = neural["variant"]
        x, _, model = stored[variant]
        evaluate(model, "bilstm", frame.iloc[te], x[te], classes, device, output / variant, passes=100)
    write_json(output / "complete.json", dict(complete=True, selected_candidate=selected["id"], selected_neural_candidate=neural["id"]))
    print(json.dumps(verify(output)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.verify_only.resolve()), indent=2))
    else:
        device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        run(args.output.resolve(), args.reference.resolve(), device)
