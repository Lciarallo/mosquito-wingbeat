"""Reproducible audio stress test of frozen ternary models, without fitting.

Species references come from labelled donor clips. Procedural sounds are
negative engineering controls, never species-labelled oscillators.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, chirp
import soundfile as sf
from threadpoolctl import threadpool_limits

from ternary_protocol import (
    ROOT, digest, json_write, now, read_audio, aggregate_groups, apply_policy,
    verify_freeze, load_thresholds,
)
from ternary_reporting import verify_evaluation
from train_ternary_nested_models import predict_pair, POLICY_NAMES

SR, SIZE, SEED = 16000, 15872, 20261015
PRIMARY = POLICY_NAMES[0]
CONDITIONS = ["clean_control", "white_snr20", "pink_snr10", "pink_snr0",
              "pink_snr_minus5", "attenuation18db", "lowpass1200", "passby_envelope"]
PROFILES = ["white", "pink", "brown", "mains_harmonics", "fan_harmonics",
            "chirp", "clicks", "modulated_fan"]


def generator(key, seed=SEED):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little")
    return np.random.default_rng(value)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def coloured_noise(n, rng, colour="pink"):
    wave = rng.normal(size=n)
    if colour != "white":
        spectrum = np.fft.rfft(wave)
        frequency = np.fft.rfftfreq(n, 1 / SR)
        scale = np.zeros_like(frequency)
        scale[1:] = frequency[1:] ** (-0.5 if colour == "pink" else -1)
        wave = np.fft.irfft(spectrum * scale, n=n)
    wave -= wave.mean()
    return wave / max(rms(wave), 1e-12)


def mix_at_ratio(signal, interference, db):
    """Ratio is full donor-clip RMS / added-component RMS, not isolated wings."""
    signal, interference = np.asarray(signal, float), np.asarray(interference, float)
    if signal.shape != interference.shape or rms(signal) <= 1e-10 or rms(interference) <= 1e-10:
        raise ValueError("Aligned, nonempty, nonzero components are required.")
    component = interference * (rms(signal) / rms(interference)) * 10 ** (-db / 20)
    result = signal + component
    scale = min(1.0, 0.95 / max(float(np.abs(result).max()), 1e-12))
    return result * scale, 20 * np.log10(rms(signal * scale) / rms(component * scale))


def transform(wave, condition, rng):
    measured = None
    if condition == "clean_control":
        result = wave.copy()
    elif condition.startswith(("pink_snr", "white_snr")):
        db = {"white_snr20": 20, "pink_snr10": 10, "pink_snr0": 0,
              "pink_snr_minus5": -5}[condition]
        result, measured = mix_at_ratio(wave, coloured_noise(len(wave), rng,
                                       "white" if condition.startswith("white") else "pink"), db)
    elif condition == "attenuation18db":
        result = wave * 10 ** (-18 / 20)
    elif condition == "lowpass1200":
        result = sosfilt(butter(4, 1200, fs=SR, output="sos"), wave)
    elif condition == "passby_envelope":
        t = np.arange(len(wave)) / SR
        envelope = 0.05 + 0.95 * np.exp(-0.5 * ((t - 0.50) / 0.13) ** 2)
        result = wave * envelope
    else:
        raise ValueError("Unknown condition: " + condition)
    return np.asarray(result, dtype="float32"), measured


def procedural(profile, rng, seconds=4):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if profile in {"white", "pink", "brown"}:
        wave = coloured_noise(n, rng, profile)
    elif profile in {"mains_harmonics", "fan_harmonics", "modulated_fan"}:
        f0 = rng.uniform(49, 61) if profile == "mains_harmonics" else rng.uniform(180, 650)
        wave = sum(rng.uniform(0.5, 1.0) / h * np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, 2*np.pi))
                   for h in range(1, 7))
        if profile == "modulated_fan":
            wave *= 0.6 + 0.4 * np.sin(2*np.pi*rng.uniform(1, 8)*t)
    elif profile == "chirp":
        wave = chirp(t, rng.uniform(180, 400), seconds, rng.uniform(1000, 1600))
    elif profile == "clicks":
        impulses = np.zeros(n)
        positions = rng.choice(n - 128, size=32, replace=False)
        impulses[positions] = rng.uniform(-1, 1, len(positions))
        wave = np.convolve(impulses, np.exp(-np.arange(128) / 12), mode="same")
    else:
        raise ValueError("Unknown procedural profile.")
    wave -= wave.mean()
    wave *= 10 ** rng.uniform(-2.3, -1.3) / max(rms(wave), 1e-12)
    wave *= min(1.0, 0.90 / max(float(np.abs(wave).max()), 1e-12))
    return wave.astype("float32")


def choose_parents(frame, partitions, per_negative_fold=24):
    """Random clip within each chosen group; no probabilities are consulted."""
    table = frame.copy()
    tests = partitions.loc[partitions.role.eq("test"), ["row_id", "fold"]]
    if tests.row_id.duplicated().any() or set(tests.row_id) != set(frame.row_id):
        raise ValueError("Each development row must have exactly one outer test fold.")
    table = table.merge(tests, on="row_id", validate="one_to_one")
    group_truth = table.groupby("group").true_label.max()
    # Noise windows inside mosquito groups cannot become pure-noise donors.
    table = table.loc[table.true_label.eq(table.group.map(group_truth))]
    chosen = []
    for fold in sorted(table.fold.unique()):
        for c in [0, 1, 2]:
            subset = table.loc[table.fold.eq(fold) & table.true_label.eq(c)]
            group_meta = subset.groupby("group").agg(domain=("domain", "first"), label=("label", "first"))
            strata = {}
            for key, rows in group_meta.groupby(["domain", "label"], sort=True):
                strata[key] = sorted(rows.index, key=lambda g: generator(f"rank:{g}").random())
            groups = []
            while any(strata.values()):
                for key in sorted(strata):
                    if strata[key]:
                        groups.append(strata[key].pop())
            if c != 2:
                if len(groups) < per_negative_fold:
                    raise ValueError("Insufficient negative donor groups in a fold.")
                groups = groups[:per_negative_fold]
            for group in groups:
                rows = subset.loc[subset.group.eq(group)].sort_values("row_id")
                row = rows.iloc[generator(f"window:{group}").integers(len(rows))].copy()
                chosen.append(row)
    return pd.DataFrame(chosen).reset_index(drop=True)


def assert_heldout(bundle, parents):
    if set(parents.row_id) & set(bundle["fit_rows"]):
        raise ValueError("A donor row was used to fit this model.")
    if set(parents.pcm_block) & set(bundle["fit_pcm_blocks"]):
        raise ValueError("A donor PCM block was used to fit this model.")


@lru_cache(maxsize=8)
def source_wave(path):
    return read_audio(ROOT / path, SR)


def donor_wave(parent):
    wave = source_wave(str(parent.path))
    start = int(round(float(parent.start_s) * SR))
    clip = wave[start:start + SIZE]
    if len(clip) != SIZE or rms(clip) <= 1e-10:
        raise ValueError("Donor cannot supply the declared real 0.992 s interval.")
    return clip


def build(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    freeze = verify_freeze(run, inputs=True)
    verify_evaluation(run)
    out.mkdir(parents=True, exist_ok=False)
    (out / "audio").mkdir()
    protocol = {"created_utc": now(), "seed": SEED, "model_run": str(run),
                "freeze_sha256": digest(run / "freeze.json"), "code_sha256": digest(__file__),
                "sample_rate": SR, "donor_samples": SIZE, "conditions": CONDITIONS,
                "negative_groups_per_fold_and_class": 24, "positive_groups": "all 60",
                "mixture_signal_to_other_db": [10, 0, -10],
                "procedural_profiles": PROFILES, "procedural_replicates": 20,
                "procedural_seconds": 4, "donor_selection": "seeded source/taxon round robin; random window per group",
                "primary_model": "original frozen outer fold excluding every real component",
                "final_model_scope": "secondary for known donors; primary for procedural negatives",
                "no_model_fitting_or_threshold_tuning": True,
                "not_new_individuals_or_field_validation": True,
                "snr_definition": "total donor clip RMS / added component RMS",
                "prevalence_is_designed_not_field_prevalence": True}
    json_write(out / "protocol.json", protocol)  # Before synthesis and inference.
    frame = pd.read_csv(run / "segments.csv.gz", low_memory=False)
    parents = choose_parents(frame, pd.read_csv(run / "partitions.csv.gz"))
    if not parents.window_duration_s.eq(SIZE / SR).all():
        raise ValueError("This stress protocol requires real 0.992 s donor intervals.")
    parents.to_csv(out / "parents.csv", index=False)
    for k, sub in parents.groupby("fold"):
        assert_heldout(joblib.load(run / "models" / f"outer_{k}.joblib"), sub)
    rows = []

    def save(wave, *, suite, condition, truth, parent=None, partner=None, measured=None):
        clip_id = f"synth_{len(rows):05d}"
        path = out / "audio" / (clip_id + ".wav")
        sf.write(path, wave, SR, subtype="FLOAT")
        rows.append({"clip_id": clip_id, "path": str(path), "suite": suite,
                     "condition": condition, "true_label": int(truth),
                     "fold": int(parent.fold) if parent is not None else -1,
                     "parent_group": str(parent.group) if parent is not None else None,
                     "parent_row_id": int(parent.row_id) if parent is not None else None,
                     "parent_pcm_block": str(parent.pcm_block) if parent is not None else None,
                     "parent_domain": str(parent.domain) if parent is not None else "procedural",
                     "partner_row_id": int(partner.row_id) if partner is not None else None,
                     "partner_pcm_block": str(partner.pcm_block) if partner is not None else None,
                     "samples": len(wave), "seconds": len(wave)/SR,
                     "measured_component_ratio_db": measured, "sha256": digest(path)})

    for _, parent in parents.iterrows():
        wave = donor_wave(parent)
        for condition in CONDITIONS:
            result, measured = transform(wave, condition, generator(f"{parent.group}:{condition}"))
            save(result, suite="paired_perturbations", condition=condition,
                 truth=parent.true_label, parent=parent, measured=measured)
    for _, parent in parents.loc[parents.true_label.eq(2)].iterrows():
        candidates = parents.loc[parents.fold.eq(parent.fold) & parents.true_label.eq(1)
                                & parents.pcm_block.ne(parent.pcm_block)]
        partner = candidates.iloc[generator(f"partner:{parent.group}").integers(len(candidates))]
        for db in [10, 0, -10]:
            wave, measured = mix_at_ratio(donor_wave(parent), donor_wave(partner), db)
            save(wave, suite="mixed_mosquitoes", condition=f"aedes_to_other_{db}db",
                 truth=2, parent=parent, partner=partner, measured=measured)
    for profile in PROFILES:
        for replicate in range(20):
            save(procedural(profile, generator(f"{profile}:{replicate}")),
                 suite="procedural_negatives", condition=profile, truth=0)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    json_write(out / "dataset_complete.json", {
        "completed_utc": now(), "clips": len(rows), "donor_groups": len(parents),
        "manifest_sha256": digest(out / "manifest.csv"), "parents_sha256": digest(out / "parents.csv"),
        "protocol_sha256": digest(out / "protocol.json"), "inference_started": False,
        "generator_sha256": digest(__file__),
    })
    source_wave.cache_clear()
    print("DATASET FROZEN", len(rows), "WAVs; donors", len(parents), flush=True)
    return out


def audio_batch(manifest, batch_size=64):
    """Same 16kHz frontend/quality rule as predict_ternary, evaluated in batches."""
    from aedes_classical import features
    from bioacoustic_invariance import extract_batch_bioacoustic_features
    frames, all_x, pending, metadata = [], [], [], []

    def flush():
        if not pending:
            return
        waves = np.stack(pending)
        pcm = np.rint(np.clip(waves, -1, 32767/32768)*32768).astype("<i2").astype("int32")
        quality = (np.abs(pcm).max(1) >= 32) & ((np.abs(pcm) >= 32700).sum(1)*100 <= SIZE)
        frame = pd.DataFrame(metadata)
        frame["quality_ok"] = quality
        frames.append(frame)
        all_x.append(np.concatenate([features(waves), extract_batch_bioacoustic_features(waves)], axis=1))
        pending.clear(); metadata.clear()

    for _, item in manifest.iterrows():
        wave = read_audio(item.path, SR)
        for j in range(len(wave)//SIZE):
            pending.append(wave[j*SIZE:(j+1)*SIZE])
            metadata.append({"group": item.clip_id, "path": item.path, "domain": item.suite,
                             "pcm_block": item.parent_pcm_block if pd.notna(item.parent_pcm_block) else item.clip_id,
                             "true_label": item.true_label, "fold": item.fold,
                             "start_s": j*SIZE/SR, "window_duration_s": SIZE/SR})
            if len(pending) == batch_size:
                flush()
    flush()
    frame, x = pd.concat(frames, ignore_index=True), np.concatenate(all_x)
    frame["row_id"] = np.arange(len(frame))
    if not np.isfinite(x).all() or x.shape[1] != 138:
        raise ValueError("Invalid synthetic frontend output.")
    return frame, x


def metrics(truth, pred):
    """Abstentions remain in all truth denominators and have a separate column."""
    truth, pred = np.asarray(truth, dtype=int), np.asarray(pred, dtype=int)
    if truth.shape != pred.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Aligned nonempty vectors required.")
    if not set(truth).issubset({0, 1, 2}) or not set(pred).issubset({-1, 0, 1, 2}):
        raise ValueError("Unexpected class.")
    cm = np.zeros((3, 4), dtype=int)
    np.add.at(cm, (truth, np.where(pred == -1, 3, pred)), 1)
    support, emitted = cm.sum(1), cm[:, :3].sum(0)
    tp, fp0, fp1 = int(cm[2, 2]), int(cm[0, 2]), int(cm[1, 2])
    ratio = lambda a, b: float(a/b) if b else None
    return {"clips": len(truth), "coverage": float((pred >= 0).mean()),
            "invalid_audio": int((pred == -1).sum()), "aedes_tp": tp, "aedes_total": int(support[2]),
            "noise_fp": fp0, "noise_total": int(support[0]), "other_fp": fp1, "other_total": int(support[1]),
            "aedes_recall": ratio(tp, support[2]), "aedes_precision": ratio(tp, emitted[2]),
            "noise_to_aedes_fpr": ratio(fp0, support[0]), "other_to_aedes_fpr": ratio(fp1, support[1]),
            "balanced_accuracy": float(np.mean([cm[c,c]/support[c] for c in range(3) if support[c]])),
            "confusion_rows_noise_other_aedes_columns_noise_other_aedes_invalid": cm.tolist()}


def model_decisions(bundle, frame, x, policies):
    valid = frame.quality_ok.to_numpy(dtype=bool)
    if not valid.any():
        return pd.DataFrame(index=pd.Index([], name="group"))
    lr, hgb = predict_pair(bundle, x[valid])
    groups = aggregate_groups(frame.loc[valid], lr, hgb, bundle.get("temporal", {
        "horizon_s": 1.984, "gap_tolerance_s": .01, "minimum_windows": 2}))
    result = pd.DataFrame({"group": groups.group})
    for name, config in policies.items():
        result[name] = apply_policy(groups, config)
    for name, family in [("lr_simple", "lr"), ("hgb_simple", "hgb"), ("ensemble_simple", "ensemble")]:
        result[name] = groups[[f"{family}_simple_{c}" for c in range(3)]].to_numpy().argmax(1)
    return result.set_index("group")


def evaluate(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    verify_evaluation(run)
    if (out / "evaluation_started.json").exists():
        raise FileExistsError("This synthetic evaluation already started; use a new directory.")
    complete = json.loads((out / "dataset_complete.json").read_text())
    for name, key in [("manifest.csv", "manifest_sha256"), ("parents.csv", "parents_sha256"),
                      ("protocol.json", "protocol_sha256")]:
        if digest(out / name) != complete[key]:
            raise ValueError("Frozen synthetic dataset changed: " + name)
    protocol = json.loads((out / "protocol.json").read_text())
    if digest(__file__) != complete["generator_sha256"] or protocol["freeze_sha256"] != digest(run / "freeze.json"):
        raise ValueError("Generator or model changed after dataset definition.")
    manifest = pd.read_csv(out / "manifest.csv")
    for item in manifest.itertuples():
        if digest(item.path) != item.sha256:
            raise ValueError("Synthetic WAV changed: " + item.clip_id)
    json_write(out / "evaluation_started.json", {"started_utc": now(), "dataset_sha256": digest(out / "dataset_complete.json")})
    parents = pd.read_csv(out / "parents.csv")
    predictions = manifest.copy().set_index("clip_id", drop=False)
    with threadpool_limits(limits=4):
        frame, x = audio_batch(manifest)
        np.save(out / "features.npy", x)
        frame.to_csv(out / "windows.csv.gz", index=False)
        final = joblib.load(run / "models/final.joblib")
        final_result = model_decisions(final, frame, x, final["policies"])
        for name in final_result:
            predictions["final_" + name] = final_result[name].reindex(predictions.index).fillna(-1).astype(int)
        thresholds = load_thresholds(run)["folds"]
        for k in range(5):
            bundle = joblib.load(run / "models" / f"outer_{k}.joblib")
            assert_heldout(bundle, parents.loc[parents.fold.eq(k)])
            mask = frame.fold.eq(k).to_numpy()
            result = model_decisions(bundle, frame.loc[mask], x[mask], thresholds[f"fold_{k}"]["policies"])
            for name in result:
                column = "heldout_" + name
                if column not in predictions:
                    predictions[column] = -1
                ids = predictions.index[predictions.fold.eq(k)]
                predictions.loc[ids, column] = result[name].reindex(ids).fillna(-1).astype(int)
            print("SYNTHETIC INFERENCE fold", k, flush=True)
    quality = frame.groupby("group").quality_ok.agg(["sum", "size"])
    predictions["valid_windows"] = quality["sum"].reindex(predictions.index).astype(int)
    predictions["invalid_windows"] = (quality["size"]-quality["sum"]).reindex(predictions.index).astype(int)
    predictions.to_csv(out / "predictions.csv", index=False)
    summaries = {}
    for (suite, condition), sub in predictions.groupby(["suite", "condition"], sort=True):
        eligible = [c for c in predictions.columns if c.startswith("final_") or
                    (c.startswith("heldout_") and suite != "procedural_negatives")]
        summaries[suite + "/" + condition] = {
            "clips": len(sub), "donor_groups": int(sub.parent_group.nunique()),
            "models": {c: metrics(sub.true_label, sub[c]) for c in eligible}}
    result = {"completed_utc": now(), "model_freeze_sha256": digest(run / "freeze.json"),
              "dataset_clips": len(predictions), "donor_groups": len(parents),
              "not_independent_new_mosquitoes": True, "models_refitted": False,
              "thresholds_retuned": False, "conditions": summaries}
    json_write(out / "metrics.json", result)
    verify_evaluation(run)
    json_write(out / "evaluation_complete.json", {
        "completed_utc": now(), "generator_sha256": digest(__file__),
        "artifacts_sha256": {name: digest(out / name) for name in
                             ["metrics.json", "predictions.csv", "features.npy", "windows.csv.gz",
                              "protocol.json", "parents.csv", "manifest.csv", "dataset_complete.json"]},
        "no_model_or_threshold_changes": True})
    print("SYNTHETIC EVALUATION COMPLETE", len(predictions), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["build", "evaluate", "all"], default="all")
    args = parser.parse_args()
    if args.phase in {"build", "all"}:
        build(args.run, args.out)
    if args.phase in {"evaluate", "all"}:
        evaluate(args.run, args.out)


if __name__ == "__main__":
    main()
