"""Shared, explicit protocol for the audited offline ternary experiment.

No function in this module writes to a historical experiment directory.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
SEGMENTS = ROOT / "data/metadata/ternary_segments.csv.gz"
FEATURES = ROOT / "data/cache/reliability_2026_10_01/features.npy"
KNOWN_MAP = ROOT / "data/metadata/known_recording_groups.csv"
CLASS_NAMES = ["no_mosquito", "other_mosquito", "aedes_aegypti"]
SEED = 20261014
TEMPORAL = {"horizon_s": 1.984, "gap_tolerance_s": 0.01, "minimum_windows": 2}
MODEL_CONFIG = {
    "lr": {"C": 0.1, "max_iter": 2000, "class_weight": None, "random_state": SEED},
    "hgb": {"max_iter": 200, "l2_regularization": 3.0, "learning_rate": 0.05,
            "early_stopping": False, "random_state": SEED},
}


def now():
    return datetime.now(timezone.utc).isoformat()


EXCLUDED_DRIVE_SHA256 = "138fce9b9780f22013011849d793938282695a06647eb3467491cf95b6838eca"


def excluded_drive_audio_mask(frame):
    """Exclude the private swarm recording, not public datasets hosted on Drive.

    Use available provenance and known byte identity. Arbitrarily transformed,
    renamed audio without provenance still requires an acquisition audit.
    """
    excluded = pd.Series(False, index=frame.index)
    for column in ("domain", "source", "group"):
        if column in frame:
            value = frame[column].fillna("").astype(str).str.lower()
            excluded |= value.str.match(r"^(?:drive|drive_test|private_swarm)(?::|$)")
    if "path" in frame:
        value = frame.path.fillna("").astype(str).str.lower().str.replace("\\", "/", regex=False)
        excluded |= value.str.contains(r"(?:^|/)(?:drive_audio\.wav|drive_test|private-audio-test)(?:/|$)", regex=True)
    for column in ("sha256", "content_sha256", "file_sha256"):
        if column in frame:
            excluded |= frame[column].astype(str).str.lower().eq(EXCLUDED_DRIVE_SHA256)
    return excluded


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_audio(path, sample_rate=16000):
    """Decode mono audio; compressed formats use the same documented ffmpeg path."""
    try:
        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
    except sf.LibsndfileError:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar",
             str(sample_rate), "-f", "f32le", "pipe:1"],
            check=True, capture_output=True,
        ).stdout
        wave, rate = np.frombuffer(raw, dtype="<f4").copy(), sample_rate
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Empty or non-finite decoded audio: " + str(path))
    if rate != sample_rate:
        common = np.gcd(rate, sample_rate)
        wave = resample_poly(wave, sample_rate // common, rate // common).astype("float32")
    return wave


def pcm_signatures(wave, sample_rate=8000):
    """Whole/aligned 1/.992/.3 s, raw and gain-normalised, nonconstant PCM."""
    wave = np.asarray(wave)
    keys = set()
    pieces = [wave]
    for seconds in [1.0, 0.992, 0.3]:
        size = int(round(seconds * sample_rate))
        pieces.extend(wave[start:start + size]
                      for start in range(0, len(wave) - size + 1, size))
    for piece in pieces:
        for variant in [piece, piece / max(float(np.abs(piece).max()), 1e-12) * 0.9]:
            pcm = np.rint(np.clip(variant, -1, 32767 / 32768) * 32768).astype("<i2")
            if np.ptp(pcm.astype("int32")) > 2:
                keys.add(str(len(pcm)) + ":" + hashlib.sha256(pcm.tobytes()).hexdigest())
    return keys


def known_collision_map(path=KNOWN_MAP):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Required PCM collision map is missing: " + str(path))
    table = pd.read_csv(path)
    if not {"group", "collision_group"}.issubset(table.columns):
        raise ValueError("PCM collision map has an invalid schema.")
    if table[["group", "collision_group"]].isna().any().any():
        raise ValueError("PCM collision map contains missing identities.")
    if table.groupby("group").collision_group.nunique().max() != 1:
        raise ValueError("Conflicting identities in PCM collision map.")
    table = table[["group", "collision_group"]].drop_duplicates()
    return dict(zip(table.group.astype(str), table.collision_group.astype(str)))


def load_dataset(segments=SEGMENTS, features=FEATURES, collision_map=KNOWN_MAP):
    """Load unchanged cached features; reconstruct the actual feature-window duration.

    The old Dryad annotation is 1 s, but its cached frontend uses 15,872 samples
    at 16 kHz. Preserve that annotation separately; do not turn it into duration.
    """
    mapping = known_collision_map(collision_map)  # Mandatory, even before loading features.
    original = pd.read_csv(segments, low_memory=False)
    raw = np.load(features, mmap_mode="r")
    if len(original) != len(raw) or raw.ndim != 2:
        raise ValueError("Feature rows do not align with the segment table.")
    required = {"path", "group", "domain", "label", "quality_ok", "bucket", "start_s", "wave_index"}
    if not required.issubset(original.columns):
        raise ValueError("Incomplete segment schema: " + str(sorted(required - set(original))))
    lower = original.label.astype(str).str.lower()
    domain = original.domain.astype(str).str.lower()
    ufrgs = (domain.eq("ufrgs") | original.path.astype(str).str.contains("ufrgs", case=False)
             | original.group.astype(str).str.contains("ufrgs", case=False))
    figshare = domain.eq("figshare") | original.path.astype(str).str.contains("figshare", case=False)
    mosquitosong = (domain.eq("mosquitosong") | original.path.astype(str).str.contains("mosquitosong", case=False)
                    | original.group.astype(str).str.contains("mosquitosong", case=False))
    unknown = lower.isin(["unknown_mosquito", "unknown", "mosquito"])
    quality = original.quality_ok.isin([True, "True", "true", 1, "1"])
    finite = np.isfinite(raw).all(axis=1)
    keep = ~ufrgs & ~figshare & ~mosquitosong & ~excluded_drive_audio_mask(original) & ~unknown & quality & finite
    frame = original.loc[keep].copy()
    frame.insert(0, "row_id", np.flatnonzero(keep))
    frame = frame.reset_index(drop=True)
    x = np.asarray(raw[keep.to_numpy()], dtype="float32")
    label = frame.label.astype(str).str.lower()
    target = np.where(label.eq("noise") | frame.domain.eq("esc50"), 0,
                      np.where(label.eq("aedes aegypti"), 2, 1)).astype(int)
    frame["true_label"] = target
    frame["ternary_target"] = target
    frame["ternary_class_name"] = np.asarray(CLASS_NAMES)[target]
    frame["declared_duration_s"] = frame.get("duration_s", pd.Series(np.nan, index=frame.index))
    durations = {}
    cache = ROOT / "data/cache/reliability_2026_10_01"
    for bucket in sorted(frame.bucket.unique()):
        if bucket not in {"long", "short"}:
            raise ValueError("Unknown waveform bucket: " + str(bucket))
        waves = np.load(cache / (bucket + "_8k.npy"), mmap_mode="r")
        duration = waves.shape[1] / 8000
        expected = 0.992 if bucket == "long" else 0.3
        if duration != expected:
            raise ValueError("Waveform cache disagrees with the frozen frontend duration.")
        index = frame.loc[frame.bucket.eq(bucket), "wave_index"].to_numpy()
        if not np.isfinite(index).all() or (index < 0).any() or (index >= len(waves)).any():
            raise ValueError("Invalid waveform cache index.")
        durations[bucket] = duration
    frame["window_duration_s"] = frame.bucket.map(durations).astype(float)
    frame["duration_s"] = frame.window_duration_s
    frame["duration_origin"] = "waveform_cache_samples_at_8000Hz"
    if not np.isfinite(frame.start_s).all() or (frame.start_s < 0).any():
        raise ValueError("Missing/invalid timestamps; cannot assert temporal continuity.")
    if frame.duplicated(["path", "start_s"]).any():
        raise ValueError("Duplicate windows with the same path and start.")
    if frame.groupby("path").group.nunique().max() != 1:
        raise ValueError("A source WAV belongs to multiple original groups.")
    frame["pcm_block"] = frame.group.map(mapping).fillna(frame.group)
    stats = {
        "total_original": len(original), "retained_total": len(frame),
        "excluded_ufrgs": int(ufrgs.sum()), "excluded_figshare": int(figshare.sum()),
        "excluded_mosquitosong": int(mosquitosong.sum()),
        "excluded_drive_swarm": int(excluded_drive_audio_mask(original).sum()),
        "excluded_unknown": int(unknown.sum()), "excluded_quality": int((~quality).sum()),
        "excluded_non_finite": int((~finite).sum()),
        "exclusion_counts_overlap": True,
        "class_counts": {CLASS_NAMES[c]: int((target == c).sum()) for c in range(3)},
        "group_counts": {CLASS_NAMES[c]: int(frame.loc[target == c, "group"].nunique()) for c in range(3)},
        "known_map_groups": int(frame.loc[frame.group.isin(mapping), "group"].nunique()),
        "duration_s_reconstructed_for_all_windows": True,
        "input_sha256": {str(Path(p).relative_to(ROOT)): digest(p)
                         for p in [segments, features, collision_map]},
    }
    return frame, x, target, stats


def audit_pcm(frame, out, collision_map=KNOWN_MAP):
    """Decode every retained source; union known links, bytes and declared PCM keys."""
    out = Path(out)
    if (out / "pcm_audit.json").exists():
        raise FileExistsError("PCM audit already exists; do not silently reuse or replace it.")
    mapping = known_collision_map(collision_map)
    groups = sorted(frame.group.astype(str).unique())
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    previous = {}
    for group in groups:
        block = mapping.get(group, group)
        if block in previous:
            union(group, previous[block])
        else:
            previous[block] = group
    owners, byteowners, collisions, files, signatures = {}, {}, [], [], {}
    paths = sorted(frame.path.unique())
    for number, name in enumerate(paths, 1):
        path = ROOT / name
        group = str(frame.loc[frame.path.eq(name), "group"].iloc[0])
        bytehash = digest(path)
        wave = read_audio(path, 8000)
        sub = frame.loc[frame.path.eq(name)]
        if {"start_s", "window_duration_s"}.issubset(sub.columns):
            if (sub.start_s + sub.window_duration_s > len(wave) / 8000 + 1 / 8000).any():
                raise ValueError("A cached window extends beyond its source audio: " + str(name))
        keys = pcm_signatures(wave)
        if bytehash in byteowners:
            union(group, byteowners[bytehash])
        byteowners[bytehash] = group
        for key in sorted(keys):
            if key in owners:
                other_group, other_path = owners[key]
                if find(group) != find(other_group):
                    collisions.append({"path": name, "other_path": other_path, "pcm_key": key})
                    union(group, other_group)
            else:
                owners[key] = (group, name)
        signatures[name] = sorted(keys)
        files.append({"path": name, "group": group, "file_sha256": bytehash,
                      "decoded_seconds": len(wave) / 8000, "pcm_signatures": len(keys)})
        if number % 200 == 0 or number == len(paths):
            print("PCM audit", number, "/", len(paths), flush=True)
    result = frame.copy()
    result["pcm_block"] = result.group.astype(str).map(lambda g: "pcm:" + find(g))
    group_map = result[["group", "pcm_block"]].drop_duplicates().sort_values("group")
    group_map.to_csv(out / "pcm_groups.csv", index=False)
    pd.DataFrame(files).to_csv(out / "source_pcm_audit.csv", index=False)
    pd.DataFrame(collisions, columns=["path", "other_path", "pcm_key"]).to_csv(out / "pcm_collisions.csv", index=False)
    with gzip.open(out / "pcm_signatures.json.gz", "wt") as stream:
        json.dump(signatures, stream, sort_keys=True)
    metadata = {
        "created_utc": now(), "all_retained_sources_decoded": True,
        "source_files": len(files), "groups": len(groups),
        "pcm_blocks": int(result.pcm_block.nunique()), "decode_errors": [],
        "known_map_sha256": digest(collision_map),
        "known_map_covered_groups": len(set(groups) & set(mapping)),
        "audited_group_coverage": 1.0, "source_file_manifest_sha256": digest(out / "source_pcm_audit.csv"),
        "group_map_sha256": digest(out / "pcm_groups.csv"),
        "signature_index_sha256": digest(out / "pcm_signatures.json.gz"),
        "limits": "Whole/aligned 1/.992/.3 s 8kHz raw/gain-normalised nonconstant PCM and exact bytes. "
                  "Does not prove absence of arbitrary shifts, edits, or shared biological/session ancestry.",
    }
    json_write(out / "pcm_audit.json", metadata)
    return result, metadata


def sample_weights(frame, y, gamma=0.5):
    y = np.asarray(y, dtype=int)
    if len(frame) != len(y) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Fit must contain all three classes and aligned rows.")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Invalid duration exponent.")
    size = frame.group.map(frame.group.value_counts()).to_numpy(dtype=float)
    raw = np.clip(size, 1, 60) ** (-gamma)
    mass = np.bincount(y, weights=raw, minlength=3)
    weights = raw * (len(y) / (3 * mass[y]))
    if not np.allclose(np.bincount(y, weights=weights), len(y) / 3):
        raise ValueError("Class weight mass is inconsistent.")
    return weights


def check_partition(frame, fit, test):
    fit, test = np.asarray(fit, dtype=int), np.asarray(test, dtype=int)
    if set(fit) & set(test):
        raise ValueError("Row leakage.")
    for column in ["row_id", "group", "path", "pcm_block"]:
        if set(frame.iloc[fit][column]) & set(frame.iloc[test][column]):
            raise ValueError(column + " leakage.")
    if set(frame.iloc[fit].true_label) != {0, 1, 2}:
        raise ValueError("Fit partition is missing a target class.")
    return {"fit_rows": len(fit), "test_rows": len(test),
            "fit_pcm_blocks": int(frame.iloc[fit].pcm_block.nunique()),
            "test_pcm_blocks": int(frame.iloc[test].pcm_block.nunique()),
            "overlap_rows_groups_paths_pcm": 0}


def activity_pool(prob):
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[1] != 3 or not len(prob):
        raise ValueError("Empty or malformed probability bag.")
    if not np.isfinite(prob).all() or (prob < 0).any() or not np.allclose(prob.sum(1), 1):
        raise ValueError("Invalid probabilities.")
    weight = (1 - np.clip(prob[:, 0], 0, 1)) ** 1.5
    if weight.sum() < 1e-6:
        weight = np.ones(len(prob))
    result = np.average(prob, axis=0, weights=weight)
    return result / result.sum()


def log_pool(lr, hgb, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Invalid opinion-pool weight.")
    z = weight * np.log(np.maximum(lr, 1e-12)) + (1 - weight) * np.log(np.maximum(hgb, 1e-12))
    p = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def persistence(frame, column, *, horizon_s=1.984, gap_tolerance_s=0.01, minimum_windows=2):
    """Best time-weighted score over a real horizon inside one WAV.

    A peak alone, an unknown duration, and a disconnected pair are not evidence
    of temporal persistence. Tiny acquisition gaps are bounded explicitly.
    """
    if horizon_s <= 0 or gap_tolerance_s < 0 or minimum_windows < 2:
        raise ValueError("Invalid temporal protocol.")
    duration_column = "window_duration_s" if "window_duration_s" in frame else "duration_s"
    required = {"path", "start_s", duration_column, column}
    empty = {"value": None, "available": False, "horizon_s": horizon_s,
             "covered_s": 0.0, "windows": 0, "path": None, "reason": "no_contiguous_horizon"}
    if not required.issubset(frame.columns):
        return dict(empty, reason="missing_temporal_metadata")
    if frame[list(required)].isna().any().any():
        return dict(empty, reason="missing_temporal_metadata")
    best = empty
    for path, sub in frame.groupby("path", sort=True):
        sub = sub.sort_values(["start_s", duration_column], kind="stable")
        starts = sub.start_s.to_numpy(dtype=float)
        durations = sub[duration_column].to_numpy(dtype=float)
        values = sub[column].to_numpy(dtype=float)
        if (not np.isfinite(np.r_[starts, durations, values]).all()
                or (durations <= 0).any() or (starts < 0).any()
                or (np.diff(starts) <= 0).any() or (values < 0).any() or (values > 1).any()):
            raise ValueError("Invalid/duplicate temporal intervals in " + str(path))
        for first in range(len(sub)):
            end = starts[first] + horizon_s
            cursor, area, covered, count = starts[first], 0.0, 0.0, 0
            for i in range(first, len(sub)):
                gap = starts[i] - cursor
                if gap < -1e-7 or gap > gap_tolerance_s + 1e-7:
                    break
                width = min(starts[i] + durations[i], end) - starts[i]
                if width <= 0:
                    break
                area += width * values[i]
                covered += width
                count += 1
                cursor = starts[i] + durations[i]
                if cursor >= end - 1e-7:
                    if count >= minimum_windows and horizon_s - covered <= gap_tolerance_s + 1e-7:
                        value = float(area / covered)
                        if best["value"] is None or value > best["value"]:
                            best = {"value": value, "available": True, "horizon_s": horizon_s,
                                    "covered_s": float(covered), "windows": count, "path": str(path),
                                    "reason": None}
                    break
    return best


def aggregate_groups(frame, lr, hgb, temporal=TEMPORAL):
    if len(frame) != len(lr) or len(frame) != len(hgb):
        raise ValueError("Misaligned group probabilities.")
    rows = []
    for group, ix in frame.groupby("group", sort=True).indices.items():
        sub = frame.iloc[ix].copy()
        a, b = np.asarray(lr)[ix], np.asarray(hgb)[ix]
        ensemble = (a + b) / 2
        sub["_q"] = ensemble[:, 2] / np.maximum(ensemble[:, 1] + ensemble[:, 2], 1e-12)
        sub["_p2"] = ensemble[:, 2]
        temporal_q = persistence(sub, "_q", **temporal)
        temporal_p2 = persistence(sub, "_p2", **temporal)
        if "pcm_block" in sub and sub.pcm_block.nunique() != 1:
            raise ValueError("One group crosses PCM blocks.")
        if sub.domain.nunique() != 1:
            raise ValueError("One group has ambiguous source metadata.")
        row = {"group": str(group), "true_label": int(sub.true_label.max()) if "true_label" in sub else None,
               "pcm_block": str(sub.pcm_block.iloc[0]) if "pcm_block" in sub else str(group),
               "source": str(sub.domain.iloc[0]), "num_windows": len(sub),
               "paths": int(sub.path.nunique()), "persistence_q": temporal_q["value"],
               "persistence_p2": temporal_p2["value"], "persistence_available": temporal_q["available"],
               "persistence_reason": temporal_q["reason"],
               "persistence_covered_s": temporal_q["covered_s"],
               "persistence_windows": temporal_q["windows"]}
        if "fold" in sub:
            if sub.fold.nunique() != 1:
                raise ValueError("One group crosses validation folds.")
            row["fold"] = int(sub.fold.iloc[0])
        for name, prob in [("lr", a), ("hgb", b), ("ensemble", ensemble)]:
            for suffix, pooled in [("simple", prob.mean(0)), ("gated", activity_pool(prob))]:
                for c in range(3):
                    row[f"{name}_{suffix}_{c}"] = float(pooled[c])
        rows.append(row)
    return pd.DataFrame(rows)


def group_probabilities(groups, family, weight=0):
    if family in {"hgb", "lr", "ensemble"}:
        return groups[[f"{family}_gated_{c}" for c in range(3)]].to_numpy()
    if family != "log_pool":
        raise ValueError("Unknown decision family.")
    return log_pool(group_probabilities(groups, "lr"), group_probabilities(groups, "hgb"), weight)


def decision_scores(groups, config):
    prob = group_probabilities(groups, config["family"], config.get("blend_weight_lr", 0))
    conditional = prob[:, 2] / np.maximum(prob[:, 1] + prob[:, 2], 1e-12)
    beta = config.get("temporal_weight", 0.0)
    if not 0 <= beta <= 1:
        raise ValueError("Invalid temporal weight.")
    available = groups.persistence_available.to_numpy(dtype=bool)
    temporal = groups.persistence_q.to_numpy(dtype=float)
    score = conditional.copy()
    score[available] = (1 - beta) * conditional[available] + beta * temporal[available]
    return prob[:, 0], score


def apply_policy(groups, config):
    if config.get("status") == "reject_all_aedes":
        p0, _ = decision_scores(groups, config)
        return np.where(p0 >= config["tau_noise"], 0, 1)
    p0, score = decision_scores(groups, config)
    noise = p0 >= config["tau_noise"]
    return np.where(noise, 0, np.where(score >= config["tau_aedes"], 2, 1))


def calibrate_policy(groups, family, *, temporal=False, noise_fpr=0.0075, other_fpr=0.0175):
    """Empirical constraints, never a claim of population error control."""
    truth = groups.true_label.to_numpy(dtype=int)
    counts = np.bincount(truth, minlength=3)
    if (counts == 0).any():
        raise ValueError("Calibration needs all three classes.")
    max0, max1 = int(np.floor(noise_fpr * counts[0])), int(np.floor(other_fpr * counts[1]))
    weights = [0, 0.05, 0.1, 0.2, 0.4, 0.6] if family == "log_pool" else [0]
    betas = [0, 0.25, 0.5] if temporal else [0]
    best, best_key = None, None
    for weight in weights:
        for beta in betas:
            config = {"family": family, "blend_weight_lr": weight, "temporal_weight": beta}
            p0, score = decision_scores(groups, config)
            for tau_noise in np.linspace(0.4, 0.7, 16):
                active = p0 < tau_noise
                # Quantiles are not used: include every distinct empirical boundary.
                thresholds = np.unique(np.r_[np.linspace(0.2, 0.9, 36), score[active], 1.000001])
                order = np.argsort(score[active], kind="stable")
                sorted_score, sorted_truth = score[active][order], truth[active][order]
                positions = np.searchsorted(sorted_score, thresholds, side="left")
                suffix = [np.r_[np.cumsum((sorted_truth == c)[::-1])[::-1], 0]
                          for c in range(3)]
                fp0, fp1, tp = (suffix[c][positions] for c in range(3))
                feasible = np.flatnonzero((fp0 <= max0) & (fp1 <= max1))
                for j in feasible:
                    key = (int(tp[j]), -int(fp0[j] + fp1[j]), -beta, -weight, -float(tau_noise),
                           float(thresholds[j]))
                    if best_key is None or key > best_key:
                        best_key = key
                        best = dict(config, tau_noise=float(tau_noise), tau_aedes=float(thresholds[j]),
                                    status="feasible", calibration_aedes_tp=int(tp[j]),
                                    calibration_aedes_total=int(counts[2]),
                                    calibration_noise_fp=int(fp0[j]), calibration_other_fp=int(fp1[j]),
                                    calibration_noise_total=int(counts[0]), calibration_other_total=int(counts[1]),
                                    empirical_limits={"noise": noise_fpr, "other": other_fpr},
                                    allowed_false_positives={"noise": max0, "other": max1})
    if best is None:
        raise ValueError("No calibration candidate was evaluated.")
    if best["calibration_aedes_tp"] == 0:
        best["status"] = "reject_all_aedes"
    best["selection_scope"] = "inner_oof_within_outer_training"
    best["no_population_guarantee"] = True
    return best


def verify_freeze(out, *, inputs=False, code=True):
    out = Path(out)
    freeze_path = out / "freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Missing frozen protocol; historical outputs cannot be treated as a verified run.")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema_version") != 1 or not freeze.get("no_outer_predictions_yet"):
        raise ValueError("Invalid experiment freeze.")
    for relative, expected in freeze["artifacts_sha256"].items():
        path = (out / relative).resolve()
        if not path.is_relative_to(out.resolve()) or digest(path) != expected:
            raise ValueError("Frozen artifact changed: " + relative)
    if code:
        for relative, expected in freeze["code_sha256"].items():
            archived = out / "source" / relative
            target = archived if archived.is_file() else (ROOT / relative)
            if digest(target) != expected:
                raise ValueError("Frozen code changed: " + relative)
    if inputs:
        for relative, expected in freeze["input_sha256"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError("Training input changed: " + relative)
    return freeze


def load_thresholds(out):
    """Never derive a missing inner threshold from the outer prediction table."""
    out = Path(out)
    path = out / "inner_thresholds.json"
    if not path.is_file():
        raise FileNotFoundError("Required inner_thresholds.json is missing; no legacy fallback is allowed.")
    freeze = verify_freeze(out)
    thresholds = json.loads(path.read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    if thresholds.get("run_id") != freeze["run_id"] or protocol["run_id"] != freeze["run_id"]:
        raise ValueError("Thresholds belong to another experiment.")
    expected = {f"fold_{k}" for k in range(protocol["outer_splits"])}
    if set(thresholds.get("folds", {})) != expected:
        raise ValueError("Inner threshold folds are missing or incompatible.")
    for name, item in thresholds["folds"].items():
        k = int(name.split("_")[-1])
        if k in item["train_folds"] or sorted(item["train_folds"]) != sorted(set(range(protocol["outer_splits"])) - {k}):
            raise ValueError("Outer fold leaked into threshold lineage.")
        if set(item["policies"]) != set(protocol["policy_names"]):
            raise ValueError("Incomplete decision policies.")
        for config in item["policies"].values():
            for key in ["tau_noise", "tau_aedes", "blend_weight_lr", "temporal_weight"]:
                if not np.isfinite(config[key]):
                    raise ValueError("Invalid threshold parameter.")
    return thresholds
