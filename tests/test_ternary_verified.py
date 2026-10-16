"""Regression contracts for leakage, real time, missing inputs and persisted runs."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from threadpoolctl import threadpool_limits

import ternary_protocol as protocol
import ternary_reporting as reporting
import train_ternary_nested_models as trainer


def intervals(scores, starts=None, paths=None, duration=1.0):
    n = len(scores)
    return pd.DataFrame({
        "path": paths or ["a.wav"] * n,
        "start_s": np.arange(n) * duration if starts is None else starts,
        "duration_s": duration, "score": scores,
    })


def test_missing_collision_map_is_a_hard_error_before_feature_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="PCM collision map"):
        protocol.load_dataset(tmp_path / "segments.csv", tmp_path / "features.npy", tmp_path / "absent.csv")


def test_inconsistent_collision_identity_is_rejected(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame({"group": ["a", "a"], "collision_group": ["x", "y"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        protocol.known_collision_map(path)


def test_pcm_audit_decodes_and_links_a_group_not_in_historical_map(tmp_path):
    wave = np.random.default_rng(10).uniform(-0.2, 0.2, 16000)
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(first, wave, 8000)
    second.write_bytes(first.read_bytes())
    frame = pd.DataFrame({"path": [str(first), str(second)], "group": ["a", "b"]})
    mapping = tmp_path / "map.csv"
    pd.DataFrame({"group": ["a"], "collision_group": ["a"]}).to_csv(mapping, index=False)
    out = tmp_path / "audit"
    out.mkdir()
    result, audit = protocol.audit_pcm(frame, out, mapping)
    assert result.pcm_block.nunique() == 1
    assert audit["source_files"] == 2 and audit["audited_group_coverage"] == 1.0
    assert audit["known_map_covered_groups"] == 1
    assert len(pd.read_csv(out / "source_pcm_audit.csv")) == 2


@pytest.mark.parametrize("column", ["row_id", "group", "path", "pcm_block"])
def test_partition_rejects_overlap_even_when_other_identifiers_differ(column):
    frame = pd.DataFrame({
        "row_id": range(6), "group": list("abcdef"), "path": list("abcdef"),
        "pcm_block": list("abcdef"), "true_label": [0, 1, 2, 0, 1, 2],
    })
    frame.loc[3, column] = frame.loc[0, column]
    with pytest.raises(ValueError, match="leakage"):
        protocol.check_partition(frame, [0, 1, 2], [3, 4, 5])


def test_real_fit_weights_and_scaler_only_receive_fit_rows(monkeypatch):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 4))
    x[24:] += 10000  # Held-out values must not affect the fit scaler.
    y = np.tile([0, 1, 2], 10)
    frame = pd.DataFrame({"row_id": range(30), "group": ["g" + str(i) for i in range(30)],
                          "pcm_block": ["p" + str(i) for i in range(30)]})
    calls = []
    original = trainer.sample_weights

    def observed(part, labels, gamma):
        calls.append(set(part.row_id))
        return original(part, labels, gamma)

    monkeypatch.setattr(trainer, "sample_weights", observed)
    config = copy.deepcopy(protocol.MODEL_CONFIG)
    config["hgb"].update(max_iter=3, min_samples_leaf=2)
    with threadpool_limits(limits=1):
        model = trainer.fit_pair(frame.iloc[:24], x[:24], y[:24], 0.5, config)
    assert calls == [set(range(24))]
    assert set(model["fit_rows"]) == set(range(24))
    assert np.allclose(model["scaler"].mean_, x[:24].mean(0))
    assert not np.allclose(model["scaler"].mean_, x.mean(0))
    assert model["hgb_iterations"] == 3


def test_persistence_is_time_weighted_and_order_independent():
    frame = intervals([0.9, 0.1])
    a = protocol.persistence(frame, "score", horizon_s=2.0)
    b = protocol.persistence(frame.iloc[::-1], "score", horizon_s=2.0)
    assert a == b and a["value"] == pytest.approx(0.5)
    assert a["covered_s"] == 2.0 and a["windows"] == 2


@pytest.mark.parametrize("frame", [
    intervals([0.9]),
    intervals([0.9, 0.9], starts=[0, 100]),
    intervals([0.9, 0.9], paths=["a.wav", "b.wav"]),
    intervals([0.9, 0.1]).drop(columns="start_s"),
    intervals([0.9, 0.1]).drop(columns="duration_s"),
])
def test_peaks_gaps_other_files_or_missing_time_are_not_persistence(frame):
    assert protocol.persistence(frame, "score", horizon_s=2.0)["value"] is None


def test_duplicate_or_overlapping_intervals_cannot_create_a_temporal_horizon():
    with pytest.raises(ValueError, match="duplicate"):
        protocol.persistence(intervals([0.9, 0.9], starts=[0, 0]), "score")
    assert protocol.persistence(intervals([0.9, 0.9], starts=[0, 0.5]), "score",
                                horizon_s=2.0)["value"] is None


def test_same_time_horizon_works_for_short_and_long_windows():
    short = protocol.persistence(intervals([0.8] * 7, duration=0.3), "score")
    long = protocol.persistence(intervals([0.8] * 2, duration=0.992), "score")
    assert short["value"] == pytest.approx(0.8) == long["value"]
    assert short["covered_s"] == pytest.approx(1.984)
    assert long["covered_s"] == pytest.approx(1.984)
    assert short["windows"] == 7 and long["windows"] == 2


def test_temporal_feature_actually_changes_the_shared_decision():
    group = pd.DataFrame({
        "hgb_gated_0": [0.1], "hgb_gated_1": [0.54], "hgb_gated_2": [0.36],
        "persistence_available": [True], "persistence_q": [0.9],
    })
    config = {"family": "hgb", "tau_noise": 0.5, "tau_aedes": 0.6, "temporal_weight": 0.0}
    assert protocol.apply_policy(group, config).tolist() == [1]
    assert protocol.apply_policy(group, config | {"temporal_weight": 0.5}).tolist() == [2]
    missing = group.assign(persistence_available=False, persistence_q=np.nan)
    assert protocol.apply_policy(missing, config | {"temporal_weight": 0.5}).tolist() == [1]


def test_empty_denominator_is_unknown_and_confidence_is_respected():
    empty = reporting.wilson_ci(0, 0)
    assert empty["point"] is None and empty["interval"] is None and not empty["estimable"]
    narrow = reporting.wilson_ci(5, 10, 0.90)
    wide = reporting.wilson_ci(5, 10, 0.99)
    assert narrow["ci_lower"] > wide["ci_lower"]
    assert narrow["ci_upper"] < wide["ci_upper"]
    with pytest.raises(ValueError):
        reporting.wilson_ci(11, 10)


def test_all_sources_including_esc50_reconcile_to_global_denominators():
    result = reporting.evaluate_predictions([0, 0, 1, 2], [0, 2, 1, 2],
                                             sources=["esc50", "dryad", "humbug", "mosquitosong"])
    parts = result["source_breakdown"]
    assert "ESC50" in parts and parts["ESC50"]["no_mosquito"]["total"] == 1
    assert sum(v["no_mosquito"]["total"] for v in parts.values()) == 2
    assert parts["MosquitoSong"]["no_mosquito"]["fpr"]["point"] is None


def test_operational_contract_uses_integers_and_reports_failure_without_raising():
    cm = np.array([[422, 0, 4], [0, 994, 20], [0, 14, 50]])
    truth, pred = [], []
    for actual in range(3):
        for guessed in range(3):
            truth.extend([actual] * cm[actual, guessed])
            pred.extend([guessed] * cm[actual, guessed])
    result = reporting.evaluate_predictions(truth, pred)
    assert reporting.targets(result)["passed_all"]
    result["integer_counts"]["aedes_tp"] = 49
    assert not reporting.targets(result)["passed_all"]


def test_changed_population_keeps_the_predeclared_rates_instead_of_old_absolute_counts():
    metrics = {"integer_counts": {
        "aedes_total": 60, "aedes_tp": 47, "noise_total": 426, "noise_fp_to_aedes": 4,
        "other_total": 1008, "other_fp_to_aedes": 19,
    }}
    result = reporting.targets(metrics)
    assert result["required"] == {"minimum_aedes_tp": 47, "maximum_noise_fp": 4, "maximum_other_fp": 19}
    assert result["passed_all"]
    metrics["integer_counts"]["other_fp_to_aedes"] = 20
    assert not reporting.targets(metrics)["passed_all"]


@pytest.mark.skipif(
    not all(path.is_file() for path in [
        protocol.FEATURES, protocol.SEGMENTS, protocol.KNOWN_MAP,
        protocol.ROOT / "data/cache/reliability_2026_10_01/long_8k.npy",
        protocol.ROOT / "data/cache/reliability_2026_10_01/short_8k.npy",
    ]),
    reason="Corpus integration requires the local audio/feature caches, excluded from Git.",
)
def test_user_excluded_mosquitosong_source_is_not_loaded_for_development():
    frame, _, _, stats = trainer.load_and_filter_data()
    assert not frame.domain.eq("mosquitosong").any()
    assert stats["excluded_mosquitosong"] > 0


def test_missing_inner_thresholds_cannot_trigger_retrospective_calibration(tmp_path):
    with pytest.raises(FileNotFoundError, match="no legacy fallback"):
        protocol.load_thresholds(tmp_path)
    from evaluate_ternary_event_pipeline import calibrate_neyman_pearson_oof
    with pytest.raises(ValueError, match="disabled"):
        calibrate_neyman_pearson_oof(pd.DataFrame())


def test_audio_frontend_uses_real_samples_and_separates_invalid_windows(tmp_path):
    from predict_ternary import audio_features
    path = tmp_path / "real_audio.wav"
    wave = np.r_[np.zeros(15872), np.sin(np.arange(15872) * 0.17) * 0.1, np.zeros(100)]
    sf.write(path, wave, 16000)
    frame, x, details = audio_features(path)
    assert x.shape == (2, 138)
    assert frame.quality_ok.tolist() == [False, True]
    assert frame.window_duration_s.tolist() == [0.992, 0.992]
    assert details["ignored_tail_seconds"] == pytest.approx(100 / 16000)
    assert not details["padding_repetition_concatenation"]


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    rows, x, y = [], [], []
    rng = np.random.default_rng(20)
    for c in range(3):
        for group in range(6):
            path = tmp_path / f"class{c}_group{group}.wav"
            sf.write(path, rng.uniform(-0.2, 0.2, 31744), 16000)
            for window in range(2):
                index = len(rows)
                rows.append({
                    "row_id": index, "path": str(path), "group": path.stem,
                    "pcm_block": path.stem, "domain": ["esc50", "humbug", "mosquitosong"][c],
                    "start_s": window * 0.992, "window_duration_s": 0.992, "duration_s": 0.992,
                    "declared_duration_s": 1.0, "duration_origin": "synthetic_fixture",
                    "bucket": "long", "true_label": c,
                })
                x.append([c + rng.normal(scale=0.1), rng.normal(), rng.normal()])
                y.append(c)
    frame, x, y = pd.DataFrame(rows), np.array(x), np.array(y)
    mapping = tmp_path / "map.csv"
    frame[["group", "pcm_block"]].drop_duplicates().rename(
        columns={"pcm_block": "collision_group"}).to_csv(mapping, index=False)
    features = tmp_path / "features.npy"
    np.save(features, x)
    stats = {"input_sha256": {str(features): protocol.digest(features), str(mapping): protocol.digest(mapping)}}
    monkeypatch.setattr(trainer, "load_and_filter_data", lambda: (frame.copy(), x.copy(), y.copy(), stats.copy()))
    monkeypatch.setattr(trainer, "COLLISION_MAP_CSV", mapping)
    monkeypatch.setattr(trainer, "FEATURES", features)
    monkeypatch.setitem(trainer.MODEL_CONFIG, "hgb",
                        trainer.MODEL_CONFIG["hgb"] | {"max_iter": 3, "min_samples_leaf": 2})
    out = tmp_path / "verified_run"
    trainer.train(out, outer_splits=3, inner_splits=2, threads=1)
    return out


def test_small_real_training_freezes_before_outer_inference_and_has_auditable_lineage(tiny_run):
    out = tiny_run
    freeze = protocol.verify_freeze(out, inputs=True)
    assert not (out / "predictions_oof.csv").exists()
    thresholds = protocol.load_thresholds(out)
    outer = pd.read_csv(out / "partitions.csv.gz")
    for k in range(3):
        item = thresholds["folds"][f"fold_{k}"]
        assert not set(item["fit_row_ids"]) & set(item["test_row_ids"])
        inner = pd.read_csv(out / "inner" / f"outer_{k}" / "partitions.csv.gz")
        assert not set(inner.row_id) & set(item["test_row_ids"])
        selection = json.loads((out / "inner" / f"outer_{k}" / "selection.json").read_text())
        assert len(selection["candidates"]) == 2
        assert (out / "models/final.joblib").is_file()
    result = trainer.evaluate(out, threads=1)
    receipt = reporting.verify_evaluation(out)
    assert receipt["completed_utc"] > freeze["created_utc"]
    assert result["evaluation_unit"] == "group"
    assert result["uncertainty"]["unit"] == "audited_pcm_block"
    assert result["event_summary"]["total_groups"] == 18
    with pytest.raises(FileExistsError):
        trainer.train(out)
    with pytest.raises(FileExistsError):
        trainer.evaluate(out)
    target = out / "event_predictions.csv"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="output changed"):
        reporting.verify_evaluation(out)


def test_thresholds_require_the_exact_frozen_run(tiny_run):
    path = tiny_run / "inner_thresholds.json"
    payload = json.loads(path.read_text())
    payload["run_id"] = "another_run"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifact changed"):
        protocol.load_thresholds(tiny_run)


def test_bioacoustic_edge_cases_avoid_nan_entropy_and_spurious_dc_hnr():
    from bioacoustic_invariance import (
        extract_swarm_modulation,
        extract_temperature_invariant_harmonics,
    )
    freqs = np.linspace(0.0, 8000.0, 1025)
    power = np.ones_like(freqs)
    power[0] = 1e9
    zero_f0 = extract_temperature_invariant_harmonics(power, freqs, 0.0, freqs[1] - freqs[0])
    assert zero_f0["harmonic_to_noise_ratio_db"] == 0.0
    assert zero_f0["harmonic_decay_slope"] == 0.0
    assert zero_f0["harmonic_h2_ratio"] == 0.0
    short_wave = np.sin(2 * np.pi * 500.0 * np.arange(400) / 16000.0).astype("float32")
    swarm = extract_swarm_modulation(short_wave, sr=16000)
    assert np.isfinite(list(swarm.values())).all()
    assert swarm["envelope_mod_entropy"] == 0.0


def test_predict_audio_succeeds_when_repo_code_differs_from_frozen_run(tiny_run, tmp_path, monkeypatch):
    import predict_ternary
    freeze_path = tiny_run / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    assert (tiny_run / "source" / "ternary_protocol.py").is_file()
    freeze["code_sha256"]["ternary_protocol.py"] = "0" * 64
    freeze_path.write_text(json.dumps(freeze))
    monkeypatch.setattr(
        predict_ternary,
        "audio_features",
        lambda path: (
            pd.DataFrame({
                "row_id": [0, 1], "path": [str(path)] * 2, "group": ["input_audio"] * 2,
                "domain": ["inference"] * 2, "pcm_block": ["input_audio"] * 2,
                "start_s": [0.0, 0.992], "window_duration_s": [0.992, 0.992],
                "duration_s": [0.992, 0.992], "quality_ok": [True, True],
            }),
            np.zeros((2, 3), dtype="float32"),
            {"real_seconds": 1.984, "windows": 2, "ignored_tail_seconds": 0.0,
             "quality_invalid_windows": 0, "padding_repetition_concatenation": False},
        ),
    )
    out_dir = tmp_path / "pred_out"
    summary = predict_ternary.predict_audio(tiny_run, tmp_path / "dummy.wav", out_dir)
    assert summary["run_id"] == freeze["run_id"]
    assert (out_dir / "summary.json").is_file()
