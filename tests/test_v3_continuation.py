"""Scientific contracts: independence, honest denominators and frozen selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from reanalysis import v3_continue_manifest as manifest
from reanalysis import v3_features as features
from reanalysis import v3_metrics as metrics
from reanalysis import v3_train as train

ROOT = Path(__file__).resolve().parents[1]


def test_actual_manifest_has_no_training_final_overlap():
    path = ROOT / "results/v3_continuation/manifest.csv"
    if not path.exists():
        import pytest
        pytest.skip("Local continuation manifest not distributed")
    frame = pd.read_csv(path)
    development = frame[frame.role.isin(["train", "noise_train"])]
    final = frame[frame.role.isin(["locked_test", "locked_noise"])]
    for key in ["path", "component", "cluster_merged"]:
        assert not set(development[key]) & set(final[key])
    assert frame[frame.role.eq("locked_noise")].target.eq(0).all()
    assert not frame[frame.role.eq("quarantine_holdout_related")].empty


def test_shared_cluster_is_quarantined_without_altering_original():
    rows = [dict(path="train", target=1, role="train", cluster_merged="shared", component="a", origin="known"),
            dict(path="test", target=2, role="locked_test", cluster_merged="shared", component="b", origin="locked")]
    frame = pd.DataFrame(rows)
    all_rows = frame.assign(raw_source=["biodcase_D1", "biodcase_D3"])
    result = manifest.prepare(frame, all_rows, {"test"})
    assert frame.role.tolist() == ["train", "locked_test"]
    assert result.role.tolist() == ["quarantine_holdout_related", "locked_test"]
    assert result.loc[1, "previously_scored_component"]
    assert result.loc[1, "provenance_note"] == "D3_related_origin_unconfirmed"


def test_zero_padding_short_audio_does_not_repeat_waveform():
    x = np.sin(2 * np.pi * 450 * np.arange(1600) / 8000).astype("float32")
    chunks = features.audio_chunks(x)
    assert len(chunks) == 1 and chunks[0].shape == (160000,)
    assert np.count_nonzero(chunks[0][:70000]) == 0
    assert np.count_nonzero(chunks[0][-70000:]) == 0
    assert np.isclose(np.sqrt(np.mean(chunks[0] ** 2)), .05, rtol=1e-5)
    assert features.acoustic_features(x).shape == (89,)
    assert np.isfinite(features.acoustic_features(np.zeros(100, dtype="float32"))).all()


def test_threshold_can_abstain_everywhere_instead_of_breaking_fpr_caps():
    y = np.array([0] * 100 + [1] * 100 + [2] * 20)
    presence = np.ones(len(y))
    tied = np.full(len(y), .7)
    assert train.threshold(y, tied, presence) > 1
    score = np.r_[np.linspace(0, .4, 100), np.linspace(.2, .8, 100), np.linspace(.7, 1., 20)]
    tau = train.threshold(y, score, presence)
    alert = score >= tau
    assert alert[y == 0].mean() <= .01 and alert[y == 1].mean() <= .02


def test_abstentions_remain_in_all_denominators():
    frame = pd.DataFrame(dict(target=[0, 1, 2, 2], pred=[3, 3, 2, 3], score=[.1, .2, .9, .8],
                              cluster_merged=list("abcd")))
    result = metrics.point(frame)
    assert result["aedes_recall"] == .5
    assert result["aedes_precision"] == 1
    assert result["fpr_other"] == 0 and result["fpr_noise"] == 0
    assert result["coverage"] == .25


def test_bootstrap_rank_metrics_match_weighted_reference_with_ties():
    y = np.array([0, 1, 2, 2, 1, 2])
    score = np.array([.2, .2, .2, .8, .8, .9])
    weights = np.array([[1, 1, 1, 1, 1, 1], [2, 1, 3, 0, 2, 1]], dtype=float)
    auc, ap = metrics._rank_metrics(y, score, weights)
    for k, w in enumerate(weights):
        assert np.isclose(auc[k], roc_auc_score(y == 2, score, sample_weight=w))
        assert np.isclose(ap[k], average_precision_score(y == 2, score, sample_weight=w))


def test_missing_container_duration_uses_existing_metadata_without_label_imputation():
    frame = pd.DataFrame(dict(path=["a.mp4", "b.wav"], duration_s=[np.nan, 2.]))
    recovered = train.repair_duration_metadata(frame, pd.Series({"a.mp4": 3.25}))
    assert recovered.tolist() == [3.25, 2.]
    assert np.isnan(frame.duration_s.iloc[0])  # the frozen manifest is not rewritten
    import pytest
    with pytest.raises(ValueError, match="Missing physical duration"):
        train.repair_duration_metadata(frame, pd.Series({"other.mp4": 1.}))


def test_no_alert_precision_is_undefined_and_counted():
    frame = pd.DataFrame(dict(target=[0, 1, 2, 2], pred=[0, 1, 3, 3], score=[.1, .2, .8, .9],
                              cluster_merged=list("abcd")))
    result = metrics.with_uncertainty(frame, draws=64)
    assert result["point"]["aedes_precision"] is None
    assert result["undefined_draws"]["aedes_precision"] == 64
    assert "zero_noise_errors_bootstrap_cannot_bound_unseen_errors" in result["cautions"]


def test_fewshot_excludes_every_file_from_touched_clusters():
    rows = []
    for target in [1, 2]:
        for cluster in range(4):
            rows.extend(dict(target=target, cluster_merged=f"{target}_{cluster}") for _ in range(10))
    frame = pd.DataFrame(rows)
    cal, evaluate, error = train.choose_fewshot(frame, np.arange(len(frame)), k=5, seed=1)
    assert error is None and len(cal) == 10
    assert not set(frame.cluster_merged.iloc[cal]) & set(frame.cluster_merged.iloc[evaluate])
    assert set(frame.target.iloc[evaluate]) == {1, 2}


def test_mmd_is_differentiable_without_using_test_data():
    torch.manual_seed(1)
    x = torch.randn(12, 4, requires_grad=True)
    domains = torch.tensor([0] * 6 + [1] * 6)
    penalty = train.mmd_penalty(x, domains)
    penalty.backward()
    assert torch.isfinite(penalty) and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_fit_scalers_and_calibration_use_separate_clusters():
    rng = np.random.default_rng(8)
    rows = [dict(file_id=i, target=(i // 4) % 3, origin="source", cluster_merged=f"c{i // 4}") for i in range(120)]
    frame = pd.DataFrame(rows)
    X = rng.normal(size=(120, 9)).astype("float32")
    X[:, 0] += frame.target.to_numpy() * 2
    fit, cal = train.split_fit_calibration(frame, np.arange(120), seed=1)
    X[cal] += 50  # A sentinel that would visibly contaminate a full-data scaler.
    bundle = train.fit_bundle("C_wingbeat", frame, {"wingbeat": X}, fit, cal, seed=2)
    assert np.allclose(bundle.stage1.scaler.mean_, X[fit].mean(0), atol=1e-5)
    assert not set(bundle.info["fit_file_ids"]) & set(bundle.info["calibration_file_ids"])


def test_parent_and_continuation_design_freezes_match():
    frozen = json.loads((ROOT / "reanalysis/protocol_v3_continuation_freeze.json").read_text())
    for path, sha in frozen["sha256"].items():
        file = ROOT / path
        if file.exists():
            assert hashlib.sha256(file.read_bytes()).hexdigest() == sha, path


def test_final_is_frozen_before_scoring_and_is_not_adaptively_repeated(tmp_path, monkeypatch):
    """A synthetic end-to-end run checks the actual final-evaluation boundary."""
    rng = np.random.default_rng(11)
    rows = [dict(file_id=i, path=f"file{i}.wav", target=(i // 4) % 3, origin="source", raw_source="source",
                 species="test", cluster_merged=f"c{i // 4}", role="train", previously_scored_component=False,
                 provenance_note="synthetic_test_fixture", duration_s=1.) for i in range(120)]
    for i, target in enumerate([1, 1, 2, 2, 0, 0], start=120):
        rows.append(dict(file_id=i, path=f"file{i}.wav", target=target, origin="held", raw_source="held",
                         species="test", cluster_merged=f"held{i // 2}", role="locked_noise" if target == 0 else "locked_test",
                         previously_scored_component=False, provenance_note="synthetic_test_fixture", duration_s=1.))
    frame = pd.DataFrame(rows)
    arrays = {"wingbeat": rng.normal(size=(len(frame), 9)).astype("float32")}
    monkeypatch.setattr(train, "OUT", tmp_path)
    monkeypatch.setattr(train, "ensure_training_identity", lambda: {"synthetic_fixture": True})
    selection = dict(candidate="C_wingbeat", eligible=False, candidates=[], reason="synthetic contract test")
    train.write_json(tmp_path / "selection.json", selection)
    train.write_json(tmp_path / "development_metrics.json", {"synthetic_fixture": True})
    real_prediction_frame = train.prediction_frame
    calls = []
    def guarded_prediction(*args, **kwargs):
        assert (tmp_path / "final_freeze.json").exists()
        calls.append(1)
        return real_prediction_frame(*args, **kwargs)
    monkeypatch.setattr(train, "prediction_frame", guarded_prediction)
    train.final(frame, arrays)
    assert len(calls) == 1
    train.final(frame, arrays)
    assert len(calls) == 1
    train.write_json(tmp_path / "selection.json", dict(selection, candidate="A_perch_plain"))
    import pytest
    with pytest.raises(ValueError, match="Final inputs changed"):
        train.final(frame, arrays)
