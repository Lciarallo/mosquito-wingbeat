"""v4: reserved rows never enter development, no cluster leakage, fit/calibration/test separation,
label-free alignment and deterministic fitting/augmentation. Synthetic data; no local corpus needed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from reanalysis import v3_train as v3
from reanalysis import v4_augment as aug
from reanalysis import v4_train as v4

ROOT = Path(__file__).resolve().parents[1]


def synthetic(seed=0, files_per_cluster=6):
    rng = np.random.default_rng(seed)
    layout = {"abuzz": [1, 1, 1, 2, 2, 2], "humbug": [0, 0, 0, 1, 1, 1, 2, 2, 2], "mosquitosong": [1, 1, 1, 2, 2, 2],
              "bdc_unknown": [1, 1, 2], "esc50": [0, 0, 0, 0], "pete_noise": [0, 0, 0, 0]}
    rows, fid = [], 0
    for origin, classes in layout.items():
        for k, target in enumerate(classes):
            for j in range(files_per_cluster):
                rows.append(dict(file_id=fid, path=f"{origin}/{k}/{j}.wav", target=target, origin=origin,
                                 cluster_merged=f"{origin}|{k}", component=f"{origin}|{k}|{j}", role="train",
                                 species="x", duration_s=1.))
                fid += 1
    frame = pd.DataFrame(rows)
    shift = {o: rng.normal(0, 1, 1) for o in layout}
    n = len(frame)
    signal = frame.target.to_numpy()[:, None]
    offset = np.vstack([shift[o] for o in frame.origin])
    arrays = {"perch_white": (rng.normal(0, 1, (n, 160)) + signal + offset).astype("float32"),
              "perch_plain": rng.normal(0, 1, (n, 160)).astype("float32"),
              "tabular_white": (rng.normal(0, 1, (n, 89)) + signal).astype("float32"),
              "tabular_plain": rng.normal(0, 1, (n, 89)).astype("float32")}
    arrays["wingbeat"] = arrays["tabular_white"][:, 80:89]
    arrays["duration"] = np.zeros((n, 2), dtype="float32")
    return frame, arrays


def test_development_only_drops_every_reserved_row():
    frame, arrays = synthetic()
    frame.loc[:9, "role"] = "locked_test"
    frame.loc[10:14, "role"] = "locked_noise"
    frame.loc[15:19, "role"] = "quarantine_holdout_related"
    kept, kept_arrays = v4.development_only(frame, arrays)
    assert set(kept.role) <= set(v4.DEV_ROLES)
    assert len(kept) == len(frame) - 20
    assert np.array_equal(kept_arrays["perch_white"], arrays["perch_white"][20:])
    assert kept_arrays["band_tabular"].shape[1] == 79


def test_band_columns_drop_bands_below_300_hz():
    assert not set(v4.LOW_BANDS) & set(v4.BAND_COLUMNS)
    assert not {40 + b for b in v4.LOW_BANDS} & set(v4.BAND_COLUMNS)
    assert v4.BAND_COLUMNS[-9:] == list(range(80, 89))


def test_lodo_partitions_have_no_cluster_or_path_leakage():
    frame, _ = synthetic()
    # A cluster shared across corpora must be purged from training.
    frame.loc[frame.origin.eq("bdc_unknown") & frame.cluster_merged.eq("bdc_unknown|0"), "cluster_merged"] = "abuzz|0"
    parts = v4.partitions(frame)
    assert [p["tag"] for p in parts] == ["LODO_abuzz", "LODO_humbug", "LODO_mosquitosong", "LODO_esc50", "LODO_pete_noise"]
    for p in parts:
        assert not set(frame.cluster_merged.iloc[p["train"]]) & set(frame.cluster_merged.iloc[p["test"]])
        assert not set(frame.path.iloc[p["train"]]) & set(frame.path.iloc[p["test"]])
        assert set(frame.origin.iloc[p["test"]]) == {p["corpus"]}


def test_fit_calibration_test_are_cluster_disjoint():
    frame, _ = synthetic()
    part = v4.partitions(frame)[1]
    fit, cal = v3.split_fit_calibration(frame, part["train"], v4.SEED)
    sets = [set(frame.cluster_merged.iloc[i]) for i in (fit, cal, part["test"])]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert set(fit) | set(cal) == set(part["train"])


def test_coral_uses_no_labels():
    frame, arrays = synthetic()
    part = v4.partitions(frame)[0]
    fit, cal = v3.split_fit_calibration(frame, part["train"], v4.SEED)
    a = v4.fold_matrix("A1_perch_coral", frame, arrays, None, fit, cal, part["test"], v4.SEED)[0]
    shuffled = frame.copy()
    shuffled["target"] = np.random.default_rng(1).permutation(frame.target.to_numpy())
    b = v4.fold_matrix("A1_perch_coral", shuffled, arrays, None, fit, cal, part["test"], v4.SEED)[0]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ["A0_perch_white", "A2_perch_dann", "B1_band_tabular"])
def test_fitting_is_deterministic(name):
    frame, arrays = synthetic()
    arrays = v4.development_only(frame, arrays)[1]
    part = v4.partitions(frame)[1]
    fit, cal = v3.split_fit_calibration(frame, part["train"], v4.SEED)
    X, extra, _ = v4.fold_matrix(name, frame, arrays, None, fit, cal, part["test"], v4.SEED)
    first = v4.fit_bundle(name, frame, X, extra, fit, cal, v4.SEED).predict(X[part["test"]])
    second = v4.fit_bundle(name, frame, X, extra, fit, cal, v4.SEED).predict(X[part["test"]])
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_augmented_rows_only_enter_fitting_with_half_mass():
    frame, arrays = synthetic()
    arrays = v4.development_only(frame, arrays)[1]
    aug_arrays = {"band_tabular": arrays["band_tabular"] + 5.}
    part = v4.partitions(frame)[1]
    fit, cal = v3.split_fit_calibration(frame, part["train"], v4.SEED)
    X, extra, _ = v4.fold_matrix("C1_aug_band_tabular", frame, arrays, aug_arrays, fit, cal, part["test"], v4.SEED)
    assert extra.shape[0] == len(fit) and np.array_equal(X, arrays["band_tabular"])
    bundle = v4.fit_bundle("C1_aug_band_tabular", frame, X, extra, fit, cal, v4.SEED)
    assert bundle.info["augmented_rows"] == len(fit)
    assert set(bundle.info["calibration_file_ids"]) == set(frame.file_id.iloc[cal])


def test_selection_never_relaxes_targets():
    frame, _ = synthetic()
    point = lambda recall, other, noise: {"point": dict(aedes_recall=recall, fpr_other=other, fpr_noise=noise,
                                                        aedes_precision=.5, auroc=.7)}
    metrics_ = {}
    for name in v4.CANDIDATES:
        for g in ["gated", "ungated"]:
            for c in v4.LODO_MOSQUITO:
                metrics_.setdefault(f"LODO_{c}", {})[f"{name}|{g}"] = point(.19, .01, .0 if c == "humbug" else None)
            for c in v4.LODO_NOISE:
                metrics_.setdefault(f"LODO_{c}", {})[f"{name}|{g}"] = point(None, None, .0)
    few = pd.DataFrame(columns=["candidate", "corpus", "k", "available", "gated", "aedes_recall", "fpr_other"])
    result = v4.select(metrics_, few)
    assert result["eligible"] is False and result["eligibility_level"] is None
    metrics_["LODO_abuzz"]["B1_band_tabular|ungated"] = point(.3, .01, None)
    metrics_["LODO_humbug"]["B1_band_tabular|ungated"] = point(.3, .01, .0)
    metrics_["LODO_mosquitosong"]["B1_band_tabular|ungated"] = point(.3, .02, None)
    result = v4.select(metrics_, few)
    assert result["eligible"] and result["candidate"] == "B1_band_tabular" and result["gated"] is False


def test_augmentation_is_deterministic(tmp_path):
    sr = 8000
    rng = np.random.default_rng(0)
    (tmp_path / "real_rirs_isotropic_noises").mkdir()
    (tmp_path / "pointsource_noises").mkdir()
    rir = np.zeros(800, dtype="float32")
    rir[0], rir[200] = 1, .3
    sf.write(tmp_path / "real_rirs_isotropic_noises/x_rir_a.wav", rir, sr)
    sf.write(tmp_path / "real_rirs_isotropic_noises/x_noise_a.wav", rng.normal(0, .1, sr).astype("float32"), sr)
    sf.write(tmp_path / "pointsource_noises/noise-1.wav", rng.normal(0, .1, sr).astype("float32"), sr)
    rirs, noises = aug.slr28_files(tmp_path)
    x = np.sin(2 * np.pi * 500 * np.arange(2 * sr) / sr).astype("float32")
    a, pa = aug.augment(x, aug.file_rng(7), rirs, noises)
    b, pb = aug.augment(x, aug.file_rng(7), rirs, noises)
    c, _ = aug.augment(x, aug.file_rng(8), rirs, noises)
    assert np.array_equal(a, b) and pa == pb and not np.array_equal(a, c)
    assert np.isfinite(a).all() and np.abs(a).max() <= 1


@pytest.mark.skipif(not (ROOT / "results/v3_continuation/manifest.csv").exists(),
                    reason="Local v3 manifest not distributed")
def test_real_augmentation_items_are_development_only():
    man = pd.read_csv(ROOT / "results/v3_continuation/manifest.csv").set_index("file_id")
    ids = [i for i, _ in aug.development_items()]
    assert set(man.loc[ids, "role"]) <= set(v4.DEV_ROLES)
    assert not set(ids) & set(man.index[man.role.isin(v4.RESERVED)])


@pytest.mark.skipif(not (ROOT / "reanalysis/protocol_v4_freeze.json").exists(), reason="v4 not frozen yet")
def test_v4_freeze_covers_protocol_and_code():
    frozen = json.loads((ROOT / "reanalysis/protocol_v4_freeze.json").read_text())["sha256"]
    for path in ["reanalysis/protocol_v4.json", "reanalysis/v4_train.py", "reanalysis/v4_augment.py",
                 "reanalysis/v4_diagnostics.py", "reports/plano_v4.md"]:
        assert path in frozen
