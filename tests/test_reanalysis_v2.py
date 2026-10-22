"""Integrity tests for reanalysis v2: frozen protocol, cluster isolation, external sets untouched."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reanalysis import train_v2 as tv

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reanalysis/manifest_files.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def man():
    m = pd.read_csv(MANIFEST)
    m["n_windows"] = 1
    return m


def test_protocol_is_frozen():
    freeze = json.loads((ROOT / "reanalysis/protocol_freeze.json").read_text())
    assert sha(ROOT / "reanalysis/protocol_v2.json") == freeze["protocol_sha256"]
    assert sha(MANIFEST) == freeze["manifest_sha256"]


def test_excluded_sources_absent(man):
    assert not man.source.isin(["figshare", "torres", "torres_2025"]).any()
    assert not man.path.str.contains("drive_audio", case=False).any()


def test_external_never_in_development(man):
    dev = tv.dev_files(man)
    assert not man.loc[dev, "role"].eq("external").any()
    assert set(man.loc[man.role == "external", "source"]) == {"mosquitosong", "ufrgs"}


def test_outer_folds_disjoint_by_cluster_and_contain_aedes(man):
    files = tv.dev_files(man)
    folds = tv.outer_folds(man, files)
    seen = []
    for tr, te in folds:
        assert not set(man.cluster[tr]) & set(man.cluster[te])
        assert (man.target[te] == 2).any()
        seen.extend(te)
    assert sorted(seen) == sorted(files)


def test_inner_splits_stay_inside_outer_training(man):
    files = tv.dev_files(man)
    tr, te = tv.outer_folds(man, files)[0]
    for a, b in tv.inner_splits(man, tr):
        assert set(a) | set(b) <= set(tr)
        assert not set(man.cluster[a]) & set(man.cluster[b])


def test_sample_weights_file_and_class_balance():
    win_files = np.array([0, 0, 0, 1, 2, 2])
    target = np.array([0, 1, 1])
    w = tv.sample_weights(win_files, target)
    per_file = pd.Series(w).groupby(win_files).sum()
    assert np.isclose(per_file[1], per_file[2])  # same class -> same mass
    per_class = pd.Series(w).groupby(target[win_files]).sum()
    assert np.isclose(per_class[0], per_class[1])


def test_aggregate_is_geometric_mean_softmax():
    p = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
    out = tv.aggregate(np.array([5, 5, 7]), p, np.array([5, 7]))
    g = np.exp(np.log(p[:2]).mean(0)); g /= g.sum()
    assert np.allclose(out[0], g) and np.allclose(out[1], p[2] / p[2].sum())


def test_point_metrics_known_case():
    df = pd.DataFrame(dict(target=[0, 1, 2, 2], pred=[0, 2, 2, 1], cluster=list("abcd"),
                           p0=[.8, .1, .1, .1], p1=[.1, .3, .1, .6], p2=[.1, .6, .8, .3]))
    m = tv.point_metrics(df)
    assert m["aedes_recall"] == 0.5 and m["aedes_precision"] == 0.5 and m["fpr_aedes_in_other"] == 1.0
    assert np.isclose(m["balanced_accuracy"], (1 + 0 + 0.5) / 3)
