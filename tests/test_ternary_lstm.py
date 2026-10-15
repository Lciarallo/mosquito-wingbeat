"""Behavioral contracts for the neural three-class comparison."""
import numpy as np
import pandas as pd
import torch
import pytest

import train_ternary_lstm as study
from ternary_protocol import activity_pool


def example_frame():
    groups = np.repeat([f"g{i}" for i in range(12)], 2)
    targets = np.repeat(np.tile([0, 1, 2], 4), 2)
    return pd.DataFrame(dict(row_id=np.arange(24), group=groups, pcm_block=groups, path=groups,
                             true_label=targets, domain="example"))


def test_pool_matches_existing_activity_rule_and_mixed_group_truth():
    f = example_frame().iloc[:6].copy()
    f.loc[4, "true_label"] = 0
    p = np.array([[.8,.1,.1],[.4,.2,.4],[.6,.3,.1],[.2,.7,.1],[.7,.1,.2],[.1,.1,.8]])
    pooled = study.GroupPool(f)
    np.testing.assert_allclose(pooled.probabilities(p), np.stack([activity_pool(p[i:i+2]) for i in range(0,6,2)]))
    assert pooled.truth.tolist() == [0,1,2]


def test_pool_preserves_noise_fallback_and_row_permutation_invariance():
    f = example_frame().iloc[:4]
    p = np.array([[1.,0.,0.],[1.,0.,0.],[.2,.6,.2],[.4,.3,.3]])
    a = study.GroupPool(f).probabilities(p)
    order = [3,0,2,1]
    np.testing.assert_allclose(a, study.GroupPool(f.iloc[order]).probabilities(p[order]))
    np.testing.assert_allclose(a[0], [1,0,0])


def test_fitted_normalization_cannot_use_heldout_rows(tmp_path):
    raw = np.random.default_rng(1).normal(size=(8,11,64)).astype("float32")
    fit = np.array([0,1,2,3])
    study.normalized(raw, fit, tmp_path/"first.npz")
    raw[4:] += 1000
    study.normalized(raw, fit, tmp_path/"second.npz")
    with np.load(tmp_path/"first.npz") as a, np.load(tmp_path/"second.npz") as b:
        np.testing.assert_array_equal(a["mean"],b["mean"])
        np.testing.assert_array_equal(a["std"],b["std"])


@pytest.mark.parametrize("architecture", ["lstm", "bilstm"])
def test_real_selection_refit_returns_frozen_ternary_policy(tmp_path, monkeypatch, architecture):
    settings = dict(study.SETTINGS, hidden=8, max_epochs=1, patience=1, batch_size=8)
    monkeypatch.setattr(study,"SETTINGS",settings)
    f = example_frame()
    raw = np.random.default_rng(2).normal(size=(len(f),11,64)).astype("float32")
    model, x, selected = study.run_context(architecture,"tiny",f,raw,np.arange(len(f)),
                                          torch.device("cpu"),tmp_path/"tiny",123)
    assert selected["best_epoch"] == 1
    assert selected["policy"]["selection_scope"] == "inner_holdout_within_outer_training"
    assert model.config["classes"] == 3
    assert model.config["bidirectional"] == (architecture == "bilstm")
    p = study.predict_probabilities(model,x)
    assert p.shape == (len(f),3)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    assert study.read_json(tmp_path/"tiny/freeze.json")["frozen_before_outer_test"]


def test_actual_8khz_frontend_retains_55_steps_and_64_bands():
    wave = torch.randn(2,7936,generator=torch.Generator().manual_seed(2))
    f = study.frontend()
    assert f.config["sample_rate"] == 8000
    assert f(wave).shape == (2,55,64)
    torch.testing.assert_close(f(wave),f(wave*3+.2),rtol=1e-4,atol=1e-4)
