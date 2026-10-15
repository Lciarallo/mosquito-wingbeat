"""Behavioral contracts for recurrent uncertainty and leakage-safe adaptation."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from torres_lstm import (TorresLSTM, MelSequence, aggregate_groups,
    monte_carlo_probabilities, predict_probabilities, purge_partition_collisions,
    selective_metrics, stochastic_dropout, vote_statistics)

torch.set_num_threads(2)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_explicit_layers_preserve_author_checkpoint_predictions(bidirectional):
    torch.manual_seed(123)
    recurrent = nn.LSTM(32, 8, 2, dropout=.2, batch_first=True, bidirectional=bidirectional)
    linear = nn.Linear(16 if bidirectional else 8, 2)
    source = {"lstm." + k: v for k, v in recurrent.state_dict().items()}
    source.update({"linear." + k: v for k, v in linear.state_dict().items()})
    model = TorresLSTM(hidden=8, bidirectional=bidirectional)
    model.load_author_state(source)
    recurrent.eval(); model.eval()
    x = torch.randn(4, 17, 32)
    with torch.inference_mode():
        sequence, _ = recurrent(x)
        reference = linear(sequence[:, -1])
        actual = model(x)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_mc_is_stochastic_but_deterministic_inference_is_repeatable():
    torch.manual_seed(3)
    model = TorresLSTM(classes=3, hidden=12, dropout=.5)
    x = np.random.default_rng(1).normal(size=(7, 11, 32)).astype("float32")
    a, b = predict_probabilities(model, x), predict_probabilities(model, x)
    np.testing.assert_array_equal(a, b)
    before = [(m, m.training) for m in model.modules()]
    draws = monte_carlo_probabilities(model, x, passes=8, batch_size=3, draw_batch=3)
    assert draws.shape == (8, 7, 3)
    assert np.max(draws.std(0)) > 1e-5
    np.testing.assert_allclose(draws.sum(-1), 1, atol=1e-6)
    assert all(m.training == mode for m, mode in before)


def test_stochastic_context_keeps_batchnorm_frozen_and_restores_modes_on_error():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5))
    model.train()
    model[1].eval()
    states = [m.training for m in model.modules()]
    with pytest.raises(RuntimeError):
        with stochastic_dropout(model):
            assert not model[0].training
            assert model[1].training
            raise RuntimeError("probe")
    assert [m.training for m in model.modules()] == states


def test_vote_uncertainty_matches_binary_paper_definition():
    decisions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]])
    draws = np.eye(2)[decisions]
    result = vote_statistics(draws)
    np.testing.assert_allclose(result["uncertainty"], [0, .25, .5])
    np.testing.assert_array_equal(result["vote_counts"], [[4, 0], [3, 1], [2, 2]])
    np.testing.assert_array_equal(result["prediction"], [0, 0, 0])


def test_multiclass_vote_uncertainty_uses_largest_class_count():
    draws = np.eye(3)[np.array([[0], [1], [2], [2], [2]])]
    stats = vote_statistics(draws)
    assert stats["prediction"].tolist() == [2]
    np.testing.assert_allclose(stats["uncertainty"], [.4])


def test_empty_accepted_subset_is_reported_without_fabricated_accuracy():
    result = selective_metrics([0, 1], [0, 0], [.3, .4], ["a", "b"], 0)
    assert result["accepted"] == 0
    assert result["accuracy"] is None
    assert result["macro_f1"] is None
    assert result["coverage"] == 0
    assert result["confusion_with_abstention"] == [[0, 0, 1], [0, 0, 1]]


def test_selective_accuracy_and_coverage_retain_distinct_denominators():
    result = selective_metrics([0, 1, 1, 0], [0, 1, 0, 1], [0, 0, .2, .3], ["a", "b"], 0)
    assert result["accuracy"] == 1
    assert result["coverage"] == .5
    assert result["overall_correct_decision_rate"] == .5
    assert sum(map(sum, result["confusion_with_abstention"])) == 4


def test_window_aggregation_preserves_mc_draws_and_recording_identity():
    frame = pd.DataFrame({"group": [20, 10, 20], "target": [1, 0, 1]})
    draws = np.array([[[.2, .8], [.9, .1], [.4, .6]], [[.1, .9], [.8, .2], [.3, .7]]])
    groups, y, actual = aggregate_groups(frame, draws)
    assert groups.tolist() == ["10", "20"]
    assert y.tolist() == [0, 1]
    np.testing.assert_allclose(actual, [[[.9, .1], [.3, .7]], [[.8, .2], [.2, .8]]])


def test_mixed_targets_inside_recording_are_rejected():
    frame = pd.DataFrame({"group": ["g", "g"], "target": [0, 1]})
    with pytest.raises(ValueError, match="one target"):
        aggregate_groups(frame, np.ones((2, 2)) / 2)


def test_entire_pcm_component_is_purged_from_development_partitions():
    frame = pd.DataFrame(dict(group=["a", "b", "c", "d"], pcm_block=["shared", "shared", "shared", "other"],
        split=["train", "validation", "test", "train"], content_sha256=["A", "B", "C", "D"]))
    keep, removed = purge_partition_collisions(frame)
    assert keep.group.tolist() == ["c", "d"]
    assert removed.group.tolist() == ["a", "b"]
    assert keep.groupby("pcm_block").split.nunique().max() == 1


def test_missing_pcm_identity_cannot_silently_enter_training():
    frame = pd.DataFrame(dict(group=["a"], pcm_block=[None], split=["train"], content_sha256=["A"]))
    with pytest.raises(ValueError, match="identities"):
        purge_partition_collisions(frame)


def test_frontend_preserves_time_and_is_invariant_to_gain():
    generator = torch.Generator().manual_seed(5)
    wave = torch.randn(2, 15872, generator=generator)
    frontend = MelSequence()
    a, b = frontend(wave), frontend(wave * 2)
    assert a.shape == (2, 59, 32)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
