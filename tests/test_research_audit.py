"""Regression checks for the scientific audit, independent of trained weights."""
import numpy as np
import pandas as pd
import pytest
from research_metrics import binary_metrics, cluster_intervals
from loso_generalization import ACOUSTIC, consistent_features, group_auc_ci


def test_dryad_mobile_phone_corpus_is_acoustic():
    assert 'dryad' in ACOUSTIC


def test_consistency_does_not_fill_remaining_slots_with_disagreeing_features():
    frame = pd.DataFrame({'domain': ['a']*4 + ['b']*4})
    y = np.tile([0, 0, 1, 1], 2)
    x = np.column_stack([y, np.r_[y[:4], 1-y[4:]], np.ones(8)])
    assert consistent_features(x, y, frame, 40).tolist() == [0]


def test_consistency_needs_two_informative_sources():
    with pytest.raises(ValueError, match='two training sources'):
        consistent_features(np.arange(4).reshape(-1, 1), [0, 0, 1, 1], pd.DataFrame({'domain': ['a']*4}), 40)


@pytest.mark.parametrize('y,p,q', [([0, 2], [.1, .2], [1, 1]),
                                  ([0, 1], [np.nan, .2], [1, 1]),
                                  ([0, 1], [.1, 1.2], [1, 1]),
                                  ([0, 1], [.1], [1, 1]),
                                  ([0, 1], [.1, .2], ['False', 'True'])])
def test_invalid_inputs_cannot_silently_change_metrics(y, p, q):
    with pytest.raises(ValueError):
        binary_metrics(y, p, q, .5)


def test_invalid_audio_and_unknown_labels_keep_correct_denominators():
    r = binary_metrics([0, 0, 1, 1, -1], [.1, .9, .9, .1, .9], [False, True, True, False, True], .5)
    assert r['confusion'] == [[0, 1, 1], [0, 1, 1]]
    assert r['specificity'] == 0 and r['aedes_recall'] == .5
    assert r['aedes_precision'] == .5 and r['unknown_windows'] == 1


def test_empty_and_positive_only_samples_have_explicit_undefined_metrics():
    assert binary_metrics([], [], [], .5)['accuracy'] is None
    r = binary_metrics([1, 1], [.9, .8], [True, True], .5)
    assert not r['precision_informative_for_binary_detection']
    assert r['specificity'] is None and r['macro_f1'] is None


def test_cluster_resampling_is_invariant_to_repeating_all_windows():
    args = [np.array(v) for v in ([0, 0, 1, 1], [.1, .9, .9, .1], [1, 1, 1, 0], ['a', 'b', 'c', 'd'])]
    original = cluster_intervals(*args, replicates=100)
    repeated = cluster_intervals(*(np.tile(v, 4) for v in args), replicates=100)
    assert original == repeated
    assert original['metrics']['aedes_precision']['undefined_replicates'] > 0


def test_one_group_cannot_provide_recording_level_uncertainty():
    ci = cluster_intervals([0, 1], [.1, .9], [True, True], ['a', 'a'], replicates=100)
    assert all(v['lower95'] is None for v in ci['metrics'].values())
    assert group_auc_ci(np.array([1, 1]), np.array([.8, .9]), np.array(['a', 'b']), np.random.default_rng(1), n=10) == (None, None)
