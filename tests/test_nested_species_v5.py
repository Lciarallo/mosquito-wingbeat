"""V5 physical, channel-invariance, taxonomy, calibration-lineage and immutable-history regressions."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits
from acoustic_features_v5 import (
    SR, FEATURE_NAMES, PSD_GRID, window_features, extract_features,
    local_noise_floor, centred_cepstrum,
)
from nested_species_ensemble_v5 import (
    OUT, V4, check_partition, verify_freeze, verify_history,
    marginalize_genus_v5, select_genus_parameters, pool_recording,
    choose_pooling, fit_blend_v5, mix_probabilities_v5, predict_features,
)
from research_metrics_v2 import classification_metrics


def catalogue():
    return np.array([f'{g} sp{i}' for g, n in [('GenA', 4), ('GenB', 12), ('GenC', 3), ('GenD', 1)] for i in range(n)])


def tone(f0=400):
    t = np.arange(SR) / SR
    return np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t) + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)


@pytest.mark.parametrize('f0', [400, 625])
def test_stationary_white_floor_25_to_12db_and_8khz_hum_invariance(f0):
    rng = np.random.default_rng(101)
    sig = tone(f0)
    noise = rng.normal(size=SR)
    noise /= np.sqrt(np.mean(noise ** 2))
    rms = np.sqrt(np.mean(sig ** 2))
    a, sa = window_features(sig + noise * rms * 10 ** (-25 / 20))
    b, sb = window_features(sig + noise * rms * 10 ** (-12 / 20))
    for ids in [np.arange(len(PSD_GRID)), np.array([FEATURE_NAMES.index(f'mfcc_centred_mean_{i}') for i in range(1, 13)])]:
        cosine = np.dot(a[ids], b[ids]) / (np.linalg.norm(a[ids]) * np.linalg.norm(b[ids]))
        assert cosine > 0.98
    # Verify 8 kHz anti-aliasing + 60 Hz harmonic hum invariance
    w_8k = resample_poly(resample_poly(sig + noise * rms * 10 ** (-12 / 20), 1, 2), 2, 1)
    t = np.arange(SR) / SR
    w_hum = w_8k + 0.5 * rms * np.sin(2 * np.pi * 60 * t) + 0.2 * rms * np.sin(2 * np.pi * 180 * t) + 0.2 * rms * np.sin(2 * np.pi * 300 * t)
    d, _ = window_features(w_hum)
    cos_8k = np.dot(b[:len(PSD_GRID)], d[:len(PSD_GRID)]) / (np.linalg.norm(b[:len(PSD_GRID)]) * np.linalg.norm(d[:len(PSD_GRID)]))
    assert cos_8k > 0.99
    assert abs(a[FEATURE_NAMES.index('shs_long_peak0_f0')] - b[FEATURE_NAMES.index('shs_long_peak0_f0')]) <= 5
    assert abs(sa - sb) < 0.1


def test_log_floor_ratio_is_finite_and_signed():
    power = np.array([0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 2.0]) * 1e-5
    floor, snr = local_noise_floor(power, 5)
    np.testing.assert_allclose(snr, 10 * np.log10(power / floor))
    assert np.isfinite(snr).all() and snr.min() > -10 and snr.max() < 10
    assert np.any(snr < 0)


def test_cepstrum_is_scale_normalized_and_retains_shape():
    rng = np.random.default_rng(8)
    p = np.exp(rng.normal(size=(801, 91)))
    floor, _ = local_noise_floor(p, 10)
    a = centred_cepstrum(p, floor)
    b = centred_cepstrum(p * 100, floor * 100)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-11)
    assert a[0][0] == 0 and np.isclose(np.sqrt(np.mean(a[0][1:] ** 2)), 2.5, atol=1e-5)


def test_frontend_is_bit_exact_inductive_and_finite_on_silence():
    waves = np.stack([tone(), np.zeros(SR), np.random.default_rng(5).normal(size=SR)])
    one, s = window_features(waves[0])
    before, sa = extract_features(waves)
    waves[1:] = 999
    after, sb = extract_features(waves)
    assert np.array_equal(one, before[0]) and np.array_equal(before[0], after[0])
    assert s == sa[0] == sb[0] and np.isfinite(before).all()


@pytest.mark.parametrize('gain', [0.01, 10.0])
def test_gain_invariance(gain):
    w = tone() + np.random.default_rng(11).normal(0, 0.01, SR)
    a, _ = window_features(w)
    b, _ = window_features(w * gain)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize('column', ['group', 'collision_group', 'pcm_block', 'path', 'content_sha256'])
def test_no_shared_source_identity_between_partitions(column):
    f = pd.DataFrame(dict(
        label=['a', 'b'] * 3, group=list('abcdef'), collision_group=list('abcdef'),
        pcm_block=list('abcdef'), path=list('ABCDEF'), content_sha256=list('123456'),
    ))
    check_partition(f, [0, 1], [2, 3], [4, 5])
    f.loc[4, column] = f.loc[0, column]
    with pytest.raises(ValueError, match='leakage'):
        check_partition(f, [0, 1], [2, 3], [4, 5])


def test_power_taxonomy_formula_and_exact_sum_special_case():
    classes = catalogue()
    p = np.random.default_rng(3).dirichlet(np.ones(20), size=3)
    g, q = marginalize_genus_v5(p, classes, 0.75, 2)
    score = np.column_stack([np.sqrt((p[:, np.char.startswith(classes, x)] ** 2).sum(1) / sum(np.char.startswith(classes, x)) ** 0.75) for x in g])
    np.testing.assert_allclose(q, score / score.sum(1, keepdims=True), rtol=1e-12, atol=1e-15)
    _, exact = marginalize_genus_v5(p, classes, 0, 1)
    expected = np.column_stack([p[:, np.char.startswith(classes, x)].sum(1) for x in g])
    np.testing.assert_allclose(exact, expected, rtol=1e-12, atol=1e-15)


def test_species_prior_is_bounded_and_calibration_only():
    c = np.array(['GenA sp0', 'GenA sp1', 'GenC sp0'])
    y = np.repeat(np.arange(3), [7, 30, 12])
    n = len(y)
    p = np.tile(np.eye(3)[y] * 0.9 + 0.1 / 3, (3, 1, 1))
    f = pd.DataFrame(dict(group=[str(i) for i in range(n)], collision_group=[str(i) for i in range(n)], label=c[y]))
    rule = fit_blend_v5(p, y, np.ones(n), f, np.ones(n), np.ones(n), c, partition='calibration')
    assert np.max(np.abs(rule['log_prior'])) <= 0.8
    assert rule['prior_strength'] <= 0.20
    with pytest.raises(ValueError, match='calibration'):
        fit_blend_v5(p, y, np.ones(n), f, np.ones(n), np.ones(n), c, partition='test')
