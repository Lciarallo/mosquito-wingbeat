import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import notebook_transfer_features as f
import reproduce_notebook_transfer as study


@pytest.fixture
def harmonic_wave():
    t = np.arange(f.SR) / f.SR
    return (.2*np.sin(2*np.pi*570*t)+.07*np.sin(2*np.pi*1140*t)+.03*np.sin(2*np.pi*1710*t)).astype(np.float32)


def test_received_notebook_original_bytes_are_preserved():
    manifest = json.loads((f.REFERENCE.parent/'source_manifest.json').read_text())
    assert hashlib.sha256(f.REFERENCE.read_bytes()).hexdigest() == manifest['sha256']


def test_reference_61_features_exactly_match_received_functions(harmonic_wave):
    original = f.load_reference_functions(cache_filterbank=False)
    expected = original['extract_features_1s'](original['pre_processar_audio'](harmonic_wave, f.SR), f.SR)
    assert len(expected) == 61
    np.testing.assert_array_equal(f.reference_window_features(harmonic_wave), np.array(list(expected.values()), np.float32))


def test_feature_batches_are_bit_inductive(harmonic_wave):
    other = np.random.default_rng(7).normal(0, .05, f.SR).astype(np.float32)
    a, b = f.extract_features([harmonic_wave, other])
    c, d = f.extract_features([harmonic_wave, 100*other])
    np.testing.assert_array_equal(a[0], c[0])
    np.testing.assert_array_equal(b[0], d[0])
    np.testing.assert_array_equal(a[0], f.reference_window_features(harmonic_wave))


def test_relative_lfcc_retains_shape_and_is_gain_invariant(harmonic_wave):
    first = f.relative_lfcc_features(harmonic_wave.astype(np.float64))
    gained = f.relative_lfcc_features(harmonic_wave.astype(np.float64)*17)
    np.testing.assert_allclose(first, gained, rtol=1e-5, atol=1e-5)
    assert first[0] == 0
    assert np.linalg.norm(first[1:13]) > .1


def test_silent_features_are_finite():
    a, b = f.extract_features([np.zeros(f.SR, np.float32)])
    assert a.shape == (1, 61) and b.shape == (1, 26)
    assert np.isfinite(a).all() and np.isfinite(b).all()


@pytest.mark.parametrize('wave', [np.zeros(100), np.zeros((f.SR, 2)), np.full(f.SR, np.nan)])
def test_invalid_window_shape_or_values_are_rejected(wave):
    with pytest.raises(ValueError):
        f.reference_window_features(wave)
    with pytest.raises(ValueError):
        f.relative_lfcc_features(wave)


def test_original_recording_and_pcm_partitions_must_be_disjoint():
    records = pd.DataFrame({'group': ['one','two','three'], 'pcm_block':['a','b','c']})
    audit = study.check_groups(records, np.array([0]), np.array([1]), np.array([2]))
    assert audit['group_overlap'] == audit['pcm_block_overlap'] == 0
    records.loc[1, 'pcm_block'] = 'a'
    with pytest.raises(ValueError, match='pcm_block'):
        study.check_groups(records, np.array([0]), np.array([1]), np.array([2]))


def test_predictions_do_not_depend_on_other_test_rows():
    rng = np.random.default_rng(7); x = rng.normal(size=(40, 4)); labels=np.where(x[:,0]>0, 'yes', 'no')
    model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression())]).fit(x,labels)
    batch=x[:3].copy(); one=study.align_probabilities(model,batch[:1],model.classes_)
    batch[1:]=1e6
    changed=study.align_probabilities(model,batch,model.classes_)
    np.testing.assert_array_equal(one[0],changed[0])


def test_binary_cutoff_is_fit_from_calibration_negatives():
    y=np.array(['Aedes aegypti']*3+['other_species_or_noise']*40)
    score=np.r_[.7,.8,.9,np.linspace(.05,.65,40)]
    t=study.select_aegypti_cutoff(y,score)
    assert np.mean(score[3:]>=t) <= .05
    assert t >= .5


def test_completed_study_models_scalers_never_use_calibration_or_test():
    import joblib
    out=study.OUT
    records=pd.read_csv(out/'recordings.csv');features=np.load(out/'feature_bags.npz')
    for path in sorted((out/'models').glob('*.joblib')):
        bundle=joblib.load(path);s=bundle['settings'];fit=np.array(s['fit_recordings']);cal=np.array(s['calibration_recordings'])
        assert not set(records.iloc[fit].pcm_block)&set(records.iloc[cal].pcm_block)
        np.testing.assert_allclose(bundle['model'].named_steps['scale'].mean_,features[s['variant']][fit].mean(axis=0),rtol=1e-12,atol=1e-12)
    assert len(list((out/'models').glob('*.joblib'))) == 48
    assert study.verify_freeze(out)['no_outer_or_external_predictions_yet']
