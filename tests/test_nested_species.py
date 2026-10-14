"""Regression checks for outer-test leakage and unsafe probability blending."""
import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path
import joblib

from nested_species_ensemble import check_partition, fit_blend, blend, select_threshold


def frame():
    return pd.DataFrame(dict(label=['a','b']*3,group=list('abcdef'),
                             path=list('ABCDEF'),content_sha256=list('123456')))


@pytest.mark.parametrize('column',['group','path','content_sha256'])
def test_derived_audio_cannot_cross_fit_calibration_test(column):
    f=frame()
    f.loc[4,column]=f.loc[0,column]
    with pytest.raises(ValueError,match='leakage'):
        check_partition(f,[0,1],[2,3],[4,5])


def test_calibration_and_test_row_overlap_fails_before_fit():
    with pytest.raises(ValueError,match='Row leakage'):
        check_partition(frame(),[0,1],[2,3],[3,5])


def test_class_missing_from_fit_is_not_silently_assumed():
    with pytest.raises(ValueError,match='missing a target class'):
        check_partition(frame(),[0],[2,3],[4,5])


def test_ensemble_and_threshold_cannot_be_fit_on_test():
    p=np.array([[[.9,.1],[.1,.9]]]*3)
    with pytest.raises(ValueError,match='calibration partition'):
        fit_blend(p,[0,1],[1,1],partition='test')
    with pytest.raises(ValueError,match='calibration groups'):
        select_threshold(frame().iloc[:2],p[0],['a','b'],partition='test')


def test_blend_is_stable_for_one_clip_and_mixed_batches():
    p=np.array([[[.9,.1],[.2,.8]],[[.8,.2],[.4,.6]],[[.7,.3],[.1,.9]]])
    parameters=fit_blend(p,[0,1],[1,1],partition='calibration')
    full=blend(p,parameters)
    separate=np.vstack([blend(p[:,i:i+1],parameters) for i in range(2)])
    np.testing.assert_allclose(full,separate)
    assert np.allclose(full.sum(1),1)


def test_tiny_calibration_does_not_claim_high_precision_on_one_emission():
    selected=select_threshold(frame().iloc[:2],np.array([[.99,.01],[.01,.99]]),
                              ['a','b'],partition='calibration')
    assert selected['threshold']>1 and selected['calibration_emitted']==0


def test_nonfinite_probabilities_cannot_enter_calibration():
    p=np.array([[[np.nan,.1],[.1,.9]]]*3)
    with pytest.raises(ValueError,match='finite'):
        fit_blend(p,[0,1],[1,1],partition='calibration')


def test_executed_partitions_keep_the_outer_test_out_of_both_fit_and_calibration():
    root=Path(__file__).resolve().parents[1]
    parts=pd.read_csv(root/'results/species_nested_2026_10_09/partitions.csv')
    for _,fold in parts.groupby('fold'):
        for column in ['row_index','group','path','content_sha256']:
            assert fold.groupby(column).partition.nunique().max()==1


def test_saved_scalers_used_only_their_fit_rows():
    root=Path(__file__).resolve().parents[1]
    base=np.load(root/'data/cache/features.npz')['features']
    for fold in ['0','final']:
        saved=joblib.load(root/f'results/species_nested_2026_10_09/models/{fold}_svm.joblib')
        scaler=saved['model'].steps[0][1]
        np.testing.assert_allclose(scaler.mean_,base[saved['fit_rows']].mean(0,dtype='float64'),rtol=1e-9)
        assert set(saved['fit_rows']).isdisjoint(saved['calibration_rows'])
        assert set(saved['fit_rows']).isdisjoint(saved['test_rows'])


def test_new_oof_has_explicit_identity_and_fixed_ensemble_is_the_declared_mean():
    root=Path(__file__).resolve().parents[1]
    frame=pd.read_csv(root/'results/segments.csv')
    a=np.load(root/'results/species_nested_2026_10_09/oof_probabilities.npz')
    rows=a['row_indices']
    np.testing.assert_array_equal(a['groups'],frame.iloc[rows].group)
    np.testing.assert_array_equal(a['folds'],frame.iloc[rows].fold_species)
    np.testing.assert_allclose(a['fixed'],(a['svm']+a['extra_trees']+a['hist_gradient_boosting'])/3,atol=1e-7)
    for name in ['svm','extra_trees','hist_gradient_boosting','fixed','calibrated']:
        assert np.isfinite(a[name]).all()
        np.testing.assert_allclose(a[name].sum(1),1,atol=1e-6)


def test_frozen_corrected_models_and_thresholds_still_match_the_manifest():
    from nested_species_ensemble import file_hash
    root=Path(__file__).resolve().parents[1]
    freeze=json.loads((root/'results/species_nested_2026_10_09/freeze.json').read_text())
    for relative,digest in freeze['artifacts_sha256'].items():
        assert file_hash(root/relative)==digest
