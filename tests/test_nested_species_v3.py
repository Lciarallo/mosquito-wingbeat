"""Physical/frontend and methodological regression checks for the v3 round."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from acoustic_features_v3 import (SR,FEATURE_NAMES,window_features,extract_features,
    stft_power,local_noise_floor,harmonic_energy_ratios,SHORT_NPERSEG,SHORT_HOP,LONG_NPERSEG,LONG_HOP)
from nested_species_ensemble_v3 import (ROOT,OUT,V2,check_partition,select_genus_threshold,
    verify_history,verify_freeze,predict_features)
from nested_species_ensemble_v2 import marginalize_genus
from research_metrics_v2 import classification_metrics


def tone():
    t=np.arange(SR)/SR
    return np.sin(2*np.pi*400*t)+.5*np.sin(2*np.pi*800*t)+.25*np.sin(2*np.pi*1200*t)


def frame():
    return pd.DataFrame(dict(label=['a','b']*3,group=list('abcdef'),collision_group=list('abcdef'),
                             pcm_block=list('abcdef'),path=list('ABCDEF'),content_sha256=list('123456')))


@pytest.mark.parametrize('column',['group','collision_group','pcm_block','path','content_sha256'])
def test_no_shared_source_identity_between_partitions(column):
    f=frame();f.loc[4,column]=f.loc[0,column]
    with pytest.raises(ValueError,match='leakage'):check_partition(f,[0,1],[2,3],[4,5])


def test_c0_is_zero_and_low_order_timbre_survives():
    f,_=window_features(tone());t=np.arange(SR)/SR
    changed=np.sin(2*np.pi*400*t)+1.4*np.sin(2*np.pi*800*t)+.7*np.sin(2*np.pi*1200*t)
    g,_=window_features(changed)
    ids=[FEATURE_NAMES.index('mfcc_centred_mean_'+str(i)) for i in (0,1,2)]
    assert f[ids[0]]==g[ids[0]]==0
    assert np.any(np.abs(f[ids[1:]])>1)
    assert not np.allclose(f[ids[1:]],g[ids[1:]],atol=.1)


@pytest.mark.parametrize('gain',[.01,.1,10.])
def test_global_gain_does_not_change_timbre_or_harmonic_features(gain):
    rng=np.random.default_rng(8);x=tone()+rng.normal(0,.003,SR)
    a,s1=window_features(x);b,s2=window_features(x*gain)
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-4)
    assert np.isclose(s1,s2,rtol=1e-10,atol=1e-10)


def test_one_window_is_bit_exact_when_other_rows_change():
    rng=np.random.default_rng(7);x=np.stack([tone(),rng.normal(size=SR),rng.normal(size=SR)])
    one,s=window_features(x[0]);before,sa=extract_features(x)
    x[1]=0;x[2]*=1000;after,sb=extract_features(x)
    assert np.array_equal(one,before[0]) and np.array_equal(before[0],after[0])
    assert s==sa[0]==sb[0]


def test_true_harmonic_energy_ratios_are_not_ratios_of_log_contrasts():
    f,_,p=stft_power(tone(),LONG_NPERSEG,LONG_HOP)
    _,clean=local_noise_floor(p.mean(1),5.)
    h2,h3=harmonic_energy_ratios(f,clean,400)
    np.testing.assert_allclose([h2,h3],[.25,.0625],rtol=1e-5,atol=1e-7)


def test_multi_resolution_has_actual_100ms_200ms_and_10hz_5hz_bins():
    f1,t1,p1=stft_power(tone(),SHORT_NPERSEG,SHORT_HOP)
    f2,t2,p2=stft_power(tone(),LONG_NPERSEG,LONG_HOP)
    assert SHORT_NPERSEG/SR==.1 and LONG_NPERSEG/SR==.2
    assert np.diff(f1)[0]==10 and np.diff(f2)[0]==5
    assert len(t1)>len(t2) and p1.shape==(801,91) and p2.shape==(1601,41)


@pytest.mark.parametrize('length',[SR-1,SR+1])
def test_no_padding_or_repetition_for_wrong_window_length(length):
    with pytest.raises(ValueError,match='exactly one'):window_features(np.zeros(length))


def calibration(n=25,wrong=3):
    genera=np.array(['Aedes','Anopheles','Culex','Culiseta'])
    y=np.repeat('Aedes',n);p=np.tile([.95,.02,.02,.01],(n,1))
    p[-wrong:]=[.2,.4,.2,.2]
    return y,p,genera


def test_genus_threshold_is_calibration_only_nonzero_and_optimizes_curve():
    y,p,g=calibration();rule=select_genus_threshold(y,p,g,'calibration')
    assert rule['minimum_confidence']==pytest.approx(.30)
    assert rule['threshold']==pytest.approx(.95)
    assert rule['threshold']>=rule['minimum_confidence']>0
    assert rule['objective']==max(r['objective'] for r in rule['curve'] if r['eligible'])
    assert rule['calibration_accuracy']==1 and rule['calibration_emitted']==22
    with pytest.raises(ValueError,match='calibration'):select_genus_threshold(y,p,g,'test')


def test_perfect_calibration_can_select_floor_but_never_zero():
    y,p,g=calibration(wrong=1);p[:]=[.7,.1,.1,.1]
    rule=select_genus_threshold(y,p,g,'calibration')
    assert rule['threshold']==pytest.approx(.30) and rule['qualified']


def test_rejection_requires_twenty_independent_pcm_blocks_not_duplicate_recordings():
    y,p,g=calibration();p[:]=[.9,.04,.04,.02]
    rule=select_genus_threshold(y,p,g,'calibration',pcm_blocks=['same']*len(y))
    assert rule['threshold']>1 and not rule['qualified']


def test_exact_genus_mass_and_strict_union_outside_support():
    _,p=marginalize_genus([[.2,.4,.4]],['Aedes aegypti','Aedes albopictus','Culex pipiens'])
    np.testing.assert_allclose(p,[[.6,.4]],rtol=0,atol=1e-15)
    assert np.isclose(p.sum(),1,atol=1e-15)
    m=classification_metrics(['Aedes','Aedes'],['Aedes','Culex'])
    assert m['macro_f1_strict_union']<m['macro_f1_support_only']


def test_all_preexisting_artifacts_are_untouched():
    assert verify_history()['unchanged']


def test_completed_v3_is_same_560_block_partition_and_fit_only_scaler():
    if not (OUT/'freeze.json').exists():pytest.skip('V3 training pending.')
    verify_freeze();new=pd.read_csv(OUT/'partitions.csv',dtype={'fold':str})
    old=pd.read_csv(V2/'partitions.csv',dtype={'fold':str})
    pd.testing.assert_frame_equal(new.drop(columns='pcm_block'),old)
    frame=pd.read_csv(OUT/'segments_v3.csv');assert frame.pcm_block.nunique()==560
    x=np.load(OUT/'features_v3.npz')['features']
    for fold in ['0','1','2','final']:
        sub=new[new.fold.eq(fold)]
        rows=[sub[sub.partition.eq(p)].row_index.to_numpy(int) for p in ['fit','calibration','test']]
        check_partition(frame,*rows)
        saved=joblib.load(OUT/'models'/f'{fold}_svm.joblib')
        np.testing.assert_allclose(saved['model'].steps[0][1].mean_,x[saved['fit_rows']].mean(0,dtype=float),rtol=1e-10,atol=1e-10)
        t=json.loads((OUT/f'calibration_{fold}.json').read_text())['threshold_genus']
        assert t['threshold']>=t['minimum_confidence']>0
    bundle=joblib.load(OUT/'models/final_bundle.joblib')
    with threadpool_limits(limits=1):
        a=predict_features(x[:1],bundle)[0];before=predict_features(x[:3],bundle)[0]
        changed=x[:3].copy();changed[1:]=1000;after=predict_features(changed,bundle)[0]
    assert np.array_equal(a[0],before[0]) and np.array_equal(before[0],after[0])


def test_completed_v3_freezes_before_oof_and_exact_17_plus_2_external():
    if not (OUT/'evaluation_audit.json').exists():pytest.skip('V3 evaluation pending.')
    freeze=verify_freeze();start=json.loads((OUT/'evaluation_started.json').read_text())
    done=json.loads((OUT/'evaluation_audit.json').read_text())
    assert freeze['all_outer_folds_and_final_frozen'] and freeze['no_test_predictions_yet']
    assert start['started_utc']>freeze['created_utc'] and done['completed_utc']>start['started_utc']
    assert done['outer_predictions_after_freeze'] and done['models_thresholds_unchanged']
    r=pd.read_csv(OUT/'external_recording_predictions.csv')
    old=pd.read_csv(V2/'external_recording_predictions.csv')
    for scope,n in [('external',17),('fresh',2)]:
        assert len(r[r.scope.eq(scope)])==n
        assert set(r[r.scope.eq(scope)].group)==set(old[old.scope.eq(scope)].group)
