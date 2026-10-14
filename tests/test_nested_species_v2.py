"""Methodological regressions for v2; no embedded compilation or external tuning."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from acoustic_features_v2 import SR,extract_features,window_features
from nested_species_ensemble_v2 import (ROOT,OUT,check_partition,marginalize_genus,
    inductive_probabilities,mix_probabilities,training_weights,pool_recording,
    choose_pooling,select_threshold,verify_history,verify_freeze)
from nested_species_ensemble_v2 import split_collision_calibration
from research_metrics_v2 import classification_metrics,bootstrap_metrics


def _frame():
    return pd.DataFrame(dict(label=['a','b']*3,group=list('abcdef'),collision_group=list('abcdef'),
                             phone=['x']*6,path=list('ABCDEF'),content_sha256=list('123456')))


@pytest.mark.parametrize('column',['group','collision_group','path','content_sha256'])
def test_no_overlap_of_any_source_identity(column):
    f=_frame();f.loc[4,column]=f.loc[0,column]
    with pytest.raises(ValueError,match='leakage'):check_partition(f,[0,1],[2,3],[4,5])


def test_three_way_disjoint_partition_succeeds():
    check_partition(_frame(),[0,1],[2,3],[4,5])


def test_mixed_label_pcm_blocks_cannot_cross_fit_and_calibration():
    f=pd.DataFrame({'group':['shared','shared','a1','a2','b1','b2'],
                    'label':['a','b','a','a','b','b']})
    fit,cal=split_collision_calibration(np.arange(6),f,7)
    assert not (set(f.iloc[fit].group)&set(f.iloc[cal].group))
    assert set(f.iloc[fit].label)==set(f.iloc[cal].label)=={'a','b'}


def test_frontend_is_bit_exact_inductive_under_changed_other_windows():
    t=np.arange(SR)/SR;rng=np.random.default_rng(7)
    waves=np.stack([np.sin(2*np.pi*400*t)+.3*np.sin(2*np.pi*800*t),rng.normal(size=SR),rng.normal(size=SR)])
    one,single=window_features(waves[0]);before,s=extract_features(waves)
    waves[1:]*=100;waves[2]=0
    after,s2=extract_features(waves)
    assert np.array_equal(one,before[0]) and np.array_equal(before[0],after[0])
    assert single==s[0]==s2[0]


@pytest.mark.parametrize('method',['svm','trees','hgb'])
def test_model_prediction_is_bit_exact_for_isolated_window(method):
    rng=np.random.default_rng(5);x=rng.normal(size=(60,8));y=np.arange(60)%3
    model={'svm':make_pipeline(StandardScaler(),SVC()),
           'trees':ExtraTreesClassifier(n_estimators=8,random_state=5,n_jobs=1),
           'hgb':HistGradientBoostingClassifier(max_iter=8,min_samples_leaf=4,random_state=5)}[method]
    with threadpool_limits(1):
        model.fit(x,y);one=inductive_probabilities(model,x[:1]);before=inductive_probabilities(model,x[:4])
        x[1:4]=999;after=inductive_probabilities(model,x[:4])
    assert np.array_equal(one[0],before[0]) and np.array_equal(before[0],after[0])


def test_blend_inductive_bit_exact():
    p=np.array([[[.8,.1,.1],[.1,.7,.2]]]*3)
    settings=dict(weights=[.2,.3,.5],log_prior=[.1,-.1,0])
    one=mix_probabilities(p[:,:1],settings)
    before=mix_probabilities(p,settings);p[:,1,:]=[0,0,1]
    after=mix_probabilities(p,settings)
    assert np.array_equal(one[0],before[0]) and np.array_equal(before[0],after[0])


def test_genus_preserves_all_probability_mass_and_can_beat_species_argmax():
    classes=['Aedes aegypti','Aedes albopictus','Culex pipiens']
    genera,p=marginalize_genus([[.3,.3,.4],[.1,.2,.7]],classes)
    np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-15)
    assert genera[p[0].argmax()]=='Aedes' and classes[2].startswith('Culex')
    np.testing.assert_allclose(p,[[.6,.4],[.3,.7]],rtol=0,atol=1e-15)


def test_strict_f1_penalizes_predictions_outside_true_support():
    m=classification_metrics(['a','a','b','b'],['a','c','b','b'])
    assert m['predicted_taxa_outside_support']==['c']
    assert m['macro_f1_strict_union']<m['macro_f1_support_only']
    assert np.isclose(m['macro_f1_strict_union'],m['macro_f1_support_only']*2/3)


def test_weights_cannot_depend_on_other_test_rows():
    f=_frame();s=np.ones(6);before=training_weights([0,1],f,s)
    f.loc[2:,'phone']='other';f.loc[2:,'label']='other';s[2:]=100
    assert np.array_equal(before,training_weights([0,1],f,s))


def test_mil_selects_salient_windows_and_preserves_invalid_denominator():
    p=np.array([[.5,.5],[.1,.9],[.9,.1],[.5,.5]])
    q,info=pool_recording(p,[.1,3,.1,.1],.25)
    np.testing.assert_allclose(q,p[1]);assert info['selected_windows']==1
    _,info=pool_recording(p,[1]*4,quality=[False]*4)
    assert not info['quality_ok']
    m=classification_metrics(['a','a'],['a','INVALID'],[True,False])
    assert m['accuracy']==.5 and m['coverage']==.5 and m['n']==2


def test_pooling_and_rejection_forbid_test_selection():
    with pytest.raises(ValueError,match='calibration'):choose_pooling(_frame(),np.tile([.5,.5],(6,1)),np.ones(6),['a','b'],'test')
    with pytest.raises(ValueError,match='calibration'):select_threshold(['a'],[[1,0]],['a','b'],'test')


def test_no_fake_zero_for_undefined_accepted_accuracy_or_one_recording_ci():
    assert classification_metrics(['a'],['b'],[False])['accepted_accuracy'] is None
    assert bootstrap_metrics(['a'],{'v1':['b']},{'v1':[False]})['intervals']=={}
    ci=bootstrap_metrics(['a','a'],{'v1':['a','b']},{'v1':[False,False]},repetitions=20)
    assert ci['intervals']['v1']['accepted_accuracy']['valid_replicates']==0


def test_all_historical_results_sha256_are_intact():
    snapshot=OUT/'historical_snapshot.json'
    if not snapshot.exists():snapshot=ROOT/'tmp/v2_2026_10_13/historical_snapshot.json'
    guard=OUT/'offline_scope_guard.json'
    if not guard.exists():guard=ROOT/'tmp/v2_2026_10_13/offline_scope_guard.json'
    assert verify_history(snapshot)['unchanged']
    assert verify_history(guard)['unchanged']


def test_completed_v2_artifacts_and_scalers_respect_fit_only():
    if not (OUT/'freeze.json').exists():pytest.skip('Training not completed yet.')
    verify_freeze()
    parts=pd.read_csv(OUT/'partitions.csv')
    for _,sub in parts.groupby('fold'):
        for col in ['row_index','group','path','content_sha256','collision_group']:
            assert sub.groupby(col).partition.nunique().max()==1
    x=np.load(OUT/'features_v2.npz')['features']
    for fold in ['0','1','2','final']:
        saved=joblib.load(OUT/'models'/f'{fold}_svm.joblib')
        mean=saved['model'].steps[0][1].mean_
        np.testing.assert_allclose(mean,x[saved['fit_rows']].mean(axis=0,dtype=float),rtol=1e-10,atol=1e-10)
    audit=json.loads((OUT/'partition_audit.json').read_text())
    assert all(a['overlap_groups_paths_pcm']==0 for a in audit['audits'])


def test_completed_external_evaluation_follows_freeze_and_preserves_exact_17():
    auditfile=OUT/'external_evaluation_audit.json'
    if not auditfile.exists():pytest.skip('External evaluation not completed yet.')
    freeze=verify_freeze();audit=json.loads(auditfile.read_text())
    fresh=json.loads((OUT/'fresh_selection_v2.json').read_text())
    web=json.loads((OUT/'postfreeze_web_search.json').read_text())
    acquisition=json.loads((OUT/'acquisition_freeze_v2.json').read_text())
    from improve_device import file_hash
    assert acquisition['code_sha256']==file_hash(ROOT/'fetch_species_v2_fresh_checked.py')
    assert acquisition['created_utc']<web['searched_utc']
    assert web['searched_utc']>freeze['created_utc']
    assert fresh['selected_utc']>freeze['created_utc']
    assert audit['evaluated_utc']>fresh['selected_utc']
    old=pd.read_csv(ROOT/'results/species_nested_2026_10_09/external_recording_predictions.csv')
    new=pd.read_csv(OUT/'external_recording_predictions.csv')
    assert set(new[new.scope.eq('external')].group)==set(old[old.model.eq('calibrated')].group)
    assert audit['paired_recordings']==17 and audit['models_thresholds_unchanged']
