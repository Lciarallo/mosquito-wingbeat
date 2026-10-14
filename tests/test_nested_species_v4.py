"""V4 physical, taxonomy, calibration-lineage and immutable-history regressions."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from threadpoolctl import threadpool_limits
from acoustic_features_v4 import (SR,FEATURE_NAMES,PSD_GRID,window_features,extract_features,
    local_noise_floor,centred_cepstrum,MEL_BANK_SHORT)
from nested_species_ensemble_v4 import (ROOT,OUT,V3,check_partition,verify_freeze,verify_history,
    marginalize_genus_v4,select_genus_parameters,select_genus_threshold,pool_recording,
    choose_pooling,fit_blend_v4,mix_probabilities_v4,predict_features)
from research_metrics_v2 import classification_metrics


def catalogue():
    return np.array([f'{g} species{i}' for g,n in [('Aedes',4),('Anopheles',12),('Culex',3),('Culiseta',1)] for i in range(n)])


def tone(f0=400):
    t=np.arange(SR)/SR
    return np.sin(2*np.pi*f0*t)+.5*np.sin(2*np.pi*2*f0*t)+.25*np.sin(2*np.pi*3*f0*t)


@pytest.mark.parametrize('f0',[400,625])
def test_stationary_white_floor_25_to_12db_preserves_spectral_shape_and_fundamental(f0):
    rng=np.random.default_rng(101);signal=tone(f0);noise=rng.normal(size=SR)
    noise/=np.sqrt(np.mean(noise**2));rms=np.sqrt(np.mean(signal**2))
    a,sa=window_features(signal+noise*rms*10**(-25/20))
    b,sb=window_features(signal+noise*rms*10**(-12/20))
    for ids in [np.arange(len(PSD_GRID)),np.array([FEATURE_NAMES.index(f'mfcc_centred_mean_{i}') for i in range(1,13)])]:
        cosine=np.dot(a[ids],b[ids])/(np.linalg.norm(a[ids])*np.linalg.norm(b[ids]))
        # This tests shape robustness, not mathematically false absolute SNR invariance.
        assert cosine>.97
    assert np.sqrt(np.mean((a[:len(PSD_GRID)]-b[:len(PSD_GRID)])**2))<3
    assert abs(a[FEATURE_NAMES.index('shs_long_peak0_f0')]-b[FEATURE_NAMES.index('shs_long_peak0_f0')])<=5
    assert abs(sa-sb)<.1


def test_log_floor_ratio_does_not_turn_half_of_bins_into_minus_120db():
    power=np.array([.5,1.,2.,1.,.5,1.,2.])*1e-5
    floor,snr=local_noise_floor(power,5)
    np.testing.assert_allclose(snr,10*np.log10(power/floor))
    assert np.isfinite(snr).all() and snr.min()>-10 and snr.max()<10
    assert np.any(snr<0)


def test_cepstrum_uses_relative_mel_power_and_retains_low_order_timbre():
    rng=np.random.default_rng(8);p=np.exp(rng.normal(size=(801,91)))
    floor,_=local_noise_floor(p,10)
    a=centred_cepstrum(p,floor);b=centred_cepstrum(p*100,floor*100)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-11,atol=1e-11)
    assert a[0][0]==0 and np.any(np.abs(a[0][1:3])>0)


def test_frontend_is_bit_exact_inductive_and_finite_on_silence():
    waves=np.stack([tone(),np.zeros(SR),np.random.default_rng(5).normal(size=SR)])
    one,s=window_features(waves[0]);before,sa=extract_features(waves)
    waves[1:]=999;after,sb=extract_features(waves)
    assert np.array_equal(one,before[0]) and np.array_equal(before[0],after[0])
    assert s==sa[0]==sb[0] and np.isfinite(before).all()


@pytest.mark.parametrize('gain',[.01,10.])
def test_gain_invariance(gain):
    w=tone()+np.random.default_rng(11).normal(0,.01,SR)
    a,_=window_features(w);b,_=window_features(w*gain)
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-4)


@pytest.mark.parametrize('column',['group','collision_group','pcm_block','path','content_sha256'])
def test_no_shared_source_identity_between_partitions(column):
    f=pd.DataFrame(dict(label=['a','b']*3,group=list('abcdef'),collision_group=list('abcdef'),
        pcm_block=list('abcdef'),path=list('ABCDEF'),content_sha256=list('123456')))
    check_partition(f,[0,1],[2,3],[4,5]);f.loc[4,column]=f.loc[0,column]
    with pytest.raises(ValueError,match='leakage'):check_partition(f,[0,1],[2,3],[4,5])


def test_power_taxonomy_formula_and_exact_sum_special_case():
    classes=catalogue();p=np.random.default_rng(3).dirichlet(np.ones(20),size=3)
    g,q=marginalize_genus_v4(p,classes,.75,2)
    score=np.column_stack([np.sqrt((p[:,np.char.startswith(classes,x)]**2).sum(1)/sum(np.char.startswith(classes,x))**.75) for x in g])
    np.testing.assert_allclose(q,score/score.sum(1,keepdims=True),rtol=1e-12,atol=1e-15)
    _,exact=marginalize_genus_v4(p,classes,0,1)
    expected=np.column_stack([p[:,np.char.startswith(classes,x)].sum(1) for x in g])
    np.testing.assert_allclose(exact,expected,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(q.sum(1),1,rtol=0,atol=1e-15)
    alone=marginalize_genus_v4(p[:1],classes,.75,2)[1]
    p[1:]=np.full(20,.05)
    assert np.array_equal(alone[0],marginalize_genus_v4(p,classes,.75,2)[1][0])


def test_uniform_species_is_strictly_below_genus_rejection_floor():
    c=catalogue();p=np.tile(np.eye(20)*.95+.05/20,(2,1));labels=np.tile(c,2)
    settings=select_genus_parameters(labels,p,c,'calibration')
    _,null=marginalize_genus_v4(np.full((1,20),1/20),c,settings['gamma'],settings['tau'])
    assert null.max()<settings['minimum_confidence']==pytest.approx(.30)
    assert settings['selection_partition']=='calibration'
    assert all(r['uniform_null_max']<.30 for r in settings['candidates'] if r['eligible'])
    with pytest.raises(ValueError,match='calibration'):select_genus_parameters(labels,p,c,'test')


def test_cardinality_correction_rescues_aedes_plurality_from_anopheles_mass():
    c=catalogue();p=np.zeros((1,20));p[0,:4]=[.14,.10,.08,.06];p[0,4:16]=.60/12;p[0,16:19]=.01/3;p[0,19]=.01
    g,old=marginalize_genus_v4(p,c,0,1);_,new=marginalize_genus_v4(p,c,1,2)
    assert g[old.argmax()]=='Anopheles' and g[new.argmax()]=='Aedes'


@pytest.mark.parametrize('n',[5,29,104])
def test_long_mil_is_selective_even_at_maximum_fraction(n):
    p=np.tile([.1,.9],(n,1));q,info=pool_recording(p,np.ones(n),.65,presence_prob=np.ones(n))
    assert info['selected_windows']<n
    np.testing.assert_allclose(q,[.1,.9])
    with pytest.raises(ValueError,match='selective'):pool_recording(p,np.ones(n),1.)


def test_short_mil_and_all_uniform_uncertainty():
    _,info=pool_recording(np.tile([.1,.9],(4,1)),np.ones(4),.2)
    assert info['selected_windows']==4
    q,info=pool_recording(np.full((30,20),.05),np.ones(30),.5)
    assert not info['quality_ok']
    np.testing.assert_allclose(q,np.full(20,.05))


def test_presence_weight_and_uniform_background_do_not_dilute_flight():
    p=np.full((100,20),.05);p[50]=.01;p[50,0]=.81
    q,info=pool_recording(p,np.ones(100),.35,presence_prob=np.r_[np.zeros(50),1,np.zeros(49)])
    np.testing.assert_allclose(q,p[50],atol=1e-12)
    assert info['weighted_windows']==1
    p2=np.array([[.9,.1],[.1,.9]]);q,_=pool_recording(p2,[1,1],.35,presence_prob=[1,0])
    expected=(p2[0]+.15*p2[1])/1.15
    np.testing.assert_allclose(q,expected,atol=1e-12)


def test_calibration_only_pooling_and_no_full_fraction_candidate():
    f=pd.DataFrame({'group':['a']*10+['b']*10,'label':['A']*10+['B']*10})
    p=np.r_[np.tile([.9,.1],(10,1)),np.tile([.1,.9],(10,1))]
    rule=choose_pooling(f,p,np.ones(20),['A','B'],'calibration',presence_prob=np.ones(20))
    assert {r['top_fraction'] for r in rule['candidates']}=={.2,.35,.5,.65}
    with pytest.raises(ValueError,match='calibration'):choose_pooling(f,p,np.ones(20),['A','B'],'test')


def test_species_prior_is_bounded_and_calibration_only():
    c=np.array(['Aedes aegypti','Aedes sierrensis','Culex pipiens']);y=np.repeat(np.arange(3),[7,30,12]);n=len(y)
    p=np.tile(np.eye(3)[y]*.9+.1/3,(3,1,1));f=pd.DataFrame(dict(group=[str(i) for i in range(n)],collision_group=[str(i) for i in range(n)],label=c[y]))
    rule=fit_blend_v4(p,y,np.ones(n),f,np.ones(n),np.ones(n),c,partition='calibration')
    assert np.max(np.abs(rule['log_prior']))<=.8
    assert rule['base_balanced_log_prior'][0]>rule['base_balanced_log_prior'][1]
    assert rule['calibration_pcm_blocks_per_species']==[7,30,12]
    a=mix_probabilities_v4(p[:,:1],rule);b=mix_probabilities_v4(p,rule)
    p[:,1:]=np.full(3,1/3);changed=mix_probabilities_v4(p,rule)
    assert np.array_equal(a[0],b[0]) and np.array_equal(b[0],changed[0])
    with pytest.raises(ValueError,match='calibration'):fit_blend_v4(p,y,np.ones(n),f,np.ones(n),np.ones(n),c,partition='test')


def test_strict_f1_and_all_historical_files_are_intact():
    m=classification_metrics(['Aedes','Aedes'],['Aedes','Culex'])
    assert m['macro_f1_strict_union']<m['macro_f1_support_only']
    assert verify_history()['unchanged']


def test_completed_v4_models_partitions_null_and_inductive_inference():
    # Before training check the physical 560-block reference; after training validate every model.
    reference=pd.read_csv(V3/'partitions.csv',dtype={'fold':str})
    frame=pd.read_csv(V3/'segments_v3.csv');assert frame.pcm_block.nunique()==560
    for fold in ['0','1','2','final']:
        sub=reference[reference.fold.eq(fold)]
        check_partition(frame,*[sub[sub.partition.eq(k)].row_index.to_numpy(int) for k in ['fit','calibration','test']])
    if not (OUT/'freeze.json').exists():return
    freeze=verify_freeze();assert freeze['all_outer_folds_and_final_frozen'] and freeze['no_test_predictions_yet']
    pd.testing.assert_frame_equal(pd.read_csv(OUT/'partitions.csv',dtype={'fold':str}),reference)
    x=np.load(OUT/'features_v4.npz')['features']
    for fold in ['0','1','2','final']:
        bundle=joblib.load(OUT/'models'/f'{fold}_bundle.joblib');settings=bundle['settings'];g=settings['genus_parameters']
        _,null=marginalize_genus_v4(np.full((1,20),.05),bundle['classes'],g['gamma'],g['tau'])
        assert null.max()<settings['threshold_genus']['minimum_confidence']<=settings['threshold_genus']['threshold']
        assert settings['pooling']['top_fraction']<=.65 and np.max(np.abs(settings['parameters']['log_prior']))<=.8
        model=bundle['models'][0]
        rows=bundle['fit_rows'];rows=rows[frame.iloc[rows].label.ne('noise')]
        np.testing.assert_allclose(model.steps[0][1].mean_,x[rows].mean(0,dtype=float),rtol=1e-10,atol=1e-10)
    with threadpool_limits(1):
        a=predict_features(x[:1],bundle)[0];b=predict_features(x[:3],bundle)[0]
        changed=x[:3].copy();changed[1:]=1000;c=predict_features(changed,bundle)[0]
    assert np.array_equal(a[0],b[0]) and np.array_equal(b[0],c[0])


def test_completed_v4_freezes_before_evaluation_and_acquires_exactly_two_new_files():
    if not (OUT/'evaluation_audit.json').exists():return
    freeze=verify_freeze();start=json.loads((OUT/'evaluation_started.json').read_text());done=json.loads((OUT/'evaluation_audit.json').read_text())
    assert freeze['created_utc']<start['started_utc']<done['completed_utc']
    new=json.loads((OUT/'fresh_selection_v4.json').read_text())
    assert new['selected_utc']>freeze['created_utc'] and len(new['accepted_files'])==2
    files=pd.read_csv(OUT/'external_recording_predictions.csv')
    for scope,n in [('external',17),('fresh',2),('postfreeze_new',2),('postfreeze_new_v4',2),('drive',1)]:assert sum(files.scope.eq(scope))==n
    assert (files.query('complete_windows>4').selected_windows<files.query('complete_windows>4').complete_windows).all()
    assert not pd.read_csv(OUT/'fresh_novelty_audit.csv').excluded.any()
