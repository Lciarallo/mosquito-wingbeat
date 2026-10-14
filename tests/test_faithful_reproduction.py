"""Scientific provenance and saved-prediction checks for actual author reruns."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/faithful_reproduction_2026_10_03'


def load(p): return json.loads(p.read_text())


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize('study',['dr_biol','mosquitosong'])
def test_author_weights_and_sources_match_execution(study):
    e=load(OUT/study/'execution.json')
    assert e['status']=='complete' and not e['training_rerun']
    if study=='dr_biol':
        source=ROOT/'tmp/reproductions_2026_10_01/dr_biol'
        models=source/'application/Pretrained_models'
        assert sha(source/'application/Evaluation.py')==e['original_evaluation_sha256']
        for p,h in e['original_framework_sha256'].items(): assert sha(source/p)==h
    else:
        source=ROOT/'tmp/reproductions_2026_10_01/mosquitosongp'
        models=ROOT/'data/external/faithful_2026_10_03/mosquitosong_models'
        assert e['source_code_cells_unchanged']
    for name,h in e['models'].items(): assert sha(models/name)==h
    assert subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip()==''


def test_all_original_notebook_cells_really_executed_unchanged():
    original=load(ROOT/'tmp/reproductions_2026_10_01/mosquitosongp/prediction.ipynb')
    executed=load(OUT/'mosquitosong/author_executed.ipynb')
    assert len(original['cells'])==len(executed['cells'])
    for a,b in zip(original['cells'],executed['cells']):
        assert ''.join(a['source'])==''.join(b['source'])
        if a['cell_type']=='code':
            assert isinstance(b['execution_count'],int)
            assert not any(o['output_type']=='error' for o in b['outputs'])
    e=load(OUT/'mosquitosong/execution.json')
    assert e['tensorflow']=='2.4.0' and e['input_wav_count']==1324 and e['noise_wav_count']==12


def test_official_test_scores_match_saved_predictions():
    rows=load(OUT/'dr_biol/metrics.json')
    assert len(rows)==10
    for r in rows:
        with np.load(OUT/'dr_biol'/(r['model']+'_seed'+str(r['seed'])+'.npz')) as a:
            assert len(a['y'])==10256
            assert np.array_equal(a['p'],a['probabilities'].argmax(1))
            assert np.isclose(accuracy_score(a['y'],a['p']),r['accuracy'])
            assert np.isclose(roc_auc_score(np.eye(8)[a['y']].ravel(),a['probabilities'].ravel()),r['roc_auc_micro'])
            assert np.isclose(average_precision_score(np.eye(8)[a['y']],a['probabilities'],average='micro'),r['average_precision_micro'])
            cm=confusion_matrix(a['y']==2,a['p']==2,labels=[False,True])
            tn,fp,fn,tp=cm.ravel()
            assert np.isclose(tp/(tp+fn),r['binary']['recall'])
            assert np.isclose(tn/(tn+fp),r['binary']['specificity'])
    assert set(r['seed'] for r in rows)=={0,1,2,3,4}


def test_mosquitosong_folds_and_pooled_metrics_are_not_training_claims():
    rows=load(OUT/'mosquitosong/metrics.json')
    assert len(rows)==40
    assert len({(r['fold'],r['gain_index']) for r in rows})==40
    y,p=[],[]
    for r in rows:
        assert r['model_filename']==f"cnn_model_min_val_round{r['fold']+1}.h5"
        assert r['labels']['Ae.Aegypti_F']==0 and r['labels']['Ae.Aegypti_M']==1
        with np.load(OUT/'mosquitosong'/f"fold{r['fold']+1}_gain{r['gain_index']}.npz") as a:
            assert np.isclose(accuracy_score(a['y_true'],a['y_pred']),r['accuracy'])
            if r['gain_index']==3:
                y.extend(a['y_true']);p.extend(a['y_pred'])
    table=pd.read_csv(OUT/'paper_comparison.csv')
    row=table[table.study.eq('MosquitoSong+')].iloc[0]
    assert len(y)==36000 and np.isclose(accuracy_score(y,p),row.accuracy)
    assert not row.training_rerun and row.unique_base_epochs_before_6_simulations==6000


def test_overlap_findings_are_reported_and_not_confused_with_independence():
    audit=load(OUT/'mosquitosong/split_audit.json')
    assert audit['source_functions_unchanged'] and len(audit['folds'])==10
    assert all(r['shared_recordings']['train_test']>0 for r in audit['folds'])
    assert all(r['cross_split_overlapping_window_pairs']['train_test']>0 for r in audit['folds'])
    dr=load(OUT/'dr_biol/data_audit.json')
    assert not dr['recording_ids_available']
    assert sum(dr['identical_feature_overlap'].values())==0


def test_external_denominators_and_quality_are_exactly_the_previous_intervals():
    meta=load(OUT/'common_test_metadata.json')
    assert len(meta)==33
    assert sum(r['windows'] for r in meta if r['scope']=='external32_retrospective')==641
    private=next(r for r in meta if r['scope']=='drive_positive_only')
    assert private['windows']==107 and sum(private['quality_ok'])==105
    old=pd.read_csv(ROOT/'results/cross_domain_strategies_2026_10_03/fresh_predictions.csv.gz')
    old=old[old.model.eq('ridge_contrast_swarm')]
    with np.load(OUT/'mosquitosong/external_derived_ensemble.npz') as a:
        assert len(a['y'])==748 and a['probabilities'].shape==(748,8)
        for r in meta:
            mask=a['group']==r['group']
            assert mask.sum()==r['windows']
            assert np.array_equal(a['quality'][mask],r['quality_ok'])
            assert np.all(a['y'][mask]==int(r['reference_aedes']))
            if r['scope']=='external32_retrospective':
                b=old[old.group.eq(r['group'])]
                assert len(b)==r['windows']
                assert np.array_equal(b.quality_ok.to_numpy(),a['quality'][mask])


def test_device_files_and_prior_models_preserved():
    before=ROOT/'results/cross_domain_strategies_2026_10_03'
    for name in ['device_files_sha256.json','original_weights_sha256.json']:
        for path,h in load(before/name).items(): assert sha(ROOT/path)==h
    v=load(OUT/'artifact_verification.json')
    assert not v['arduino_compiled'] and not v['training_rerun'] and not v['private_raw_audio_exported']
    assert v['device_files_and_previous_models_sha256_match']
    for path,h in v['artifact_sha256'].items():assert sha(ROOT/path)==h
