"""Protect data isolation and real inference for non-recurrent alternatives."""
import json
import numpy as np
import pandas as pd
import pytest
import torch

from pattern_classifiers import (SpectrogramCNN, AcousticMLP, source_class_weights,
                                 source_class_score, assert_partitions)
import train_pattern_classifiers as study


def tiny_frame():
    rows=[]
    for fold in range(5):
        for label in range(3):
            for window in range(2):
                group=f'f{fold}_c{label}'
                rows.append(dict(row_id=len(rows),group=group,pcm_block=group,fold=fold,
                    true_label=label,window_target=label,domain='original',path=group))
    return pd.DataFrame(rows)


def test_source_balancing_does_not_let_torres_multiply_positive_mass():
    f=pd.DataFrame(dict(group=['a','a','b','c'],domain=['dryad','dryad','torres','dryad'],window_target=[2,2,2,1]))
    w=source_class_weights(f)
    np.testing.assert_allclose([w[:2].sum(),w[2],w[3]],np.repeat(w.sum()/3,3))
    larger=pd.concat([f,*[f.iloc[[2]].assign(group=f'torres_{i}')for i in range(10)]],ignore_index=True)
    w=source_class_weights(larger)
    sums=pd.DataFrame(dict(source=larger.domain,label=larger.window_target,w=w)).groupby(['source','label']).w.sum()
    np.testing.assert_allclose(sums,sums.iloc[0])


def test_group_leakage_is_rejected_even_when_row_ids_differ():
    f=tiny_frame();fit,val,test,_=study.partitions(f,0)
    f.loc[test[0],'pcm_block']=f.pcm_block.iloc[fit[0]]
    with pytest.raises(ValueError,match='Leakage'):assert_partitions(f,fit,val,test)


def test_source_score_does_not_hide_failed_original_aedes():
    f=pd.DataFrame(dict(source=['dryad','torres','torres'],true_label=[2,2,2]))
    assert source_class_score(f,[1,2,2])==.5


def test_cnn_classifies_real_variable_duration_without_padding():
    torch.manual_seed(42);m=SpectrogramCNN().eval()
    for duration in [15,55,179]:
        p=m(torch.randn(2,duration,16)).softmax(1)
        assert p.shape==(2,3)
        torch.testing.assert_close(p.sum(1),torch.ones(2))
    with pytest.raises(ValueError):m(torch.zeros(2,5,16))


@pytest.mark.parametrize('name',['mlp','cnn'])
def test_selection_refit_and_frozen_test_inference(tmp_path,monkeypatch,name):
    torch.set_num_threads(2)
    monkeypatch.setitem(study.SETTINGS,'max_epochs',1)
    monkeypatch.setitem(study.SETTINGS,'batch_size',16)
    f=tiny_frame();rng=np.random.default_rng(5)
    raw=rng.normal(size=(len(f),15,16)).astype('float32')
    stats=rng.normal(size=(len(f),80)).astype('float32')
    study.neural_context(name,0,f,raw,stats,tmp_path,torch.device('cpu'))
    d=tmp_path/'outer_0'/name
    saved=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
    model=study.nn_factory(name);model.load_state_dict(saved['state_dict']);model.eval()
    test=np.flatnonzero(f.fold.eq(0));z=np.load(d/'scaler.npz');features=raw if name=='cnn' else stats
    p=study.predict_probabilities(model,((features[test]-z['mean'])/z['std']).astype('float32'))
    np.testing.assert_allclose(p,np.load(d/'window_test_probabilities.npy'),atol=1e-6)
    assert json.loads((d/'freeze.json').read_text())['before_test_inference']
    assert json.loads((d/'selection.json').read_text())['best_epoch']==1
    assert len(pd.read_csv(d/'group_predictions.csv'))==3


def test_document_revision_receipt_cannot_authorize_changed_weights(tmp_path):
    from verify_pattern_classifiers import verify_protected_files
    from torres_lstm import digest
    notebook=tmp_path/'notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb'
    notebook.parent.mkdir();notebook.write_text('before')
    weights=tmp_path/'weights.pt';weights.write_bytes(b'frozen')
    expected={str(notebook.relative_to(tmp_path)):digest(notebook),'weights.pt':digest(weights)}
    notebook.write_text('updated explanation')
    receipt=dict(kind='authorized_document_revision',files={str(notebook.relative_to(tmp_path)):dict(original_sha256=expected[str(notebook.relative_to(tmp_path))],current_sha256=digest(notebook))})
    assert verify_protected_files(tmp_path,expected,receipt)==[str(notebook.relative_to(tmp_path))]
    with pytest.raises(ValueError,match='explicit matching receipt'):verify_protected_files(tmp_path,expected)
    weights.write_bytes(b'changed')
    receipt['files']['weights.pt']=dict(original_sha256=expected['weights.pt'],current_sha256=digest(weights))
    with pytest.raises(ValueError,match='scientific artifact'):verify_protected_files(tmp_path,expected,receipt)


def test_full_miniature_pattern_pipeline_finalize_verify_and_predict(tmp_path,monkeypatch):
    import soundfile as sf
    import predict_pattern
    import verify_pattern_classifiers
    from torres_lstm import digest
    torch.set_num_threads(2)
    monkeypatch.setitem(study.SETTINGS,'max_epochs',1)
    monkeypatch.setitem(study.SETTINGS,'batch_size',16)
    monkeypatch.setitem(study.GRIDS,'svm_rbf',[dict(C=1.,gamma='scale')])
    monkeypatch.setitem(study.GRIDS,'extra_trees',[dict(min_samples_leaf=2)])
    monkeypatch.setitem(study.GRIDS,'histgb',[dict(l2_regularization=1.)])
    f=tiny_frame();rng=np.random.default_rng(11)
    raw=rng.normal(size=(len(f),15,16)).astype('float32')
    stats=rng.normal(size=(len(f),80)).astype('float32')
    ref=tmp_path/'ref';ref.mkdir()
    out=tmp_path/'out';out.mkdir()
    f.to_csv(ref/'segments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    f.to_csv(out/'segments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    np.save(ref/'classical_features.npy',stats)
    seq_path=tmp_path/'sequences.npy';np.save(seq_path,raw)
    study.write(ref/'feature_cache.json',dict(sha256=digest(seq_path),identity=dict(shape=list(raw.shape))))
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in ['pattern_classifiers.py','train_pattern_classifiers.py']:
        (snapshot/p).write_bytes((study.ROOT/p).read_bytes())
    study.write(out/'protocol.json',dict(
        segments_sha256=digest(ref/'segments.csv.gz'),
        features_sha256=digest(ref/'classical_features.npy'),
        code_sha256={p:digest(snapshot/p) for p in ['pattern_classifiers.py','train_pattern_classifiers.py']},
        protected_sha256={}))
    for fold in range(5):
        for name in study.NAMES:
            if name in study.NAMES[:3]:
                study.ml_context(name,fold,f,stats,out)
            else:
                study.neural_context(name,fold,f,raw,stats,out,torch.device('cpu'))
    study.summarize(out)
    assert (out/'complete.json').is_file()
    manifest=study.finalize_model(out,f,raw,stats,torch.device('cpu'))
    assert (out/'final'/'manifest.json').is_file()
    assert isinstance(manifest['calibrated_rule'],dict)
    verified=verify_pattern_classifiers.verify_run(out,ref,seq_path,device=torch.device('cpu'))
    assert verified['complete'] and verified['group_coverage_per_model']==15
    wav_path=tmp_path/'sample.wav'
    wave=rng.normal(scale=0.05,size=47616+8000).astype('float32')
    sf.write(wav_path,wave,48000)
    seqs=predict_pattern.audio_sequences(wav_path)
    assert len(seqs)==1
    pred_argmax=predict_pattern.predict(out/'final',wav_path,rule='argmax')
    pred_cal=predict_pattern.predict(out/'final',wav_path,rule='calibrated')
    assert pred_argmax['windows']==1 and pred_argmax['rule']=='argmax'
    assert pred_cal['windows']==1 and pred_cal['rule']=='calibrated'
    bad_manifest=dict(manifest,model_sha256='0'*64)
    study.write(out/'final'/'manifest.json',bad_manifest)
    with pytest.raises(ValueError,match='SHA-256'):
        predict_pattern.predict(out/'final',wav_path)
    with pytest.raises(ValueError,match='Final model SHA-256'):
        verify_pattern_classifiers.verify_run(out,ref,seq_path,device=torch.device('cpu'))
