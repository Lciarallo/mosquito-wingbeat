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
