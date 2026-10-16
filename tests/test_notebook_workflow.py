"""Published notebook works without ignored caches, weights or old modules."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_executes_without_training_or_local_experiment_files(tmp_path):
    names = ['train_ternary_nested_models.py', 'ternary_protocol.py',
             'train_ternary_lstm.py', 'torres_lstm.py', 'predict_ternary_lstm.py',
             'verify_ternary_lstm.py', 'ternary_reporting.py',
             'evaluate_ternary_event_pipeline.py', 'predict_ternary.py',
             'aedes_classical.py', 'bioacoustic_invariance.py',
             'pattern_classifiers.py', 'train_pattern_classifiers.py',
             'verify_pattern_classifiers.py', 'predict_pattern.py',
             'pattern_reporting.py',
             'notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb',
             'data/metadata/ternary_neural_reference.json',
             'data/metadata/ternary_segments.csv.gz',
             'data/metadata/known_recording_groups.csv',
             'data/metadata/pattern_classifier_reference.json',
             'data/metadata/document_revision.json']
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    command = '''
import json
from IPython.display import display
nb=json.load(open('notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb'))
ns={'__name__':'__main__','display':display}
for cell in nb['cells']:
    if cell['cell_type']=='code': exec(compile(''.join(cell['source']),'notebook','exec'),ns)
assert not ns['RUN_NEW_TRAINING'] and not ns['RUN_NEURAL_TRAINING']
assert not ns['RUN_PATTERN_TRAINING'] and not ns['RUN_PATTERN_VERIFY']
assert ns['PATTERN_REFERENCE']['groups']==1794
assert ns['TRAINED_RUN'] is None and ns['NEURAL_RUN'] is None
assert len(ns['comparison_table'](ns['NEURAL_REFERENCE']['metrics']))==5
from pathlib import Path
assert not Path('results').exists()
'''
    result = subprocess.run([sys.executable, '-c', command], cwd=tmp_path,
                            env={**os.environ, 'MPLBACKEND': 'Agg'},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_historical_reference_metrics_reconcile_with_confusion_counts():
    reference = json.loads((ROOT / 'data/metadata/ternary_neural_reference.json').read_text())
    assert reference['kind'] == 'historical_reference_not_new_training'
    for row in [reference['metrics']['reference'], *reference['metrics']['neural_models']]:
        cm = np.asarray(row['confusion_matrix_raw'])
        assert cm.sum() == reference['protocol_summary']['groups']
        assert row['accuracy'] == pytest.approx(cm.trace() / cm.sum())
        assert row['aedes_recall'] == pytest.approx(cm[2, 2] / cm[2].sum())
    assert not reference['protocol_summary']['pretrained_infection_weights_used']


def test_current_pattern_reference_reconciles_counts_and_all_source_metrics():
    reference=json.loads((ROOT/'data/metadata/pattern_classifier_reference.json').read_text())
    assert sum(v['aedes'] for v in reference['source_counts'])==360
    assert sum(v['total'] for v in reference['source_counts'])==1794
    assert all(v['test_aedes']==72 and v['development_aedes']==288 for v in reference['folds'])
    for model in reference['metrics'].values():
        for rule in model.values():
            for metric in rule.values():
                cm=np.asarray(metric['confusion_matrix'])
                assert cm.sum()==metric['n']
                assert metric['accuracy']==pytest.approx(cm.trace()/cm.sum())
                assert cm[2,2]==metric['aedes_tp']
    assert len(reference['verification']['contexts'])==25
    assert len(reference['source_holdout_verification']['contexts'])==8


def test_missing_neural_training_inputs_fail_before_creating_output(tmp_path):
    import torch
    from train_ternary_lstm import run
    output = tmp_path / 'new_run'
    with pytest.raises(FileNotFoundError, match='completed classical run'):
        run(output, tmp_path / 'missing_reference', torch.device('cpu'))
    assert not output.exists()


@pytest.mark.parametrize('architecture', ['lstm', 'bilstm'])
def test_changed_neural_checkpoint_is_rejected_before_inference(tmp_path, architecture):
    from predict_ternary_lstm import predict
    from torres_lstm import digest, write_json
    directory = tmp_path / 'final' / architecture
    directory.mkdir(parents=True)
    (directory / 'model.pt').write_bytes(b'original weights')
    (directory / 'scaler.npz').write_bytes(b'original scaler')
    write_json(directory / 'freeze.json', {
        'checkpoint_sha256': digest(directory / 'model.pt'),
        'scaler_sha256': digest(directory / 'scaler.npz')})
    write_json(tmp_path / 'complete.json', {'complete': True})
    write_json(tmp_path / 'freeze.json', {
        'selected_final_neural_model': architecture,
        'context_freeze_sha256': {f'final/{architecture}': digest(directory / 'freeze.json')}})
    (directory / 'model.pt').write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='Frozen model/scaler verification failed'):
        predict(tmp_path, tmp_path / 'not_read.wav', name=architecture, passes=0, device='cpu')
