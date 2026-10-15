"""Recalculate the Git update's saved species results without training models.

This retrospective audit does not repair nested-validation leakage or establish
independence of individuals/sessions. Historical artifacts are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from leakage_auditor import audit_splits
from train_hierarchical_models import (
    load_dataset_and_predictions, evaluate_approach_at_all_tiers,
)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Choose another output directory; audit already exists.')
    inputs = [
        'results/segments.csv', 'results/hierarchical_metrics.csv',
        'results/improvements/species_protocol.json',
        'results/improvements/species_oof.npz',
        'results/improvements/calibrated_ensemble_oof.npz',
        'results/mixed_dataset_experiment.json', 'results/mixed_dataset_metrics.csv',
        'optimize_ensemble_oof.py', 'improve_device.py',
        'train_hierarchical_models.py', 'leakage_auditor.py',
        'experiment_mixed_datasets.py', 'predict_audio.py',
        'robust_audio_features.py', 'audit_git_species.py',
        'drive_audio.wav',
        *[f'results/arduino_species/models/species_fold{k}.npz' for k in range(3)],
    ]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}
    segments, frame, classes, genera, unique_genera, models = load_dataset_and_predictions()
    protocol = json.loads((ROOT / inputs[2]).read_text())
    if list(classes) != protocol['classes']:
        raise ValueError('Class order differs from base-model protocol.')
    if list(classes) != sorted(frame.label.unique()):
        raise ValueError('Class labels differ from segment labels.')
    if frame.groupby('group').label.nunique().max() != 1:
        raise ValueError('A recording has conflicting species labels.')
    historical = pd.read_csv(ROOT / inputs[1]).set_index('model')
    metrics, per_species = [], []
    for name, probs in models.items():
        if probs.shape != (len(frame), len(classes)) or not np.isfinite(probs).all():
            raise ValueError(f'Invalid probabilities for {name}.')
        if (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError(f'Unnormalised probabilities for {name}.')
        row = evaluate_approach_at_all_tiers(name, probs, frame, classes, genera, unique_genera)
        for key, value in row.items():
            if key != 'model' and not np.isclose(value, historical.loc[name, key], atol=1e-12, rtol=1e-10):
                raise ValueError(f'Saved metric mismatch: {name}, {key}.')
        metrics.append(row)
        rec = pd.DataFrame(probs, index=frame.group, columns=classes).groupby(level=0).mean()
        true_rec = frame.groupby('group').label.first().reindex(rec.index)
        for unit, truth, predicted in [
            ('window', frame.label.to_numpy(), classes[probs.argmax(1)]),
            ('recording', true_rec.to_numpy(), rec.idxmax(axis=1).to_numpy()),
        ]:
            cm = confusion_matrix(truth, predicted, labels=classes)
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                out / f'confusion_{list(models).index(name)}_{unit}.csv')
            for i, species in enumerate(classes):
                n, emitted, correct = int(cm[i].sum()), int(cm[:, i].sum()), int(cm[i, i])
                per_species.append(dict(
                    model=name, unit=unit, species=species, support=n,
                    recording_groups=int(frame.loc[frame.label.eq(species), 'group'].nunique()),
                    correct=correct, predicted=emitted, recall=correct/n if n else None,
                    precision=correct/emitted if emitted else None,
                    f1=2*correct/(n+emitted) if n+emitted else None,
                    scope='Forced closed-set classification on previously examined Dryad recordings',
                ))
    pd.DataFrame(metrics).to_csv(out / 'hierarchical_recalculated.csv', index=False)
    pd.DataFrame(per_species).to_csv(out / 'per_species.csv', index=False)
    split_check = audit_splits(segments)
    manifest = dict(
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_sha256=hashes, saved_metrics_recalculated=True,
        split_disjointness=split_check, base_class_order_matches_protocol=True,
        row_identity_scope='Arrays lack explicit row IDs; segment order assumed as in historical scripts.',
        calibrated_ensemble_independent_nested_oof=False,
        reason='Calibration for outer fold k uses OOF rows from j != k whose base estimators trained on fold k.',
        pipeline_weights='predict_audio uses historical Arduino MLP weights, not the calibrated 7-model ensemble.',
        validation_scope='Retrospective arithmetic and inference audit; no retraining, new acquisition or firmware compilation.',
        can_claim_reliable_species_classification=False,
    )
    # Check that reads did not mutate any historical artifact.
    for p, digest in hashes.items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Input changed during audit: {p}.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(pd.DataFrame(metrics)[['model', 'species_window_accuracy', 'species_recording_accuracy',
                                 'species_recording_balanced_acc']].to_string(index=False))
    print('Historical metrics match; split check:', split_check['status'])
    print('Species reliability remains unproven; see manifest limitations.')


if __name__ == '__main__':
    main()
