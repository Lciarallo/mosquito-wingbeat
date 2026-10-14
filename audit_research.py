"""Recalculate LOSO evidence from saved predictions; never train or tune.

Run: .venv/bin/python audit_research.py
Outputs are retrospective, with inputs hashed and old artefacts preserved.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from loso_generalization import load, SEGMENTS, FEATURES
from research_metrics import binary_metrics, cluster_intervals

ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT / 'results/loso_2026_10_03'
CORRECTED = ROOT / 'results/loso_corrected_2026_10_07'
OUT = ROOT / 'results/research_audit_2026_10_07'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Audit already exists; choose a new output directory.')
    frame, _, labels = load()
    rows, intervals = [], {}
    for evaluation, folder in [('historical', HISTORICAL), ('corrected_acoustic_filter', CORRECTED)]:
      old = pd.read_csv(folder / 'loso_metrics.csv')
      with np.load(folder / 'loso_predictions.npz', allow_pickle=False) as saved:
        expected = {f'{r.model}|{r.held_out}' for r in old.itertuples()}
        if set(saved.files) != expected:
            raise ValueError('Saved candidate/source pairs differ from the historical table.')
        for row in old.itertuples():
            key = f'{row.model}|{row.held_out}'
            mask = frame.domain.eq(row.held_out).to_numpy()
            scores, y = saved[key], labels[mask]
            metrics = binary_metrics(y, scores, np.ones(len(y), bool), .5)
            comparisons = dict(roc_auc=row.auc, aedes_recall=row.recall,
                               specificity=row.specificity, balanced_accuracy=row.balanced_accuracy_at_half)
            if not all(np.isclose(metrics[k], v, atol=1e-12) for k, v in comparisons.items()):
                raise ValueError('Historical predictions no longer align with segments: ' + key)
            ci = cluster_intervals(y, scores, np.ones(len(y), bool), frame.loc[mask, 'group'].to_numpy())
            intervals[evaluation + '|' + key] = ci
            rows.append(dict(evaluation=evaluation, model=row.model, held_out=row.held_out,
                             protocol='transductive' if 'source_norm' in row.model else 'inductive',
                             historical_acoustic_filter_misclassified_dryad=evaluation == 'historical' and row.model == 'hgb_acoustic_only',
                             groups=ci['groups'], **metrics))
    counts = frame.groupby(['domain', 'label']).agg(windows=('group', 'size'), groups=('group', 'nunique'))
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(),stage='Retrospective recalculation of saved predictions, not a new blind test.',
                 threshold=.5, threshold_tuned=False, training_rerun=False,
                 quality_scope='Historical LOSO retained only quality-valid known-labelled windows; these estimates do not measure end-to-end quality rejection.',
                 source_normalisation='Transductive; uses unlabelled statistics from all evaluated windows.',
                 historical_scores_checked=True, can_release_to_user_device=False,
                 inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in [SEGMENTS, FEATURES] + [folder/name for folder in [HISTORICAL,CORRECTED] for name in ['loso_metrics.csv','loso_predictions.npz','protocol.json']]},
                 code_sha256={p: sha(ROOT/p) for p in ['audit_research.py', 'research_metrics.py', 'loso_generalization.py']})
    OUT.mkdir()
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv', index=False)
    counts.to_csv(OUT/'source_counts.csv')
    (OUT/'intervals.json').write_text(json.dumps(intervals, indent=2, allow_nan=False))
    (OUT/'manifest.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(pd.DataFrame(rows)[['model', 'held_out', 'aedes_precision', 'aedes_recall', 'macro_f1', 'specificity']].to_string(index=False))


if __name__ == '__main__':
    main()
