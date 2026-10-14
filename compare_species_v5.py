"""Post-freeze v1/v2/v3/v4/v5 paired audit. No fitting, selection or retuning."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from improve_device import ROOT, file_hash, atomic_json
from nested_species_ensemble import now
from research_metrics_v2 import classification_metrics, bootstrap_metrics


def historical_predictions_on_v5_new(new, out):
    from compare_species_v4 import historical_predictions_on_v4_new
    from nested_species_ensemble_v4 import audio_probabilities, prediction_records
    records = historical_predictions_on_v4_new(new, out).to_dict('records')
    b = joblib.load(ROOT / 'results/species_nested_v4_2026_10_14/models/final_bundle.joblib')
    for item in new.itertuples():
        p, pp, s, q, _ = audio_probabilities(ROOT / item.path, b)
        f = pd.DataFrame({'group': [item.group] * len(p), 'label': [item.label] * len(p)})
        r, _, _ = prediction_records(f, p, s, np.array(b['classes']), b['settings'], pp, q, 'postfreeze_new_v5')
        r = r.iloc[0]
        records.append(dict(
            group=item.group, version='v4',
            species=r.predicted_species if r.quality_ok else 'INVALID',
            genus=r.genus_marginalized if r.quality_ok else 'INVALID',
            posthoc=r.genus_posthoc if r.quality_ok else 'INVALID',
            accepted_species=bool(r.accepted_species),
            accepted_genus=bool(r.accepted_genus),
        ))
    data = pd.DataFrame(records)
    data.to_csv(out / 'historical_postfreeze_new_v5_predictions.csv', index=False)
    return data


def main():
    from nested_species_ensemble_v5 import OUT, V4, SEED, verify_freeze, verify_history
    freeze = verify_freeze()
    if (OUT / 'comparison_audit.json').exists():
        raise FileExistsError('Never replace a completed comparison.')
    done = json.loads((OUT / 'evaluation_audit.json').read_text())
    if done['completed_utc'] <= freeze['created_utc']:
        raise ValueError('Evaluation predates freeze.')
    internal = pd.read_csv(OUT / 'recording_predictions.csv')
    external = pd.read_csv(OUT / 'external_recording_predictions.csv')
    baseline = pd.read_csv(V4 / 'paired_recording_predictions.csv')
    frame = pd.read_csv(OUT / 'segments_v5.csv')
    mapping = frame.groupby('group').pcm_block.first()
    all_scopes = [('dryad', internal)] + list(external.groupby('scope'))
    v5_new_sub = external.query('scope=="postfreeze_new_v5"')
    oldnew = historical_predictions_on_v5_new(v5_new_sub, OUT) if len(v5_new_sub) else pd.DataFrame()
    rows = []
    taxa = []
    pairs = []
    intervals = {}
    matrices = OUT / 'confusion_matrices'
    matrices.mkdir(exist_ok=False)
    for scope, new in all_scopes:
        new = new[new.label.ne('noise')].sort_values('group').reset_index(drop=True)
        if new.empty:
            continue
        if new.group.duplicated().any():
            raise ValueError('Duplicate recording identity.')
        clusters = new.group.map(mapping).fillna(new.group).to_numpy(str)
        for task, truth, column, acceptcol in [
            ('species', new.label, 'predicted_species', 'accepted_species'),
            ('genus_marginalized', new.true_genus, 'genus_marginalized', 'accepted_genus'),
            ('genus_posthoc', new.true_genus, 'genus_posthoc', None),
        ]:
            predictions = {}
            accepted = {}
            if scope == 'postfreeze_new_v5':
                key = {'species': 'species', 'genus_marginalized': 'genus', 'genus_posthoc': 'posthoc'}[task]
                for version in ['v1', 'v2', 'v3', 'v4']:
                    old = oldnew[oldnew.version.eq(version)].set_index('group').reindex(new.group)
                    predictions[version] = old[key].to_numpy()
                    accepted[version] = old[acceptcol].to_numpy(bool) if acceptcol else np.ones(len(new), bool)
            else:
                old = baseline[baseline.scope.eq(scope) & baseline.task.eq(task)].set_index('group').reindex(new.group)
                for version in ['v1', 'v2', 'v3', 'v4']:
                    predictions[version] = old['prediction_' + version].to_numpy()
                    accepted[version] = old['accepted_' + version].to_numpy(bool)
            predictions['v5'] = np.where(new.quality_ok, new[column], 'INVALID')
            accepted['v5'] = new[acceptcol].to_numpy(bool) if acceptcol else new.quality_ok.to_numpy(bool)
            intervals[scope + '|' + task] = dict(
                v1_reference=bootstrap_metrics(truth, predictions, accepted, seed=SEED, clusters=clusters),
                v4_reference=bootstrap_metrics(
                    truth, {v: predictions[v] for v in ['v4', 'v5']},
                    {v: accepted[v] for v in ['v4', 'v5']}, seed=SEED, clusters=clusters,
                ),
            )
            for version, pred in predictions.items():
                m = classification_metrics(truth, pred, accepted[version])
                m['predicted_taxa_outside_support'] = '|'.join(m['predicted_taxa_outside_support'])
                rows.append(dict(scope=scope, task=task, version=version, unit='recording', **m))
                labels = np.union1d(np.unique(truth), np.unique(pred))
                cm = confusion_matrix(truth, pred, labels=labels)
                pd.DataFrame(cm, index=labels, columns=labels).to_csv(matrices / f'{scope}_{task}_{version}.csv')
                for taxon in labels:
                    y = np.asarray(truth)
                    tp = int(((y == taxon) & (pred == taxon)).sum())
                    support = int((y == taxon).sum())
                    count = int((pred == taxon).sum())
                    taxa.append(dict(
                        scope=scope, task=task, version=version, taxon=taxon,
                        support=support, correct=tp, predicted=count,
                        recall=tp / support if support else None,
                        precision=tp / count if count else None,
                    ))
            for i, g in enumerate(new.group):
                pairs.append(dict(
                    scope=scope, task=task, group=g, pcm_block=clusters[i], label=np.asarray(truth)[i],
                    **{'prediction_' + v: predictions[v][i] for v in predictions},
                    **{'accepted_' + v: bool(accepted[v][i]) for v in accepted},
                ))
        m = classification_metrics(new.true_genus, np.where(new.quality_ok, new.genus_raw_sum, 'INVALID'), new.quality_ok)
        m['predicted_taxa_outside_support'] = '|'.join(m['predicted_taxa_outside_support'])
        rows.append(dict(scope=scope, task='genus_raw_sum', version='v5', unit='recording', **m))
    metrics = pd.DataFrame(rows)
    taxa = pd.DataFrame(taxa)
    metrics.to_csv(OUT / 'comparison_metrics.csv', index=False)
    taxa.to_csv(OUT / 'per_taxon.csv', index=False)
    pd.DataFrame(pairs).to_csv(OUT / 'paired_recording_predictions.csv', index=False)
    atomic_json(OUT / 'bootstrap_intervals.json', intervals)
    atomic_json(OUT / 'comparison_audit.json', dict(
        completed_utc=now(), exact_v2_v3_v4_v5_partitions=True,
        historical_pairs_sha256=file_hash(V4 / 'paired_recording_predictions.csv'),
        no_retuning=True, historical_files=verify_history(), release_allowed=False,
    ))
    verify_freeze()
    verify_history()
    print(metrics[metrics.task.isin(['species', 'genus_marginalized'])].to_string(index=False))


if __name__ == '__main__':
    main()
