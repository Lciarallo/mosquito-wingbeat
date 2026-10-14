"""Verify saved presence/genus/species decisions without training or retuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from evaluate_nested_species import verify_freeze
from research_metrics_v2 import classification_metrics

ROOT = Path(__file__).resolve().parent
SPECIES = ROOT / 'results/species_nested_2026_10_09'
OUT = ROOT / 'results/three_levels_2026_10_09'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Choose a new output directory; preserve completed analyses.')
    verify_freeze(SPECIES)
    inputs = ['results/arduino/predictions.csv.gz', 'results/arduino/metrics.csv',
              'results/species_nested_2026_10_09/recording_predictions.csv',
              'results/species_nested_2026_10_09/external_recording_predictions.csv',
              'results/species_nested_2026_10_09/metrics.csv',
              'results/species_nested_2026_10_09/external_metrics.csv',
              'results/species_nested_2026_10_09/genus_analysis/predictions.csv.gz',
              'results/species_nested_2026_10_09/genus_analysis/metrics.csv']
    rows = []
    presence = pd.read_csv(ROOT / inputs[0])
    reference = pd.read_csv(ROOT / inputs[1])
    for mode, f in presence[presence.model.eq('Escolha por validação interna')].groupby('mode'):
        y = f.label.ne('noise').to_numpy()
        pred = f.detected.to_numpy(dtype=bool)
        assert np.array_equal(pred, f.score.ge(f.threshold)), 'Saved presence decisions changed.'
        tn, fp = int((~y & ~pred).sum()), int((~y & pred).sum())
        fn, tp = int((y & ~pred).sum()), int((y & pred).sum())
        recall, fpr = tp/(tp+fn), fp/(tn+fp)
        old = reference[reference.model.eq('Escolha por validação interna') & reference['mode'].eq(mode)].iloc[0]
        assert np.isclose(recall, old.window_recall) and np.isclose(fpr, old.window_false_positive_rate)
        rows.append(dict(task='presence', scope='historical_corpus', unit='window', model=mode,
                         n=len(f), groups=f.group.nunique(), correct=tp+tn,
                         accuracy=(tp+tn)/len(f), balanced_accuracy=(recall+1-fpr)/2,
                         recall=recall, false_positive_rate=fpr,
                         positive_windows=tp+fn, negative_windows=tn+fp,
                         true_positive=tp, false_negative=fn, true_negative=tn, false_positive=fp))

    genus = pd.read_csv(ROOT / inputs[6])
    reference = pd.read_csv(ROOT / inputs[7])
    per_genus = []
    for scope in ['dryad', 'external']:
        g = genus[genus.model.eq('calibrated') & genus.unit.eq('recording') & genus.scope.eq(scope)]
        y, pred = g.true_genus, g.predicted_genus
        old = reference[reference.model.eq('calibrated') & reference.unit.eq('recording') & reference.scope.eq(scope)].iloc[0]
        acc = accuracy_score(y, pred)
        assert np.isclose(acc, old.accuracy) and int((y == pred).sum()) == old.correct
        rows.append(dict(task='genus', scope=scope, unit='recording', model='calibrated', n=len(g),
                         groups=g.group.nunique(), correct=int((y == pred).sum()), accuracy=acc,
                         balanced_accuracy=balanced_accuracy_score(y, pred),
                         macro_f1=f1_score(y, pred, labels=np.unique(y), average='macro', zero_division=0),
                         **{k:v for k,v in classification_metrics(y,pred).items()
                            if k.startswith('macro_f1_') or k=='predicted_taxa_outside_support'}))
        for taxon, part in g.groupby('true_genus'):
            per_genus.append(dict(scope=scope, genus=taxon, n=len(part),
                                  correct=int(part.predicted_genus.eq(taxon).sum()),
                                  recall=float(part.predicted_genus.eq(taxon).mean())))
        path = inputs[2] if scope == 'dryad' else inputs[3]
        s = pd.read_csv(ROOT / path)
        s = s[s.model.eq('calibrated')]
        assert set(s.group) == set(g.group), 'Genus and species use different recordings.'
        truth = s.true_species if scope == 'dryad' else s.label
        predicted = s.predicted_species if scope == 'dryad' else s.forced_species
        accepted = s.accepted.to_numpy(dtype=bool)
        correct = truth.eq(predicted).to_numpy()
        table = pd.read_csv(ROOT / (inputs[4] if scope == 'dryad' else inputs[5]))
        old = table[table.model.eq('calibrated') & table.unit.eq('recording')].iloc[0]
        assert np.isclose(correct.mean(), old.accuracy)
        assert np.isclose(accepted.mean(), old.coverage)
        assert np.isclose(correct[accepted].mean(), old.accepted_accuracy)
        rows.append(dict(task='species', scope=scope, unit='recording', model='calibrated', n=len(s),
                         groups=s.group.nunique(), correct=int(correct.sum()), accuracy=float(correct.mean()),
                         balanced_accuracy=balanced_accuracy_score(truth, predicted),
                         coverage=float(accepted.mean()), emitted=int(accepted.sum()),
                         correct_emissions=int((correct & accepted).sum()),
                         accepted_accuracy=float(correct[accepted].mean())))
    verify_freeze(SPECIES)
    OUT.mkdir()
    pd.DataFrame(rows).to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(per_genus).to_csv(OUT / 'per_genus.csv', index=False)
    (OUT / 'audit.json').write_text(json.dumps(dict(
        analysis='Retrospective verification of saved decisions; no new training or acquisition.',
        presence_is_separate_historical_model=True, presence_external_with_noise_not_validated=True,
        genus_species_recording_sets_match=True, genus_post_hoc_forced_choice=True,
        inputs_sha256={p:sha(ROOT/p) for p in inputs}, code_sha256=sha(Path(__file__)),
        can_release_to_user_device=False), indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
