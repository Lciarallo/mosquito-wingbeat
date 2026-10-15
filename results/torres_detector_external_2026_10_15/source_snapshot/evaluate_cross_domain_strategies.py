"""Frozen comparison, explicit uncertainty, byte/PCM novelty and Drive control."""
import hashlib, json
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.signal import resample_poly
from train_cross_domain_strategies import ROOT, OUT
from prepare_reliability_data import CACHE as BASE_CACHE
from cross_domain_strategy_models import StrategyNet, multipitch_features
from corrected_validation import audio_windows
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from reliability_metrics import binary_metrics
from evaluate_sex_balanced_reliability import bootstrap_values
from train_reproduction import sha, now
from aedes_binary import targets


def pcmhash(a):
    pcm = np.rint(np.clip(a, -1, 32767 / 32768) * 32768).astype('<i2')
    # Constant/silent matches are reported separately, not called duplicate flight.
    return hashlib.sha256(pcm.tobytes()).hexdigest(), bool(np.ptp(pcm.astype('int32')) > 2)


class Predictor:
    def __init__(self):
        self.freeze = json.loads((OUT / 'freeze.json').read_text())
        for p, h in {**self.freeze['models_sha256'], **self.freeze['source_sha256']}.items(): assert sha(ROOT / p) == h, p
        for name in ['protocol', 'segments', 'selection']:
            assert sha(OUT / (name + ('.csv' if name == 'segments' else '.json'))) == self.freeze[name + '_sha256']
        self.selected = json.loads((OUT / 'selection.json').read_text())['selected_model']
        torch.set_num_threads(4); self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}; self.cuts = {}
        for path in self.freeze['models_sha256']:
            p = ROOT / path
            if p.suffix == '.pt':
                s = torch.load(p, map_location='cpu', weights_only=True)
                model = StrategyNet(len(s['taxa'])).to(self.device); model.load_state_dict(s['state_dict']); model.eval()
                self.models[s['name']] = ('neural', model)
            else:
                s = joblib.load(p); self.models[s['name']] = ('multipitch', s['model'])
            self.cuts[s['name']] = s['cut']
        # Old models are locked comparisons, not candidates in current selection.
        names = ['humbug__svm_rbf', 'dryad+humbug__svm_rbf', 'dryad__logistic']
        self.baseline_hashes = {}
        for name in names:
            path = ROOT / 'results/reduced_databases_2026_10_02/models' / (name + '.joblib')
            s = joblib.load(path); key = 'prior_' + name
            self.models[key] = ('prior', s['model']); self.cuts[key] = s['cut']; self.baseline_hashes[key] = sha(path)

    def audio(self, path):
        a, q, size = audio_windows(path); a8 = resample_poly(a, 1, 2, axis=1).astype('float32')
        out = {name: [] for name in self.models}
        # Raw physical features for old baselines only.
        previous = np.concatenate([features(a), extract_batch_bioacoustic_features(a)], 1)
        for start in range(0, len(a8), 96):
            w = torch.tensor(a8[start:start + 96], device=self.device)
            with torch.no_grad():
                engineered = multipitch_features(w).cpu().numpy()
                for name, (kind, m) in self.models.items():
                    if kind == 'neural': score = m(w)[0].cpu().numpy()
                    elif kind == 'prior': score = m.decision_function(previous[start:start + 96])
                    elif hasattr(m, 'decision_function'): score = m.decision_function(engineered)
                    else: score = m.predict_proba(engineered)[:, 1]
                    out[name].extend(score)
        return {k: np.asarray(v) for k, v in out.items()}, q, size, a8


def main():
    p = Predictor(); plan = json.loads((OUT / 'fresh_selection.json').read_text())
    assert plan['selection_utc'] > p.freeze['frozen_utc']
    assert plan['models_sha256'] == p.freeze['models_sha256']
    # Decoded fragment novelty across the base and all downloaded prior tests.
    seen = {}; groups = set(pd.read_csv(ROOT / 'results/reliability_2026_10_01/segments.csv', low_memory=False).group)
    for bucket in ['long', 'short']:
        for a in np.load(BASE_CACHE / (bucket + '_8k.npy'), mmap_mode='r'):
            h, meaningful = pcmhash(a); seen[h] = seen.get(h, False) or meaningful
    previous_paths = set()
    for manifest in (ROOT / 'results').rglob('*selection.json'):
        if manifest == OUT / 'fresh_selection.json': continue
        try:
            data = json.loads(manifest.read_text())
            previous_paths.update(item['path'] for item in data.get('files', []) if 'path' in item)
        except (ValueError, TypeError): pass
    for relative in previous_paths:
        path = ROOT / relative
        if not path.exists(): continue
        try:
            a, _, _ = audio_windows(path)
        except (ValueError, RuntimeError): continue
        for w in resample_poly(a, 1, 2, axis=1):
            h, meaningful = pcmhash(w); seen[h] = seen.get(h, False) or meaningful
    records = []; audit = []; accepted = set()
    for item in plan['files']:
        path = ROOT / item['path']; assert sha(path) == item['sha256']; assert item['group'] not in groups
        a, _, _ = audio_windows(path); hashes = [pcmhash(w) for w in resample_poly(a, 1, 2, axis=1)]
        matches = sum(h in seen and meaningful and seen[h] for h, meaningful in hashes)
        constants = sum(h in seen and not meaningful for h, meaningful in hashes)
        duplicate = matches > 0 or item['sha256'] in accepted; accepted.add(item['sha256'])
        audit.append(dict(path=item['path'],nonconstant_pcm_matches=matches,constant_pcm_matches=constants,
                          excluded=duplicate, original_individual_identity_available=False))
        if duplicate: continue
        scores, q, size, _ = p.audio(path)
        for name, v in scores.items():
            records.extend(dict(model=name, path=item['path'], source=item['source'], group=item['group'],
                                label=int(item['reference_aedes']), taxon=item['reference_species'],
                                quality_ok=bool(quality), score=float(s), start_s=j * size / 16000)
                           for j, (s, quality) in enumerate(zip(v, q)))
        print('INFERENCE', path.name, flush=True)
    d = pd.DataFrame(records); d.to_csv(OUT / 'fresh_predictions.csv.gz', index=False)
    pd.DataFrame(audit).to_csv(OUT / 'novelty_audit.csv', index=False)
    rows = []; confidence = {}; values = {}; file_rows = []
    for name, a in d.groupby('model'):
        cut = p.cuts[name]
        rows.append(dict(model=name, scope='all_fresh', selected=name == p.selected, files=a.group.nunique(),
                         **binary_metrics(a.label, a.score, a.quality_ok, cut)))
        for taxon, b in a.groupby('taxon'):
            rows.append(dict(model=name, scope=taxon, selected=name == p.selected, files=b.group.nunique(),
                             **binary_metrics(b.label, b.score, b.quality_ok, cut)))
        # All sources equal here; bootstrap stratifies by class to retain positives.
        a = a.copy(); a['source'] = a.label.map({0: 'negative', 1: 'positive'})
        v, count = bootstrap_values(a, cut, 2000); values[name] = v
        confidence[name] = dict(groups=count, metrics={k: dict(lower95=float(np.quantile(z, .025)), upper95=float(np.quantile(z, .975))) for k, z in v.items()})
        for group, b in a.groupby('group'):
            file_rows.append(dict(model=name, group=group, label=int(b.label.iloc[0]),
                                  **binary_metrics(b.label, b.score, b.quality_ok, cut)))
    pd.DataFrame(rows).to_csv(OUT / 'fresh_metrics.csv', index=False)
    pd.DataFrame(file_rows).to_csv(OUT / 'file_metrics.csv', index=False)
    (OUT / 'confidence_intervals.json').write_text(json.dumps(confidence, indent=2))
    comparisons = {}
    for baseline in ['ridge_control', 'prior_humbug__svm_rbf', 'prior_dryad+humbug__svm_rbf']:
        comparisons[baseline] = {k: dict(mean_delta=float(np.mean(values[p.selected][k] - values[baseline][k])),
                                        lower95=float(np.quantile(values[p.selected][k] - values[baseline][k], .025)),
                                        upper95=float(np.quantile(values[p.selected][k] - values[baseline][k], .975)))
                                 for k in values[p.selected]}
    (OUT / 'paired_comparisons.json').write_text(json.dumps(comparisons, indent=2))
    # Original locked test/challenge partitions: retrospective regression checks,
    # never used to choose this round's checkpoint, threshold or candidate.
    base = pd.read_csv(ROOT / 'results/reliability_2026_10_01/segments.csv', low_memory=False)
    historical = base[base.split.isin(['test', 'challenge']) & (targets(base) >= 0) &
                      base.domain.isin(['dryad', 'humbug', 'mosquitosong', 'ufrgs'])]
    original_features = np.load(BASE_CACHE / 'features.npy', mmap_mode='r')
    waves = {b: np.load(BASE_CACHE / (b + '_8k.npy'), mmap_mode='r') for b in ['long', 'short']}
    historical_rows = []
    for domain, frame in historical.groupby('domain'):
        outputs = {name: np.empty(len(frame)) for name in p.models}
        for bucket, part in frame.groupby('bucket'):
            positions = frame.index.get_indexer(part.index); wi = part.wave_index.astype(int).to_numpy()
            for start in range(0, len(part), 96):
                ix = positions[start:start + 96]
                w = torch.tensor(np.array(waves[bucket][wi[start:start + 96]]), device=p.device)
                with torch.no_grad():
                    physical = multipitch_features(w).cpu().numpy()
                    for name, (kind, m) in p.models.items():
                        if kind == 'neural': v = m(w)[0].cpu().numpy()
                        elif kind == 'prior': v = m.decision_function(original_features[frame.iloc[ix].index])
                        elif hasattr(m, 'decision_function'): v = m.decision_function(physical)
                        else: v = m.predict_proba(physical)[:, 1]
                        outputs[name][ix] = v
        for name, v in outputs.items():
            historical_rows.append(dict(model=name, scope=domain, selected=name == p.selected,
                                        **binary_metrics(targets(frame), v, frame.quality_ok, p.cuts[name])))
        print('HISTORICAL', domain, len(frame), flush=True)
    pd.DataFrame(historical_rows).to_csv(OUT / 'historical_metrics.csv', index=False)
    scores, q, _, _ = p.audio(ROOT / 'tmp/private-audio-test/download.audio'); private = []
    for name, v in scores.items():
        private.append(dict(model=name, selected=name == p.selected,
                            **binary_metrics(np.ones(len(v), int), v, q, p.cuts[name])))
    pd.DataFrame(private).to_csv(OUT / 'drive_positive_control.csv', index=False)
    chosen = next(r for r in rows if r['scope'] == 'all_fresh' and r['selected'])
    failures = []
    for metric in ['aedes_precision', 'aedes_recall', 'specificity', 'macro_f1']:
        if chosen[metric] is None or chosen[metric] < .6: failures.append(metric + ' below .60')
        if confidence[p.selected]['metrics'][metric]['lower95'] < .6: failures.append(metric + ' lower95 below .60')
    for path, h in json.loads((OUT / 'device_files_sha256.json').read_text()).items(): assert sha(ROOT / path) == h
    for path, h in json.loads((OUT / 'original_weights_sha256.json').read_text()).items(): assert sha(ROOT / path) == h
    decision = dict(selected_model=p.selected, reliable=not failures, failures=failures, completed_utc=now(),
                    private_only_positive=True, new_test_files=int(chosen['files']), no_test_tuning=True,
                    baseline_hashes=p.baseline_hashes, arduino_compiled=False)
    (OUT / 'reliability_decision.json').write_text(json.dumps(decision, indent=2))
    print(pd.DataFrame(rows).query("scope=='all_fresh'")[[ 'model', 'aedes_precision', 'aedes_recall', 'specificity', 'balanced_accuracy']].to_string(index=False), flush=True)
    print(pd.DataFrame(private)[['model', 'true_positive', 'known_windows']].to_string(index=False), flush=True)


if __name__ == '__main__': main()
