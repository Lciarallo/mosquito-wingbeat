"""Predeclared ablations; all tuning restricted to train/validation/calibration."""
import copy, json, time
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from aedes_binary import ROOT, targets, weights
from corrected_validation import assert_disjoint
from prepare_reliability_data import CACHE as BASE_CACHE
from train_reproduction import now, sha
from reliability_metrics import binary_metrics
from reliability_models import channel_augmentation
from cross_domain_strategy_models import (StrategyNet, cross_domain_contrast,
                                         conditional_mmd, mix_known_species,
                                         multipitch_features, FEATURE_NAMES)

OUT = ROOT / 'results/cross_domain_strategies_2026_10_03'
CACHE = ROOT / 'data/cache/cross_domain_strategies_2026_10_03'
CONFIGS = [dict(name='ridge_control', contrast=False, swarm=False),
           dict(name='ridge_contrast', contrast=True, swarm=False),
           dict(name='ridge_contrast_swarm', contrast=True, swarm=True)]
SEED = 20261003


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def cells(frame, p, cut):
    y = targets(frame); result = {}
    for domain in sorted(frame.domain.unique()):
        for label in [0, 1]:
            use = frame.domain.eq(domain).to_numpy() & (y == label)
            if use.any():
                result[f'{domain}:{label}'] = float(np.average((p[use] >= cut) == y[use],
                                                               weights=weights(frame.loc[use])))
    return result


def calibration_cut(frame, score, *, partition):
    if partition != 'calibration':
        raise ValueError('Only calibration data may choose the threshold.')
    candidates = np.unique(np.r_[score.min() - 1e-6,
                                np.quantile(score, np.linspace(0, 1, 201)),
                                score.max() + 1e-6])
    ranked = [(min(cells(frame, score, c).values()),
               np.mean(list(cells(frame, score, c).values())), float(c)) for c in candidates]
    return max(ranked, key=lambda row: (row[0], row[1]))[2]


def validation_result(frame, score, cut):
    detail = cells(frame, score, cut)
    m = binary_metrics(targets(frame), score, np.ones(len(frame), bool), cut)
    auc = {source: float(roc_auc_score(targets(a), score[frame.index.get_indexer(a.index)],
                                      sample_weight=weights(a)))
           for source, a in frame.groupby('domain')}
    return dict(validation_precision=m['aedes_precision'], validation_recall=m['aedes_recall'],
                validation_macro_f1=m['macro_f1'], validation_cells=detail,
                validation_auc=auc, selection_score=min(detail.values()) + .05 * np.mean(list(detail.values())))


def infer(model, frame, waves, device, batch=96):
    model.eval(); out = np.empty(len(frame), dtype='float32')
    with torch.no_grad():
        for bucket, sub in frame.groupby('bucket'):
            positions = frame.index.get_indexer(sub.index)
            indices = sub.wave_index.astype(int).to_numpy()
            for start in range(0, len(sub), batch):
                a = torch.tensor(np.array(waves[bucket][indices[start:start + batch]]), device=device)
                out[positions[start:start + batch]] = model(a)[0].cpu().numpy()
    return out


def main():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True); (OUT / 'models').mkdir(exist_ok=True)
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Do not retune a frozen experiment.')
    torch.set_num_threads(4); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED); np.random.seed(SEED)
    base = pd.read_csv(ROOT / 'results/reliability_2026_10_01/segments.csv', low_memory=False)
    assert_disjoint(base)
    f = base[base.domain.isin(['dryad', 'humbug']) & base.quality_ok &
             (targets(base) >= 0) & base.split.isin(['train', 'validation', 'calibration'])].copy()
    assert_disjoint(f); f.to_csv(OUT / 'segments.csv', index=True, index_label='original_row')
    frames = {s: f[f.split.eq(s)].copy() for s in ['train', 'validation', 'calibration']}
    tr, val, cal = [frames[s] for s in frames]
    assert not tr.path.str.contains('private|download.audio|ufrgs|S_1_D_3', case=False, regex=True).any()
    original = json.loads((ROOT / 'results/sex_balanced_reliability_2026_10_01/original_weights_sha256.json').read_text())
    devices = json.loads((ROOT / 'results/sex_balanced_reliability_2026_10_01/device_files_before_sha256.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in {**original, **devices}.items())
    write('original_weights_sha256.json', original); write('device_files_sha256.json', devices)
    taxa = sorted(tr.label.unique()); labels = {label: i for i, label in enumerate(taxa)}
    species = torch.tensor(tr.label.map(labels).to_numpy(), device=device)
    domain = torch.tensor(tr.domain.map({'dryad': 0, 'humbug': 1}).to_numpy(), device=device)
    group = torch.tensor(pd.Categorical(tr.group).codes.astype('int64'), device=device)
    y = torch.tensor(targets(tr), device=device, dtype=torch.float32)
    write('protocol.json', dict(created_utc=now(), seed=SEED, configs=CONFIGS, taxa=taxa,
          training_sources=['dryad', 'humbug'], other_sources_excluded_from_fit=True,
          training_groups=int(tr.group.nunique()), training_windows=len(tr),
          representation='STFT 2048/128 at8k, real frames, 200-3500Hz, frequency/time local contrast, no waveform extension',
          auxiliary_losses='exact-taxon CE .2, cross-domain same-taxon different-group contrast .08 tau=.1, conditional RBF MMD .15',
          simulation='up to 50% same exact taxon/different-group superpositions; no mixed-taxon species labels',
          epochs_max=24, steps_per_epoch=80, batch_size=96, early_stop='minimum source weighted validation AUC, patience6 after epoch10',
          calibration='maximize minimum source/class/group-balanced recall; no test/Drive tuning',
          model_selection='validation minimum source/class weighted recall at frozen calibration cut + .05 mean recall',
          fresh_test='After freeze: first16 unused Ae acoustic D3 and first4 unused files each of IDs2/7/8/9, CRC/hash/PCM novelty audit',
          fresh_data_limitations=['file labels do not annotate flight in every window', 'CDMSC original individuals/sessions unknown'],
          reliability_gate=dict(precision_min=.6, recall_min=.6, specificity_min=.6, macro_f1_min=.6, lower95_min=.6),
          private_in_training=False, arduino_compiled=False, paper_replication='DR-BioL-inspired objectives, not exact framework/architecture reproduction',
          biological_covariates='No temperature/sex imputed; unavailable factors cannot be validated away.'))
    waves = {b: np.load(BASE_CACHE / (b + '_8k.npy'), mmap_mode='r') for b in ['long', 'short']}
    # Prepare exactly the new deterministic features for fit partitions only.
    featpath = CACHE / 'multipitch_features.npy'
    if not featpath.exists():
        x = np.empty((len(f), len(FEATURE_NAMES)), dtype='float32')
        for bucket, sub in f.groupby('bucket'):
            positions = f.index.get_indexer(sub.index); indices = sub.wave_index.astype(int).to_numpy()
            for start in range(0, len(sub), 192):
                a = torch.tensor(np.array(waves[bucket][indices[start:start + 192]]), device=device)
                x[positions[start:start + 192]] = multipitch_features(a).cpu().numpy()
        assert np.isfinite(x).all(); np.save(featpath, x)
    x = np.load(featpath); assert x.shape == (len(f), len(FEATURE_NAMES))
    write('feature_names.json', FEATURE_NAMES)
    pool = torch.tensor(np.array(waves['long'][tr.wave_index.astype(int)]), device=device)
    assert tr.bucket.eq('long').all()
    tw = weights(tr)
    def sample_mask(mask, count):
        ix = np.flatnonzero(mask)
        if not len(ix):
            raise ValueError('Required sampling cell absent.')
        p = torch.tensor(tw[ix], device=device)
        return torch.tensor(ix, device=device)[torch.multinomial(p, count, replacement=True)]
    common = tr[tr.label.ne('noise') & (targets(tr) == 0)].groupby('label').domain.nunique()
    shared = common[common.eq(2)].index.tolist(); assert shared
    rng = np.random.default_rng(SEED)
    def sample():
        selected = []
        for d in ['dryad', 'humbug']:
            selected.append(sample_mask(tr.domain.eq(d).to_numpy() & (targets(tr) == 1), 24))
            selected.append(sample_mask(tr.domain.eq(d).to_numpy() & (targets(tr) == 0) & tr.label.ne('noise').to_numpy(), 8))
            selected.append(sample_mask(tr.domain.eq(d).to_numpy() & tr.label.eq('noise').to_numpy(), 8))
        for label in rng.choice(shared, size=4, replace=False):
            for d in ['dryad', 'humbug']:
                selected.append(sample_mask(tr.domain.eq(d).to_numpy() & tr.label.eq(label).to_numpy(), 2))
        return torch.cat(selected)
    records = []
    for ci, config in enumerate(CONFIGS):
        name = config['name']; path = OUT / 'models' / (name + '.pt')
        if path.exists():
            records.append(json.loads((OUT / (name + '_training.json')).read_text())); continue
        torch.manual_seed(SEED + ci); rng = np.random.default_rng(SEED + ci)
        model = StrategyNet(len(taxa)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 24, eta_min=.00002)
        best = -np.inf; stale = 0; history = []; started = time.time()
        print('TRAIN', name, sum(p.numel() for p in model.parameters()), flush=True)
        for epoch in range(1, 25):
            model.train(); losses = []; contrast_values = []; anchors = []; mixtures = 0
            for step in range(80):
                ix = sample(); a = pool[ix]
                if config['swarm']:
                    a, pairs = mix_known_species(a, species[ix], group[ix]); mixtures += len(pairs)
                a = channel_augmentation(a)
                # Same noise augmentation in all ablations; real training noise only.
                ni = sample_mask(tr.label.eq('noise').to_numpy(), len(ix)); snr = 5 + 25 * torch.rand(len(ix), device=device)
                a = a + channel_augmentation(pool[ni]) * (10 ** (-snr / 20) * (torch.rand(len(ix), device=device) < .5))[:, None]
                opt.zero_grad(set_to_none=True); z, taxon, emb = model(a)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y[ix])
                if config['contrast']:
                    cl, count = cross_domain_contrast(emb, species[ix], domain[ix], group[ix])
                    mm = conditional_mmd(emb, species[ix], domain[ix])
                    loss = loss + .2 * torch.nn.functional.cross_entropy(taxon, species[ix]) + .08 * cl + .15 * mm
                    contrast_values.append(float(cl.detach())); anchors.append(count)
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); losses.append(float(loss.detach()))
            scheduler.step(); zv = infer(model, val, waves, device)
            auc = {d: float(roc_auc_score(targets(a), zv[val.index.get_indexer(a.index)], sample_weight=weights(a))) for d, a in val.groupby('domain')}
            early_score = min(auc.values())
            row = dict(epoch=epoch, loss=float(np.mean(losses)), validation_auc=auc, early_score=early_score,
                       contrast_loss=float(np.mean(contrast_values)) if contrast_values else None,
                       contrast_anchors_mean=float(np.mean(anchors)) if anchors else 0,
                       simulated_mixtures=mixtures, elapsed_s=time.time() - started)
            history.append(row); write(name + '_history.json', history); print(name, json.dumps(row), flush=True)
            if early_score > best + 1e-4:
                best = early_score; best_epoch = epoch; state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); stale = 0
            else:
                stale += 1
            if epoch >= 10 and stale >= 6:
                break
        model.load_state_dict(state)
        zc = infer(model, cal, waves, device); cut = calibration_cut(cal, zc, partition='calibration')
        zv = infer(model, val, waves, device)
        saved = dict(name=name, state_dict=state, taxa=taxa, cut=cut, best_epoch=best_epoch)
        torch.save(saved, path)
        record = dict(name=name, best_epoch=best_epoch, cut=cut, parameters=sum(p.numel() for p in model.parameters()),
                      sha256=sha(path), **validation_result(val, zv, cut))
        write(name + '_training.json', record); records.append(record); del model, opt; torch.cuda.empty_cache()
    del pool; torch.cuda.empty_cache()
    # Classical comparison on the new multiple-candidate descriptors alone.
    ti = f.index.get_indexer(tr.index); vi = f.index.get_indexer(val.index); ca = f.index.get_indexer(cal.index)
    for method in ['svm_multipitch', 'trees_multipitch']:
        model = (make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', cache_size=1000)) if method.startswith('svm')
                 else ExtraTreesClassifier(n_estimators=400, max_depth=16, min_samples_leaf=3, n_jobs=4, random_state=SEED))
        kw = {'svc__sample_weight': tw} if method.startswith('svm') else {'sample_weight': tw}
        print('CLASSICAL', method, flush=True); model.fit(x[ti], targets(tr), **kw)
        predict = lambda a: model.decision_function(a) if method.startswith('svm') else model.predict_proba(a)[:, 1]
        cut = calibration_cut(cal, predict(x[ca]), partition='calibration')
        path = OUT / 'models' / (method + '.joblib'); joblib.dump(dict(name=method, model=model, cut=cut), path)
        record = dict(name=method, cut=cut, sha256=sha(path), **validation_result(val, predict(x[vi]), cut))
        write(method + '_training.json', record); records.append(record)
    write('validation_search.json', records)
    chosen = max(records, key=lambda r: r['selection_score'])
    write('selection.json', dict(selected_model=chosen['name'], selected_before_test=True, **chosen))
    assert all(sha(ROOT / p) == h for p, h in {**original, **devices}.items())
    write('freeze.json', dict(frozen_utc=now(), models_sha256={str(p.relative_to(ROOT)): sha(p) for p in (OUT / 'models').iterdir()},
                             source_sha256={p: sha(ROOT / p) for p in ['cross_domain_strategy_models.py', 'train_cross_domain_strategies.py']},
                             protocol_sha256=sha(OUT / 'protocol.json'), segments_sha256=sha(OUT / 'segments.csv'),
                             selection_sha256=sha(OUT / 'selection.json'), arduino_compiled=False))
    print('FROZEN', chosen['name'], flush=True)


if __name__ == '__main__':
    main()
