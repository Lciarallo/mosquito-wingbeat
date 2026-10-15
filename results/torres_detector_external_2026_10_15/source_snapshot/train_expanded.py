"""Expand the Arduino classifier without contaminating held-out source groups.

Run fetch_humbug.py first. Keep the baseline and its historical audits intact;
save the retrained model for offline evaluation, without Arduino compilation.
"""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import arduino_frontend as frontend
from arduino_models import TinyPresenceMLP
from arduino_species_models import TinySpeciesMLP, fused_layers, probabilities
import train_arduino_species as species_export
from improve_device import file_hash, atomic_json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/external/humbug'
OUT = ROOT / 'results/expanded'
CACHE = ROOT / 'data/cache/expanded'
SEED = 20260930
WINDOW = frontend.WINDOW_SAMPLES


def source_weights(frame, target):
    """Equal target mass; equal corpus mass within target; equal source mass.

    Class counts are not enough: several excerpts can belong to the same source.
    A pair (group, target) can contain positive and negative excerpts.
    """
    table = pd.DataFrame(dict(group=frame.group.to_numpy(), domain=frame.domain.to_numpy(), target=target))
    counts = table.groupby(['target', 'domain', 'group']).size()
    result = 1 / counts.reindex(pd.MultiIndex.from_frame(table[['target', 'domain', 'group']])).to_numpy()
    sources = table.drop_duplicates(['target', 'domain', 'group']).groupby(['target', 'domain']).size()
    result /= sources.reindex(pd.MultiIndex.from_frame(table[['target', 'domain']])).to_numpy()
    domains = table[['target', 'domain']].drop_duplicates().groupby('target').size()
    result /= table.target.map(domains).to_numpy()
    return result / result.mean()


def build_external():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA / 'selected_metadata.csv')
    manifest = json.loads((DATA / 'selected_manifest.json').read_text())
    entries = {item['id']: item for item in manifest['files']}
    blocks, records = [], []
    # Exact byte-identical excerpts share a source group, even with different IDs.
    union = {group: group for group in metadata.group.unique()}
    def root(group):
        while union[group] != group:
            group = union[group]
        return group
    observed = {}
    for row in metadata.itertuples():
        sha = entries[row.id]['sha256']
        if sha in observed:
            union[root(row.group)] = root(observed[sha])
        else:
            observed[sha] = row.group
    metadata['group'] = metadata.group.map(lambda value: 'humbug:' + root(value))
    for row in metadata.itertuples():
        path = DATA / 'audio' / f'{row.id}.wav'
        assert file_hash(path) == entries[row.id]['sha256']
        wave, sr = sf.read(path, dtype='float32', always_2d=True)
        wave = wave.mean(axis=1)
        divisor = np.gcd(sr, frontend.SAMPLE_RATE)
        if sr != frontend.SAMPLE_RATE:
            wave = resample_poly(wave, frontend.SAMPLE_RATE // divisor, sr // divisor)
        pcm = np.rint(np.clip(wave, -1, 32767/32768) * 32768).astype('int16')
        windows = len(pcm) // WINDOW
        if not windows:
            continue
        # Retain at most 96 decisions per excerpt, spaced across longer clips.
        selected = np.unique(np.linspace(0, windows-1, min(windows, 96), dtype=int))
        clips = np.stack([pcm[i*WINDOW:(i+1)*WINDOW] for i in selected])
        xx = frontend.features(clips, replay_pcm=False)
        peaks = np.abs(clips.astype('int32')).max(axis=1)
        clipped = (np.abs(clips.astype('int32')) >= 32700).sum(axis=1)
        valid = (peaks >= 32) & (clipped*100 <= WINDOW)
        blocks.append(xx)
        for j, index in enumerate(selected):
            records.append(dict(path=str(path.relative_to(ROOT)), group=row.group, label=row.label,
                start_s=index*WINDOW/16000, step_s=WINDOW/16000, domain='humbug',
                country=row.country, quality_ok=bool(valid[j]), id=int(row.id),
                original_species=row.species, original_name=row.name,
                original_datetime=row.record_datetime))
    frame = pd.DataFrame(records)
    x = np.concatenate(blocks)
    assert len(frame) == len(x) and np.isfinite(x).all()
    frame.to_csv(CACHE / 'external_segments.csv', index=False)
    np.save(CACHE / 'external_features.npy', x)
    return frame, x


def partitions(frame):
    """Historical fold-2 test stays held out; new groups have four-way splits."""
    parent = json.loads((ROOT / 'results/arduino/protocol.json').read_text())['source_splits'][2]
    rng = np.random.default_rng(SEED)
    split = {}
    original = frame[frame.domain.eq('dryad')]
    for group in parent['test_groups']:
        split[group] = 'test'
    for group in parent['calibration_groups']:
        split[group] = 'calibration'
    fit = original[original.group.isin(parent['fit_groups'])].drop_duplicates('group')
    for _, part in fit.groupby('label', sort=True):
        groups = rng.permutation(part.group.to_numpy())
        nval = min(len(groups)-1, max(1, int(np.ceil(.18*len(groups))))) if len(groups)>1 else 0
        split.update({group: 'validation' if i<nval else 'train' for i, group in enumerate(groups)})
    external = frame[frame.domain.eq('humbug')]
    known = set(frame.label.unique()) - {'unknown_mosquito', 'noise'}
    groups = external.groupby('group', sort=True).label.apply(
        lambda values: next((value for value in values if value in known),
                            'unknown_mosquito' if 'unknown_mosquito' in values.values else 'noise'))
    # Both previously tested recordings (all their excerpts) are adaptation only.
    adaptation = set(external[external.id.isin([220292, 222024])].group)
    unsupported = []
    for label in sorted(groups.unique()):
        values = [group for group in groups[groups.eq(label)].index if group not in adaptation]
        values = rng.permutation(values)
        if len(values)<4:
            unsupported.append(label)
            split.update({group: 'train' for group in values})
            continue
        ntest = max(1, int(np.floor(len(values)*.2)))
        ncal = max(1, int(np.floor(len(values)*.15)))
        nval = max(1, int(np.floor(len(values)*.15)))
        for i, group in enumerate(values):
            split[group] = ('test' if i<ntest else 'calibration' if i<ntest+ncal else
                            'validation' if i<ntest+ncal+nval else 'train')
    split.update({group: 'train' for group in adaptation})
    result = frame.group.map(split)
    assert result.notna().all()
    assert frame.assign(split=result).groupby('group').split.nunique().max() == 1
    return result, unsupported


def assemble():
    ext, ex = build_external()
    original = pd.read_csv(ROOT / 'results/segments.csv')
    original['domain'] = 'dryad'
    original['country'] = 'original_collection'
    original['quality_ok'] = True  # Cache was normalized for replay, not raw microphone PCM.
    original['step_s'] = 1.
    x = np.concatenate([np.load(ROOT/'data/cache/arduino_features.npy'), ex])
    frame = pd.concat([original, ext], ignore_index=True)
    frame['split'], unsupported = partitions(frame)
    frame.to_csv(OUT / 'segments.csv', index=False)
    np.save(CACHE/'combined_features.npy', x)
    counts = frame.groupby(['domain','split','label']).agg(windows=('label','size'), groups=('group','nunique'))
    counts.to_csv(OUT/'data_counts.csv')
    atomic_json(OUT/'data_protocol.json', dict(seed=SEED, original_test_fold=2,
        same_group_disjoint=True, groups=frame[['group','domain','split']].drop_duplicates().to_dict('records'),
        external_classes_without_independent_holdout=unsupported,
        external_manifest_sha256=file_hash(DATA/'selected_manifest.json'),
        window_count=len(frame), original_windows=len(original), external_windows=len(ext),
        original_cache_normalized=True, external_raw_pcm=True,
        unknown_complexes_not_mapped_to_exact_species=True,
        adaptation_only_ids=[220292,222024]))
    print('Data', len(original), 'existing windows +', len(ext), 'external windows', flush=True)
    return frame, x


def gate_scores(bundle, x):
    return expit(bundle['calibration_slope']*bundle['model'].decision_function(x)+bundle['calibration_bias'])


def infer_portable(model, x):
    value = np.asarray(x, dtype='float32')
    layers = sorted(int(key[7:]) for key in model if key.startswith('weights'))
    for layer in layers:
        value = value@model[f'weights{layer}'].T+model[f'bias{layer}']
        if layer != layers[-1]:
            value = np.maximum(value, 0)
    probability = softmax(value/model['temperature'], axis=1)
    hidden = np.maximum(0, x@model['presence_weights0'].T+model['presence_bias0'])
    gate = expit(model['presence_calibration_slope']*(hidden@model['presence_weights1']+
                  model['presence_bias1'])+model['presence_calibration_bias'])
    return probability, gate


def decide(frame, p, gate, presence_threshold, species_threshold):
    eligible = frame.quality_ok.to_numpy() & (gate>=presence_threshold) & (p.max(axis=1)>=species_threshold)
    candidates = np.where(eligible, p.argmax(axis=1), -1)
    confirmed = np.full(len(frame), -1, dtype='int32')
    histories = {}
    # Input may be interleaved; history is per excerpt, never across recordings.
    for i, row in enumerate(frame.itertuples()):
        history, previous = histories.get(row.path, ([], None))
        if previous is None or abs(row.start_s-previous-row.step_s)>1e-5 or not row.quality_ok:
            history = []
        if row.quality_ok:
            history = (history+[int(candidates[i])])[-3:]
            if len(history)==3 and candidates[i]>=0 and history.count(int(candidates[i]))>=2:
                confirmed[i] = candidates[i]
        histories[row.path] = (history, row.start_s)
    return candidates, confirmed


def evaluate(frame, x, models, classes):
    rows, predicted, per_class = [], [], []
    target = frame.label.map({name:i for i,name in enumerate(classes)}).fillna(-1).to_numpy().astype(int)
    for name, model in models.items():
        p, gate = infer_portable(model, x)
        candidate, confirmed = decide(frame, p, gate, model['presence_threshold'], model['confidence_threshold'])
        out = frame[['path','group','domain','label','split','start_s','id']].copy()
        out['model'], out['forced'], out['presence_score'], out['class_score'] = name, p.argmax(axis=1), gate, p.max(axis=1)
        out['candidate'], out['confirmed'] = candidate, confirmed
        predicted.append(out)
        for domain in ['dryad','humbug']:
            mask = frame.domain.eq(domain).to_numpy()
            positive = mask & (target>=0)
            noise = mask & frame.label.eq('noise').to_numpy()
            unknown = mask & frame.label.eq('unknown_mosquito').to_numpy()
            emitted = confirmed>=0
            correct = confirmed==target
            source_frame = pd.DataFrame(p[positive], index=frame.loc[positive,'group']).groupby(level=0).mean()
            source_truth = frame.loc[positive].groupby('group').label.first().map({c:i for i,c in enumerate(classes)})
            forced = p.argmax(axis=1)
            row = dict(model=name,domain=domain,known_windows=int(positive.sum()),
                known_groups=int(frame.loc[positive,'group'].nunique()),
                forced_window_accuracy=float(accuracy_score(target[positive],forced[positive])) if positive.any() else None,
                forced_balanced_accuracy=float(balanced_accuracy_score(target[positive],forced[positive])) if positive.any() else None,
                forced_recording_accuracy=float(accuracy_score(source_truth.reindex(source_frame.index),source_frame.to_numpy().argmax(axis=1))) if len(source_frame) else None,
                accepted_known_windows=int((emitted&positive).sum()),
                known_coverage=float(emitted[positive].mean()) if positive.any() else None,
                accepted_known_accuracy=float(correct[emitted&positive].mean()) if (emitted&positive).any() else None,
                correct_known_fraction=float((correct&emitted)[positive].mean()) if positive.any() else None,
                noise_windows=int(noise.sum()),noise_false_species_rate=float(emitted[noise].mean()) if noise.any() else None,
                unknown_windows=int(unknown.sum()),unresolved_species_emission_rate=float(emitted[unknown].mean()) if unknown.any() else None,
                presence_known_recall=float((gate[positive]>=model['presence_threshold']).mean()) if positive.any() else None,
                presence_noise_false_positive_rate=float((gate[noise]>=model['presence_threshold']).mean()) if noise.any() else None)
            rows.append(row)
            for index, species in enumerate(classes):
                truth = mask & (target==index)
                if not truth.any():
                    continue
                per_class.append(dict(model=name,domain=domain,species=species,
                    windows=int(truth.sum()),groups=int(frame.loc[truth,'group'].nunique()),
                    forced_recall=float((forced[truth]==index).mean()),
                    coverage=float(emitted[truth].mean()),
                    correctly_identified_fraction=float((emitted&correct)[truth].mean()),
                    accepted_accuracy=float(correct[truth&emitted].mean()) if (truth&emitted).any() else None))
    pd.concat(predicted,ignore_index=True).to_csv(OUT/'test_predictions.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(per_class).to_csv(OUT/'per_class.csv',index=False)
    return result


def main():
    started = time.time()
    frame, x = assemble()
    classes = np.load(ROOT/'results/arduino_species/models/species_fold2.npz')['classes'].tolist()
    y = frame.label.map({name:i for i,name in enumerate(classes)}).fillna(-1).to_numpy().astype(int)
    def indices(split, known=False):
        mask = frame.split.eq(split).to_numpy() & frame.quality_ok.to_numpy()
        if known: mask &= y>=0
        return np.flatnonzero(mask)
    train, val, cal = [indices(split,True) for split in ['train','validation','calibration']]
    assert len(np.unique(y[train])) == 20
    (OUT/'models').mkdir(exist_ok=True)
    choices = []
    for hidden in [(64,), (128,64)]:
        print('Training expanded species MLP', hidden, len(train), 'windows', flush=True)
        model = TinySpeciesMLP(hidden=hidden, seed=SEED)
        model.fit(x[train], y[train], source_weights(frame.iloc[train], y[train]),
                  validation=(x[val],y[val],source_weights(frame.iloc[val],y[val])))
        ww, bb = fused_layers(model)
        bundle = dict(model=model,weights=ww,biases=bb,classes=classes,temperature=1.,
                      model_name='Expanded MLP '+str(hidden),neural_epoch=model.best_epoch)
        guessed = model.decision_function(x[val]).argmax(axis=1)
        bundle['validation_accuracy'] = float(np.average(guessed==y[val],weights=source_weights(frame.iloc[val],y[val])))
        choices.append(bundle)
        print('Validation', hidden, bundle['validation_accuracy'], 'epoch', model.best_epoch, flush=True)
    bundle = max(choices, key=lambda item:item['validation_accuracy'])
    # Existing conservative calibration rule; test labels never select threshold.
    species_export.calibrate(bundle,x[cal],y[cal],source_weights(frame.iloc[cal],y[cal]),frame.iloc[cal].group.to_numpy())
    atomic_json(OUT/'species_selection.json',dict(candidates=[dict(name=b['model_name'],
        validation_accuracy=b['validation_accuracy'],epoch=b['neural_epoch']) for b in choices],
        selected=bundle['model_name'],calibration=bundle['calibration']))
    binary = frame.label.ne('noise').to_numpy().astype(int)
    bt,bv,bc = [indices(split) for split in ['train','validation','calibration']]
    print('Training expanded presence MLP',len(bt),'windows',flush=True)
    gate = TinyPresenceMLP().fit(x[bt],binary[bt],source_weights(frame.iloc[bt],binary[bt]),
        validation=(x[bv],binary[bv],source_weights(frame.iloc[bv],binary[bv])))
    calibrator = LogisticRegression(C=10,max_iter=2000).fit(gate.decision_function(x[bc])[:,None],binary[bc],
                sample_weight=source_weights(frame.iloc[bc],binary[bc]))
    presence = dict(model=gate,model_name='Expanded presence MLP 68-16-1',
        calibration_slope=float(calibrator.coef_[0,0]),calibration_bias=float(calibrator.intercept_[0]))
    score = gate_scores(presence,x[bc]); negative = binary[bc]==0
    weight = source_weights(frame.iloc[bc],binary[bc])
    thresholds = np.r_[np.linspace(.01,.99,197),1.0001]
    for key, limit in [('threshold_05',.05),('threshold_10',.1)]:
        for threshold in thresholds:
            if np.average(score[negative]>=threshold,weights=weight[negative])<=limit:
                presence[key] = float(np.nextafter(np.float32(threshold),np.float32(np.inf)))
                break
    print('Calibrated thresholds',presence['threshold_10'],bundle['confidence_threshold'],flush=True)
    joblib.dump(dict(species=bundle,presence=presence),OUT/'models/expanded.joblib',compress=3)
    # User requires evidence of reliability before any Arduino compilation or
    # packaging. Save numeric weights for offline evaluation only; no headers,
    # sketch or device binaries are produced by this training command.
    coef = gate.w1/gate.scaler.scale_[None]
    expanded = dict(classes=np.array(classes),temperature=np.float32(bundle['temperature']),
        confidence_threshold=np.float32(bundle['confidence_threshold']),fold=np.int32(-1),
        presence_threshold=np.float32(presence['threshold_10']),
        presence_weights0=coef.astype('float32'),
        presence_bias0=(gate.b1-coef@gate.scaler.mean_).astype('float32'),
        presence_weights1=gate.w2.astype('float32'),presence_bias1=np.float32(gate.b2),
        presence_calibration_slope=np.float32(presence['calibration_slope']),
        presence_calibration_bias=np.float32(presence['calibration_bias']))
    for layer,(ww,bb) in enumerate(zip(bundle['weights'],bundle['biases'])):
        expanded[f'weights{layer}'],expanded[f'bias{layer}']=ww,bb
    np.savez_compressed(OUT/'models/species_fold0.npz',**expanded)
    checkpoint = OUT/'models/species_fold0.npz'
    # Starting a new cycle invalidates previous fresh-test reports as proof for
    # that cycle, even if deterministic training reproduces the same weights.
    from datetime import datetime, timezone
    atomic_json(OUT/'reliability_decision.json',dict(status='PENDING_NEW_POST_TRAINING_AUDIO',
        arduino_compile_allowed=False,model_sha256=file_hash(checkpoint),
        frozen_utc=datetime.fromtimestamp(checkpoint.stat().st_mtime,timezone.utc).isoformat(),
        note='Obtain a genuinely new recording after this freeze; never reuse a previous fresh test as new.'))
    export_info = dict(numeric_checkpoint_only=True,arduino_headers_generated=False,
        arduino_compiled=False,dimensions=[68]+[w.shape[0] for w in bundle['weights']])
    with np.load(OUT/'models/species_fold0.npz') as z:
        expanded = {key:z[key].copy() for key in z.files}
    portable_p,portable_g = infer_portable(expanded,x[cal])
    assert np.allclose(portable_p,probabilities(bundle,x[cal]),atol=1e-5)
    assert np.allclose(portable_g,gate_scores(presence,x[cal]),atol=1e-4)
    with np.load(ROOT/'results/arduino_species/models/species_fold2.npz') as z:
        baseline = {key:z[key].copy() for key in z.files}
    test = np.flatnonzero(frame.split.eq('test'))
    comparison = evaluate(frame.iloc[test].reset_index(drop=True),x[test],
                          dict(baseline=baseline,expanded=expanded),classes)
    print(comparison.to_string(index=False),flush=True)
    # In-sample adaptation outcomes are reported separately, never as validation.
    adapted = np.flatnonzero(frame.id.isin([220292,222024]))
    adaptation = frame.iloc[adapted].reset_index(drop=True)
    p,g = infer_portable(expanded,x[adapted]);_,confirmed = decide(adaptation,p,g,
        expanded['presence_threshold'],expanded['confidence_threshold'])
    adaptation['forced_species'] = [classes[i] for i in p.argmax(axis=1)]
    adaptation['identified_species'] = [classes[i] if i>=0 else '-' for i in confirmed]
    adaptation.to_csv(OUT/'adaptation_examples.csv',index=False)
    atomic_json(OUT/'training_audit.json',dict(seconds=time.time()-started,export=export_info,
        presence_epoch=gate.best_epoch,presence_threshold=float(expanded['presence_threshold']),
        species_threshold=float(expanded['confidence_threshold']),
        species_checkpoint_sha256=file_hash(OUT/'models/species_fold0.npz'),
        physical_board_tested=False,baseline_replaced=False,arduino_compiled=False,
        portable_python_equivalence=True,
        frozen_utc=datetime.fromtimestamp(checkpoint.stat().st_mtime,timezone.utc).isoformat(),
        note='Test is held out of this retraining; historical Dryad test was already analysed in earlier work.'))


if __name__ == '__main__':
    main()
