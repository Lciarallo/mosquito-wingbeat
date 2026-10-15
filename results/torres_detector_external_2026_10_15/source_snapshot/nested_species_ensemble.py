"""Group-separated fit/calibration/test experiment for twenty mosquito species.

Three base models (SVM, ExtraTrees, HistGradientBoosting) share the same fit data.
The ensemble is calibrated on reserved groups within each outer training fold.
Base models are not refitted on calibration groups before predicting outer test.
This is a new three-model experiment, not a repetition of the old seven-model
ensemble. Outputs and weights are kept separate from historical artifacts.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from improve_device import (ROOT, METHODS, model_for, fit_classical, predict_proba,
    split_calibration, weights, file_hash, atomic_json, load_legacy_frontend,
    EXTRA_NAMES, improved_wave_features)

SEED = 20261009
NAMES = ['svm', 'extra_trees', 'hist_gradient_boosting', 'fixed', 'calibrated']


def now():
    return datetime.now(timezone.utc).isoformat()


def check_partition(frame, fit, calibration, test):
    """Fail on row, recording, path or content overlap and missing fit classes."""
    parts = [np.asarray(x, dtype=int) for x in (fit, calibration, test)]
    if any(len(p) != len(np.unique(p)) for p in parts):
        raise ValueError('Duplicate row in partition.')
    for a in range(3):
        for b in range(a+1, 3):
            if set(parts[a]) & set(parts[b]):
                raise ValueError('Row leakage.')
            for key in ['group', 'path', 'content_sha256']:
                if set(frame.iloc[parts[a]][key]) & set(frame.iloc[parts[b]][key]):
                    raise ValueError(f'{key} leakage.')
    expected = set(frame.iloc[np.concatenate(parts)].label)
    if set(frame.iloc[parts[0]].label) != expected:
        raise ValueError('Fit data are missing a target class.')
    return {name: dict(rows=len(p), groups=int(frame.iloc[p].group.nunique()))
            for name, p in zip(['fit', 'calibration', 'test'], parts)}


def validate_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 3 or min(p.shape) < 1 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected finite nonnegative model/row/class probabilities.')
    if not np.allclose(p.sum(2), 1, atol=1e-5):
        raise ValueError('Probability vectors must sum to one.')
    return p


def blend(p, parameters):
    p = validate_probabilities(p)
    w = np.asarray(parameters['weights'])
    prior = np.asarray(parameters['log_prior'])
    if w.shape != (p.shape[0],) or prior.shape != (p.shape[2],):
        raise ValueError('Parameters do not match probability dimensions.')
    if not np.isfinite(w).all() or not np.isfinite(prior).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('Invalid ensemble parameters.')
    mixture = np.einsum('m,mnc->nc', w, p)
    return softmax(np.log(np.maximum(mixture, 1e-12)) + prior, axis=1)


def fit_blend(p, y, sample_weight, *, partition):
    if partition != 'calibration':
        raise ValueError('Ensemble fitting requires a calibration partition.')
    p = validate_probabilities(p)
    y = np.asarray(y, dtype=int)
    sample_weight = np.asarray(sample_weight, dtype=float)
    m, n, c = p.shape
    if y.shape != (n,) or sample_weight.shape != (n,) or (y < 0).any() or (y >= c).any():
        raise ValueError('Labels/weights are not aligned.')
    if not np.isfinite(sample_weight).all() or (sample_weight <= 0).any():
        raise ValueError('Positive finite sample weights required.')
    sample_weight = sample_weight / sample_weight.sum()

    def loss_gradient(z):
        w = softmax(z[:m])
        prior = z[m:] - z[m:].mean()
        mixture = np.maximum(np.einsum('m,mnc->nc', w, p), 1e-12)
        q = softmax(np.log(mixture) + prior, axis=1)
        loss = -np.sum(sample_weight * np.log(np.maximum(q[np.arange(n), y], 1e-12)))
        loss += .01 * np.mean(prior**2)
        residual = q.copy()
        residual[np.arange(n), y] -= 1
        residual *= sample_weight[:, None]
        gw = np.einsum('nc,mnc->m', residual/mixture, p)
        gz = w * (gw - np.dot(gw, w))
        gp = residual.sum(0) + .02 * prior/c
        gp -= gp.mean()
        return loss, np.r_[gz, gp]

    result = minimize(loss_gradient, np.zeros(m+c), jac=True, method='L-BFGS-B',
                      bounds=[(-6, 6)]*m + [(-3, 3)]*c,
                      options={'maxiter': 250, 'ftol': 1e-10})
    if not result.success:
        raise RuntimeError(f'Calibration optimiser failed: {result.message}')
    prior = result.x[m:] - result.x[m:].mean()
    return dict(weights=softmax(result.x[:m]).tolist(), log_prior=prior.tolist(),
                loss=float(result.fun), iterations=int(result.nit), converged=True,
                partition='calibration', geographic_prior=False)


def recording_probabilities(frame, p):
    if len(frame) != len(p) or frame.groupby('group').label.nunique().max() != 1:
        raise ValueError('Recording labels/probabilities are inconsistent.')
    probabilities = pd.DataFrame(p, index=frame.group).groupby(level=0).mean()
    labels = frame.groupby('group').label.first().reindex(probabilities.index)
    return probabilities.index.to_numpy(), labels.to_numpy(), probabilities.to_numpy()


def select_threshold(frame, p, classes, *, partition):
    if partition != 'calibration':
        raise ValueError('Threshold selection requires calibration groups.')
    _, labels, rec = recording_probabilities(frame, p)
    correct = np.asarray(classes)[rec.argmax(1)] == labels
    scores = rec.max(1)
    # A developmental target, not a reliability certificate. At least 20
    # recording groups must be emitted; otherwise abstain on every recording.
    for threshold in np.r_[0., np.sort(np.unique(scores))]:
        accepted = scores >= threshold
        if accepted.sum() >= 20 and correct[accepted].mean() >= .90:
            return dict(threshold=float(threshold), calibration_emitted=int(accepted.sum()),
                        calibration_accuracy=float(correct[accepted].mean()),
                        calibration_groups=len(labels), target=.90, minimum_emissions=20)
    return dict(threshold=1.000001, calibration_emitted=0, calibration_accuracy=None,
                calibration_groups=len(labels), target=.90, minimum_emissions=20)


def metric_rows(frame, p, classes, model, fold, threshold):
    groups, truth, rec = recording_probabilities(frame, p)
    rows, species = [], []
    for unit, y, scores in [('window', frame.label.to_numpy(), p), ('recording', truth, rec)]:
        predicted = classes[scores.argmax(1)]
        cm = confusion_matrix(y, predicted, labels=classes)
        row = dict(model=model, fold=fold, unit=unit, n=len(y), groups=len(groups),
                   accuracy=float(accuracy_score(y, predicted)),
                   balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                   macro_f1=float(f1_score(y, predicted, labels=classes, average='macro', zero_division=0)))
        if unit == 'recording':
            accepted = rec.max(1) >= threshold
            row.update(coverage=float(accepted.mean()), emitted=int(accepted.sum()),
                       accepted_accuracy=float(np.mean(predicted[accepted]==y[accepted])) if accepted.any() else None,
                       correct_emissions_over_all=float(np.mean(accepted & (predicted==y))))
        rows.append(row)
        for i, sp in enumerate(classes):
            n, predicted_count, tp = int(cm[i].sum()), int(cm[:, i].sum()), int(cm[i, i])
            species.append(dict(model=model, fold=fold, unit=unit, species=str(sp),
                                support=n, correct=tp, predicted=predicted_count,
                                recall=tp/n if n else None,
                                precision=tp/predicted_count if predicted_count else None,
                                f1=2*tp/(n+predicted_count) if n+predicted_count else None))
    return rows, species


def load_inputs():
    frame = pd.read_csv(ROOT/'results/segments.csv')
    with np.load(ROOT/'data/cache/features.npz') as cached:
        base = cached['features'].copy()
    extended = np.c_[base, np.load(ROOT/'data/cache/improved_features.npy')].astype('float32')
    if len(frame) != len(base) or not np.isfinite(extended).all():
        raise ValueError('Cached features not aligned or finite.')
    _, names = load_legacy_frontend()
    if extended.shape[1] != len(names)+len(EXTRA_NAMES):
        raise ValueError('Feature dimensions differ from frontend.')
    return frame, base, extended, names+EXTRA_NAMES


def train(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('Choose a new experiment directory; protocol already exists.')
    frame, base, extended, feature_names = load_inputs()
    ix = np.flatnonzero(frame.label.ne('noise') & frame.fold_species.ge(0))
    classes = np.sort(frame.iloc[ix].label.unique())
    labels = np.full(len(frame), -1)
    labels[ix] = np.searchsorted(classes, frame.iloc[ix].label)
    inputs = ['results/segments.csv', 'data/cache/features.npz', 'data/cache/improved_features.npy',
              'data/cache/features_manifest.json', 'data/cache/improved_features.json',
              'nested_species_ensemble.py', 'improve_device.py', 'notebook_code.py']
    hashes = {p:file_hash(ROOT/p) for p in inputs}
    protocol = dict(created_utc=now(), seed=SEED, classes=classes.tolist(),
        members=NAMES[:3], outer_folds=[0,1,2], calibration_rule='20% recording groups within each species, ceil, at least one; seed 20261009 + fold',
        calibration_target_recording_accuracy=.90, minimum_calibration_emissions=20,
        primary_model='calibrated', fixed_weights=[1/3]*3, log_prior_regularisation=.01,
        same_fit_groups_for_all_members=True, refit_on_calibration=False,
        input_sha256=hashes, feature_names=feature_names,
        external_inclusion='After final freeze: first unused numeric BioDCASE D3 IDs, 4 per each of 9 exact species; real duration >=1.0 s, no padding/repetition; exclude known file names/byte hashes/PCM fragments before inference. Short files stay documented, never replaced based on predictions.',
        external_limitations='New files in familiar corpus; individual/session ancestry may be unavailable. Not a newly acquired campaign.',
        species_only_no_presence_gate=True, internal_scope='Retrospective Dryad closed-set development evaluation; recording labels are weak event labels.',
        geographic_prior=False, arduino_compiled=False)
    atomic_json(out/'protocol.json', protocol)
    features = [base, extended, extended]
    all_probs = {name:np.zeros((len(ix),len(classes)), dtype='float32') for name in NAMES}
    partition_rows, audits, metric_records, species_records = [], [], [], []
    rec_frames = []
    (out/'models').mkdir()
    # Final fit/calibration split is declared now, before any test result.
    runs = [(f, ix[frame.iloc[ix].fold_species.to_numpy()!=f], ix[frame.iloc[ix].fold_species.to_numpy()==f]) for f in range(3)]
    runs.append(('final', ix, np.array([], dtype=int)))
    for fold, outer_train, test in runs:
        fit, cal = split_calibration(outer_train, frame, SEED+(fold if isinstance(fold,int) else 100))
        counts = check_partition(frame, fit, cal, test)
        audits.append(dict(fold=fold, **counts, overlap_groups_paths_hashes=0))
        for part, rows in [('fit',fit),('calibration',cal),('test',test)]:
            partition_rows.extend(dict(fold=fold, partition=part, row_index=int(i), group=frame.iloc[i].group,
                label=frame.iloc[i].label, path=frame.iloc[i].path, content_sha256=frame.iloc[i].content_sha256) for i in rows)
        pd.DataFrame(partition_rows).to_csv(out/'partitions.csv', index=False)
        calibration_probs, test_probs, models = [], [], []
        for k, name in enumerate(NAMES[:3]):
            print(f'FOLD {fold} {name}: {len(fit)} fit / {len(cal)} calibration / {len(test)} test',flush=True)
            start = time.monotonic()
            model = fit_classical(model_for(METHODS[k]), features[k][fit], labels[fit], weights(fit,labels,frame))
            # Every scaler and base estimator sees fit groups only.
            calibration_probs.append(predict_proba(model, features[k][cal]))
            if len(test): test_probs.append(predict_proba(model, features[k][test]))
            models.append(model)
            joblib.dump(dict(model=model, classes=classes.tolist(), fit_rows=fit, calibration_rows=cal,
                             test_rows=test, member=name, seconds=time.monotonic()-start),
                        out/'models'/f'{fold}_{name}.joblib', compress=3)
            print(f'  finished {time.monotonic()-start:.1f}s',flush=True)
        # Equal recording contribution; partly balance classes, determined on calibration only.
        sw = .5*weights(cal,labels,frame,False)+.5*weights(cal,labels,frame,True)
        parameters = fit_blend(np.stack(calibration_probs), labels[cal], sw, partition='calibration')
        cal_scores = [*calibration_probs, np.mean(calibration_probs,axis=0), blend(np.stack(calibration_probs),parameters)]
        thresholds = {name:select_threshold(frame.iloc[cal],p,classes,partition='calibration')
                      for name,p in zip(NAMES,cal_scores)}
        atomic_json(out/f'calibration_{fold}.json',dict(parameters=parameters, thresholds=thresholds, partitions=counts))
        if len(test):
            scores = [*test_probs, np.mean(test_probs,axis=0), blend(np.stack(test_probs),parameters)]
            positions = np.searchsorted(ix,test)
            for name, p in zip(NAMES,scores):
                all_probs[name][positions] = p
                rows, species = metric_rows(frame.iloc[test],p,classes,name,fold,thresholds[name]['threshold'])
                metric_records.extend(rows);species_records.extend(species)
                groups, true_rec, rec = recording_probabilities(frame.iloc[test],p)
                rec_frames.append(pd.DataFrame(dict(model=name,fold=fold,group=groups,true_species=true_rec,
                    predicted_species=classes[rec.argmax(1)], score=rec.max(1),
                    accepted=rec.max(1)>=thresholds[name]['threshold'])))
            pd.DataFrame(metric_records).to_csv(out/'fold_metrics.csv',index=False)
            print(pd.DataFrame(rows)[['model','unit','accuracy','balanced_accuracy']].to_string(index=False),flush=True)
        else:
            joblib.dump(dict(models=models,classes=classes.tolist(),parameters=parameters,thresholds=thresholds,
                             feature_names=feature_names,training_inputs_sha256=hashes), out/'models/final_bundle.joblib',compress=3)
    rec_predictions = pd.concat(rec_frames, ignore_index=True)
    rec_predictions.to_csv(out/'recording_predictions.csv',index=False)
    summary=[]
    for name,p in all_probs.items():
        rows,species = metric_rows(frame.iloc[ix],p,classes,name,'pooled',0)
        r=rec_predictions[rec_predictions.model.eq(name)]
        a=r.accepted.to_numpy();correct=r.true_species.eq(r.predicted_species).to_numpy()
        rows[1].update(coverage=float(a.mean()),emitted=int(a.sum()),
                       accepted_accuracy=float(correct[a].mean()) if a.any() else None,
                       correct_emissions_over_all=float((a&correct).mean()))
        summary.extend(rows);species_records.extend(species)
    pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(species_records).to_csv(out/'per_species.csv',index=False)
    np.savez_compressed(out/'oof_probabilities.npz',**all_probs,classes=classes,row_indices=ix,
                        groups=frame.iloc[ix].group.to_numpy().astype(str),folds=frame.iloc[ix].fold_species.to_numpy())
    for p,digest in hashes.items():
        if file_hash(ROOT/p)!=digest: raise RuntimeError(f'Input mutated: {p}')
    atomic_json(out/'partition_audit.json',dict(runs=audits,all_pairwise_disjoint=True,
        no_refit_on_calibration=True,biological_independence_not_proven=True))
    frozen={str(p.relative_to(ROOT)):file_hash(p) for p in sorted((out/'models').glob('*.joblib'))}
    frozen.update({str(p.relative_to(ROOT)):file_hash(p) for p in out.glob('calibration_*.json')})
    atomic_json(out/'freeze.json',dict(frozen_utc=now(),artifacts_sha256=frozen,
        inference_code_sha256=file_hash(Path(__file__)), primary_model='calibrated',
        no_external_selection_yet=True,release_allowed=False))
    print('FROZEN',json.dumps(pd.DataFrame(summary).to_dict('records')),flush=True)


def audio_probabilities(path,bundle):
    import soundfile as sf
    from scipy.signal import resample_poly
    wave,sr=sf.read(path,dtype='float32',always_2d=True)
    wave=wave.mean(1)
    if sr != 16000:
        d=np.gcd(sr,16000);wave=resample_poly(wave,16000//d,sr//d)
    n=len(wave)//16000
    if n<1: raise ValueError('No real one-second window; padding/repetition forbidden.')
    windows=wave[:n*16000].reshape(n,16000)
    frontend,_=load_legacy_frontend()
    x=np.stack([improved_wave_features(w,frontend) for w in windows])
    _,base_names=load_legacy_frontend()
    components=np.stack([predict_proba(model,x[:,:len(base_names)] if k==0 else x)
                         for k,model in enumerate(bundle['models'])])
    outputs={**{name:p for name,p in zip(NAMES[:3],components)},
             'fixed':components.mean(0),'calibrated':blend(components,bundle['parameters'])}
    quality=(np.abs(windows).max(1)>=32/32768)&((np.abs(windows)>=32700/32768).mean(1)<=.01)
    return outputs,quality,dict(duration_s=len(wave)/16000,complete_windows=n,original_sample_rate=sr,
                                unanalysed_tail_s=(len(wave)%16000)/16000,no_padding_or_repetition=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/species_nested_2026_10_09')
    args=parser.parse_args()
    train(args.output.resolve())


if __name__=='__main__':main()
