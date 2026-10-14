"""Audited notebook transfer: inductive features, nested PCM blocks, frozen evaluation.

No historical source/result is modified. All external files are retrospective.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import minimize_scalar
from scipy.signal import resample_poly
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from notebook_transfer_features import SR, ROOT, REFERENCE, NAMES, LFCC_NAMES, extract_features, pool_features, hybrid_window_features
from research_metrics_v2 import classification_metrics

OUT = ROOT / 'results/notebook_transfer_2026_10_14_2030'
V4 = ROOT / 'results/species_nested_v4_2026_10_14'
SEED = 20261014
SOURCES = ['notebook_transfer_features.py', 'reproduce_notebook_transfer.py', 'research_metrics_v2.py', 'acoustic_features_v4.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def guarded_output(out):
    out = Path(out).resolve()
    if out == V4 or not out.is_relative_to(ROOT / 'results'):
        raise ValueError('New experiment output must be inside results and outside historical results.')
    if (out / 'protocol.json').exists():
        raise FileExistsError('Never overwrite an existing experiment.')
    return out


def check_groups(records, fit, calibration, test):
    for column in ['group', 'pcm_block']:
        sets = [set(records.iloc[ix][column]) for ix in [fit, calibration, test]]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError('Partition leakage: ' + column)
    return {'fit': len(fit), 'calibration': len(calibration), 'test': len(test), 'group_overlap': 0, 'pcm_block_overlap': 0}


def _feature_chunk(waves):
    with threadpool_limits(limits=1):
        return extract_features(waves)


def prepare_features(frame, out, workers):
    waves = np.load(ROOT / 'data/cache/waves.npy', mmap_mode='r')
    if waves.shape != (len(frame), SR):
        raise ValueError('PCM cache row alignment/length differs.')
    reference = np.empty((len(frame), 61), np.float32)
    lfcc = np.empty((len(frame), 26), np.float32)
    chunks = (np.array(waves[i:i+128]) for i in range(0, len(waves), 128))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = 0
        for a, b in pool.map(_feature_chunk, chunks, chunksize=1, buffersize=8):
            count = len(a); reference[done:done+count] = a; lfcc[done:done+count] = b; done += count
            if done % 2048 == 0 or done == len(frame):
                print('FEATURES', done, '/', len(frame), flush=True)
    z = np.load(V4 / 'features_v4.npz')
    if not np.array_equal(z['row_indices'], np.arange(len(frame))) or not np.array_equal(z['pcm_blocks'], frame.pcm_block.to_numpy(str)):
        raise ValueError('Frozen v4 feature identity differs.')
    variants = {'notebook61': reference, 'v4_308': z['features'], 'v4_lfcc334': np.c_[z['features'], lfcc]}
    records = []; bags = {k: [] for k in variants}
    for group, sub in frame.groupby('group', sort=False):
        if sub.label.nunique() != 1 or sub.pcm_block.nunique() != 1:
            raise ValueError('Recording labels/PCM blocks not single-valued.')
        indices = sub.index.to_numpy()
        row = sub.iloc[0]
        records.append(dict(group=group, pcm_block=row.pcm_block, label=row.label, path=row.path, windows=len(sub)))
        for key, x in variants.items():
            bags[key].append(pool_features(x[indices], z['salience'][indices], 1.))
    records = pd.DataFrame(records)
    records.to_csv(out / 'recordings.csv', index=False)
    np.savez_compressed(out / 'feature_bags.npz', **{k: np.array(v) for k, v in bags.items()})
    np.savez_compressed(out / 'new_window_features.npz', notebook61=reference, relative_lfcc=lfcc)
    return records, {k: np.array(v) for k, v in bags.items()}


def make_model(family):
    if family == 'lr':
        model = LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED)
        grid = {'model__C': [.01, .1, 1., 10.]}
    elif family == 'tree':
        model = DecisionTreeClassifier(class_weight='balanced', random_state=SEED)
        grid = {'model__max_depth': [3, 6, None], 'model__min_samples_split': [2, 10]}
    elif family == 'rf':
        model = RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1)
        grid = {'model__n_estimators': [100, 200], 'model__max_depth': [6, None]}
    elif family == 'knn':
        model = KNeighborsClassifier(n_jobs=1)
        grid = {'model__n_neighbors': [3, 7, 15], 'model__weights': ['uniform', 'distance']}
    else:
        raise ValueError('Unknown model family.')
    return Pipeline([('scale', StandardScaler()), ('model', model)]), grid


def align_probabilities(model, x, classes):
    """Identical numerical prediction for a row alone and in a changed batch."""
    ix = np.searchsorted(classes, model.classes_)
    rows = []
    with threadpool_limits(limits=1):
        for row in np.asarray(x):
            p = np.zeros(len(classes)); p[ix] = model.predict_proba(row[None])[0]; rows.append(p)
    return np.array(rows)


def apply_temperature(p, temperature):
    return np.vstack([softmax(np.log(np.maximum(row, 1e-12)) / temperature) for row in p])


def calibrate(p, labels, classes):
    target = np.searchsorted(classes, labels)
    counts = pd.Series(labels).value_counts()
    weights = np.array([1. / counts[v] for v in labels]); weights /= weights.sum()
    def loss(log_t):
        q = apply_temperature(p, np.exp(log_t))
        return float(np.sum(weights * -np.log(np.maximum(q[np.arange(len(q)), target], 1e-12))))
    fit = minimize_scalar(loss, bounds=(np.log(.25), np.log(4.)), method='bounded')
    return float(np.exp(fit.x))


def select_rejection(y, p, classes):
    predicted = classes[p.argmax(1)]; confidence = p.max(1)
    floor = .55 if len(classes) == 2 else 1./len(classes) + .05
    options = sorted(set([floor, 1.] + confidence[confidence >= floor].tolist()))
    eligible = []
    for threshold in options:
        mask = confidence >= threshold
        if mask.sum() >= 20 and np.mean(predicted[mask] == y[mask]) >= .90:
            eligible.append((int(mask.sum()), -threshold, threshold))
    return max(eligible)[2] if eligible else 1.0000001


def select_aegypti_cutoff(y, score):
    positive = np.asarray(y) == 'Aedes aegypti'; negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError('Binary calibration must include both classes.')
    candidates = np.unique(np.r_[.5, score[score >= .5], 1.0000001])
    curve = []
    for t in candidates:
        recall = float(np.mean(score[positive] >= t)); fpr = float(np.mean(score[negative] >= t))
        if fpr <= .05:
            curve.append((recall, -fpr, -float(t), float(t)))
    return max(curve)[3]


def combinations():
    result = [('binary', 'notebook61', f) for f in ['lr', 'tree', 'rf', 'knn']]
    result += [('binary', v, 'lr') for v in ['v4_308', 'v4_lfcc334']]
    result += [('species', v, f) for v in ['notebook61', 'v4_308', 'v4_lfcc334'] for f in ['lr', 'rf']]
    return result


def train(out=OUT, workers=4, jobs=2):
    guarded_output(out); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(V4 / 'segments_v4.csv'); frame.index = np.arange(len(frame))
    parts = pd.read_csv(V4 / 'partitions.csv', dtype={'fold': str})
    source = json.loads((ROOT / 'external/notebook_reference_2026_10_14/source_manifest.json').read_text())
    protocol = dict(created_utc=now(), source=source, task_scope='Binary Ae. aegypti vs other species/noise; secondary 20 species',
        reference_results_not_reproduced_verbatim=True, source_windows_dataset_unavailable=True,
        source_feature_function_exact=True, modifications=['16 kHz aligned PCM', 'peak normalisation and causal bandpass reset within each isolated window',
        'no original leaky window train/test model fitted', 'arithmetic feature-bag mean; one training row per recording',
        'all four notebook classifiers; reduced documented grid', '3-fold PCM-group CV with StandardScaler inside Pipeline',
        'temperature and rejection only on internal calibration', 'v4 vs v4 plus 26 floor-relative LFCC controlled ablation'],
        variants=['notebook61', 'v4_308', 'v4_lfcc334'], primary_metric='OOF recording macro-F1',
        prediction_inductive_by_window=True, pooling='fixed arithmetic mean within recording, no other test recordings',
        outer_partitions='Exact frozen v4 fit/calibration/test membership, 560 PCM blocks',
        inner_cv_folds=3, scoring='f1_macro', combinations=combinations(), feature_names={'notebook61': NAMES, 'additional_relative_lfcc': LFCC_NAMES},
        external_retrospective=True, external_plan_sha256=digest(V4/'external_plan.json'), private_label='user-provided, no independent biological confirmation',
        model_release_allowed=False, arduino_compiled=False,
        inputs_sha256={str(p.relative_to(ROOT)): digest(p) for p in [REFERENCE, V4/'segments_v4.csv', V4/'partitions.csv', V4/'features_v4.npz', ROOT/'data/cache/waves.npy']},
        code_sha256={p: digest(ROOT/p) for p in SOURCES})
    save(out / 'protocol.json', protocol)
    # Audit, but do not fit, the received notebook's random-window split on an analogous 1200-window local subset.
    y = frame.label.eq('Aedes aegypti').to_numpy()
    subset = np.r_[np.flatnonzero(y)[:600], np.flatnonzero(~y)[:600]]
    a, b = train_test_split(subset, test_size=.3, random_state=42, stratify=y[subset])
    overlap = set(frame.iloc[a].group) & set(frame.iloc[b].group)
    save(out / 'reference_split_audit.json', dict(reference_windows=1200, source_original_dataset='unavailable',
         local_subset_windows=len(subset), local_positive_files=int(frame.iloc[subset][y[subset]].group.nunique()),
         local_negative_files=int(frame.iloc[subset][~y[subset]].group.nunique()), recording_group_overlap=len(overlap),
         pcm_block_overlap=len(set(frame.iloc[a].pcm_block)&set(frame.iloc[b].pcm_block)),
         test_windows_from_fit_recordings=int(frame.iloc[b].group.isin(overlap).sum()), test_windows=len(b),
         trained_on_leaky_partition=False))
    records, x = prepare_features(frame, out, workers)
    species = np.sort(records[records.label.ne('noise')].label.unique())
    audits = []; plans = {}; calibration_scores = []
    (out/'models').mkdir(exist_ok=False)
    for fold in ['0', '1', '2', 'final']:
        sub = parts[parts.fold.eq(fold)]; partition = []
        for role in ['fit', 'calibration', 'test']:
            groups = set(sub[sub.partition.eq(role)].original_group)
            partition.append(np.flatnonzero(records.group.isin(groups)))
        fit, cal, test = partition
        audits.append(dict(fold=fold, **check_groups(records, fit, cal, test)))
        plans[fold] = dict(fit=fit.tolist(), calibration=cal.tolist(), test=test.tolist())
        for task, variant, family in combinations():
            if task == 'binary':
                labels = np.where(records.label.eq('Aedes aegypti'), 'Aedes aegypti', 'other_species_or_noise'); classes=np.sort(np.unique(labels)); fi, ci=fit,cal
            else:
                labels = records.label.to_numpy(); classes=species;fi=fit[records.iloc[fit].label.ne('noise')];ci=cal[records.iloc[cal].label.ne('noise')]
            cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(x[variant][fi], labels[fi], records.iloc[fi].pcm_block))
            for aa,bb in cv:
                if set(records.iloc[fi[aa]].pcm_block)&set(records.iloc[fi[bb]].pcm_block):raise ValueError('Inner PCM leakage.')
            model, grid = make_model(family)
            print('FIT',fold,task,variant,family,len(fi),'recordings; NO OUTER/EXTERNAL PREDICTIONS',flush=True)
            search = GridSearchCV(model, grid, scoring='f1_macro', cv=cv, n_jobs=jobs, error_score='raise', refit=True)
            with threadpool_limits(limits=1):search.fit(x[variant][fi], labels[fi])
            model = search.best_estimator_
            p = align_probabilities(model,x[variant][ci],classes);temp=calibrate(p,labels[ci],classes);p=apply_temperature(p,temp)
            rejection=select_rejection(labels[ci],p,classes)
            cutoff=select_aegypti_cutoff(labels[ci],p[:,np.searchsorted(classes,'Aedes aegypti')]) if task=='binary' else None
            settings=dict(fold=fold,task=task,variant=variant,family=family,classes=classes.tolist(),temperature=temp,rejection_threshold=rejection,
                aegypti_cutoff=cutoff,best_params=search.best_params_,cv_macro_f1=float(search.best_score_),calibration_macro_f1=classification_metrics(labels[ci],classes[p.argmax(1)])['macro_f1_strict_union'],
                fit_recordings=fi.tolist(),calibration_recordings=ci.tolist(),inner_pcm_group_isolation=True)
            joblib.dump(dict(model=model,settings=settings),out/'models'/f'{fold}_{task}_{variant}_{family}.joblib',compress=3)
            calibration_scores.append(settings)
    save(out/'partitions.json',plans);save(out/'partition_audit.json',audits);save(out/'calibration.json',calibration_scores)
    # Selected model uses FINAL internal CV only, never OOF/external performance.
    selected={task:max([s for s in calibration_scores if s['fold']=='final' and s['task']==task],key=lambda s:(s['cv_macro_f1'],s['variant']=='v4_308',s['family']=='lr')) for task in ['binary','species']}
    save(out/'selected_models.json',selected)
    paths=list((out/'models').glob('*.joblib'))+[out/f for f in ['protocol.json','feature_bags.npz','new_window_features.npz','recordings.csv','partitions.json','partition_audit.json','calibration.json','selected_models.json']]
    save(out/'freeze.json',dict(created_utc=now(),all_outer_folds_and_final_frozen=True,no_outer_or_external_predictions_yet=True,
         artifacts_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},code_sha256={p:digest(ROOT/p) for p in SOURCES},release_allowed=False))
    print('FROZEN',now(),len(paths),'artifacts',flush=True)


def verify_freeze(out=OUT):
    f=json.loads((out/'freeze.json').read_text())
    for path,h in {**f['artifacts_sha256'],**f['code_sha256']}.items():
        if digest(ROOT/path)!=h:raise ValueError('Frozen transfer artifact changed: '+path)
    return f


def audio_bag(path):
    wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
    if sr!=SR:
        g=np.gcd(sr,SR);wave=resample_poly(wave,SR//g,sr//g)
    windows=wave[:len(wave)//SR*SR].reshape(-1,SR)
    if not len(windows):raise ValueError('Audio lacks complete 1 s window.')
    ref,lfcc=extract_features(windows)
    base=[];salience=[]
    from acoustic_features_v4 import window_features
    for window in windows:
        a,s=window_features(window);base.append(a);salience.append(s)
    base=np.array(base)
    arrays={'notebook61':ref,'v4_308':base,'v4_lfcc334':np.c_[base,lfcc]}
    return {k:pool_features(a,salience) for k,a in arrays.items()},arrays


def predict_bundle(bundle,x):
    s=bundle['settings'];classes=np.array(s['classes']);p=align_probabilities(bundle['model'],x,classes)
    return apply_temperature(p,s['temperature'])


def evaluate(out=OUT):
    frozen=verify_freeze(out)
    if (out/'evaluation_started.json').exists():raise FileExistsError('Do not overwrite completed/started evaluations.')
    save(out/'evaluation_started.json',dict(created_utc=now(),freeze_utc=frozen['created_utc'],freeze_sha256=digest(out/'freeze.json')))
    records=pd.read_csv(out/'recordings.csv');x=np.load(out/'feature_bags.npz');plan=json.loads((out/'partitions.json').read_text());rows=[]
    for fold in ['0','1','2']:
        ix=np.array(plan[fold]['test'],int)
        for task,variant,family in combinations():
            bundle=joblib.load(out/'models'/f'{fold}_{task}_{variant}_{family}.joblib');s=bundle['settings'];keep=ix if task=='binary' else ix[records.iloc[ix].label.ne('noise')]
            p=predict_bundle(bundle,x[variant][keep]);classes=np.array(s['classes'])
            truth=np.where(records.iloc[keep].label.eq('Aedes aegypti'),'Aedes aegypti','other_species_or_noise') if task=='binary' else records.iloc[keep].label.to_numpy()
            for j,i in enumerate(keep):
                rows.append(dict(scope='dryad',fold=fold,task=task,variant=variant,family=family,group=records.iloc[i].group,pcm_block=records.iloc[i].pcm_block,
                    source_label=records.iloc[i].label,true_label=truth[j],predicted_label=classes[p[j].argmax()],confidence=float(p[j].max()),accepted=bool(p[j].max()>=s['rejection_threshold']),
                    aegypti_score=float(p[j,np.searchsorted(classes,'Aedes aegypti')]),aegypti_detected=bool(p[j,np.searchsorted(classes,'Aedes aegypti')]>=s['aegypti_cutoff']) if task=='binary' else None))
    external=json.loads((V4/'external_plan.json').read_text());files=[]
    for key,scope in [('paired_files','external17'),('fresh_previous','fresh2'),('postfreeze_previous','postfreeze2')]:
        files.extend(dict(**r,scope=scope) for r in external[key])
    for path in sorted((V4/'postfreeze_audio').glob('*.wav')):
        files.append(dict(path=str(path.relative_to(ROOT)),group=path.stem,reference_species='Aedes aegypti',scope='postfreeze_v4_2',sha256=digest(path)))
    files.append(dict(path='data/raw/drive_test/drive_audio.wav',group='private_drive',reference_species='Aedes aegypti',scope='drive',sha256=external['private_file']['sha256']))
    selected=json.loads((out/'selected_models.json').read_text());drive_windows=[]
    for file in files:
        path=ROOT/file['path']
        if digest(path)!=file['sha256']:raise ValueError('External audio differs: '+file['path'])
        bags,arrays=audio_bag(path);print('EXTERNAL',file['scope'],file['group'],flush=True)
        for task,variant,family in combinations():
            bundle=joblib.load(out/'models'/f'final_{task}_{variant}_{family}.joblib');s=bundle['settings'];classes=np.array(s['classes']);p=predict_bundle(bundle,np.array([bags[variant]]))[0]
            truth=('Aedes aegypti' if file['reference_species']=='Aedes aegypti' else 'other_species_or_noise') if task=='binary' else file['reference_species']
            rows.append(dict(scope=file['scope'],fold='final',task=task,variant=variant,family=family,group=file['group'],pcm_block=file['group'],source_label=file['reference_species'],true_label=truth,
                predicted_label=classes[p.argmax()],confidence=float(p.max()),accepted=bool(p.max()>=s['rejection_threshold']),aegypti_score=float(p[np.searchsorted(classes,'Aedes aegypti')]),
                aegypti_detected=bool(p[np.searchsorted(classes,'Aedes aegypti')]>=s['aegypti_cutoff']) if task=='binary' else None))
            if file['scope']=='drive':
                # Diagnostic window fractions, NOT window accuracy or calibrated bag probabilities.
                wp=predict_bundle(bundle,arrays[variant])
                for i,v in enumerate(wp):drive_windows.append(dict(task=task,variant=variant,family=family,start_s=i,end_s=i+1,predicted_label=classes[v.argmax()],aegypti_score=float(v[np.searchsorted(classes,'Aedes aegypti')])))
    table=pd.DataFrame(rows);table.to_csv(out/'recording_predictions.csv',index=False);pd.DataFrame(drive_windows).to_csv(out/'drive_window_diagnostics.csv',index=False)
    metrics=[]
    for (scope,task,variant,family),g in table.groupby(['scope','task','variant','family'],sort=False):
        m=classification_metrics(g.true_label,g.predicted_label,g.accepted)
        pos=g.source_label.eq('Aedes aegypti');negative=~pos
        m.update(scope=scope,task=task,variant=variant,family=family,aegypti_n=int(pos.sum()),aegypti_correct=int((g.predicted_label[pos]=='Aedes aegypti').sum()),
                 selected_from_fit_cv=variant==selected[task]['variant'] and family==selected[task]['family'])
        if task=='binary':
            detected=g.aegypti_detected.astype(bool)
            m.update(detection_sensitivity=float(detected[pos].mean()) if pos.any() else None,detection_specificity=float((~detected[negative]).mean()) if negative.any() else None,
                     detection_precision=float(pos[detected].mean()) if detected.any() else None,
                     mosquito_only_accuracy=float((g.true_label[g.source_label.ne('noise')]==g.predicted_label[g.source_label.ne('noise')]).mean()))
        metrics.append(m)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);save(out/'metrics.json',metrics)
    verify_freeze(out);save(out/'evaluation_complete.json',dict(created_utc=now(),freeze_sha256=digest(out/'freeze.json'),rows=len(rows),files_evaluated=len(files),external_retrospective=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['train','evaluate','verify'],required=True);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args()
    if args.phase=='train':train(workers=args.workers,jobs=args.jobs)
    elif args.phase=='evaluate':evaluate()
    else:print(json.dumps(verify_freeze(),indent=2))
