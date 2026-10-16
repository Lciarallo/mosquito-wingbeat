"""Grouped five-fold CNN/MLP/SVM/Extra Trees/HistGB experiment.

Requires the explicit real-data common-front-end cache and matching metadata.
Does not modify slides, notebooks or previous weights. No synthetic augmentation.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from pattern_classifiers import (SpectrogramCNN, AcousticMLP, source_class_weights,
    source_class_loss, source_class_score, assert_partitions, temporal_statistics, require)
from torres_lstm import digest, predict_probabilities
from train_ternary_lstm import GroupPool
from ternary_protocol import calibrate_policy, apply_policy

ROOT = Path(__file__).resolve().parent
NAMES = ['svm_rbf', 'extra_trees', 'histgb', 'mlp', 'cnn']
SETTINGS = dict(seed=20261020, max_epochs=30, patience=5, batch_size=256,
                learning_rate=.0006, weight_decay=.001, gamma=.5)
GRIDS = dict(svm_rbf=[dict(C=c, gamma='scale') for c in [.1, 1., 10.]],
    extra_trees=[dict(min_samples_leaf=n) for n in [2, 8]],
    histgb=[dict(l2_regularization=n) for n in [1., 5.]])


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def metrics(groups, prediction):
    y = groups.true_label.to_numpy(int); p = np.asarray(prediction, int)
    cm = confusion_matrix(y, p, labels=[0, 1, 2]); counts = cm.sum(1)
    recalls = np.divide(np.diag(cm), counts, out=np.zeros(3), where=counts > 0)
    return dict(n=len(y), correct=int((y == p).sum()), accuracy=float((y == p).mean()),
        balanced_accuracy=float(recalls[counts > 0].mean()),
        macro_f1=float(f1_score(y, p, labels=np.flatnonzero(counts), average='macro', zero_division=0)),
        source_class_score=source_class_score(groups, p),
        aedes_tp=int(cm[2, 2]), aedes_total=int(counts[2]),
        aedes_recall=float(recalls[2]) if counts[2] else None,
        noise_fp=int(cm[0, 2]), noise_total=int(counts[0]),
        noise_fpr=float(cm[0, 2] / counts[0]) if counts[0] else None,
        other_fp=int(cm[1, 2]), other_total=int(counts[1]),
        other_fpr=float(cm[1, 2] / counts[1]) if counts[1] else None,
        confusion_matrix=cm.tolist())


def table(frame, probabilities):
    return GroupPool(frame).table(probabilities)


def factory(name, parameters):
    if name == 'svm_rbf':
        # Internal probability=True would use ungrouped CV. Keep raw margins,
        # converted to a normalized score, not claimed as calibrated probabilities.
        return SVC(**parameters, kernel='rbf', probability=False,
                   decision_function_shape='ovr', cache_size=512)
    if name == 'extra_trees':
        return ExtraTreesClassifier(**parameters, n_estimators=250, max_depth=18,
                                    random_state=SETTINGS['seed'], n_jobs=4)
    return HistGradientBoostingClassifier(**parameters, max_iter=200,
        max_leaf_nodes=15, min_samples_leaf=12, learning_rate=.05,
        early_stopping=False, random_state=SETTINGS['seed'])


def ml_proba(model, x):
    if isinstance(model, SVC):
        # OVR ranks dominate a bounded margin correction. Scores are only for
        # pooling/thresholding; argmax is the ordinary SVC decision.
        z = model.decision_function(x)
        z -= z.max(1, keepdims=True)
        p = np.exp(z); return p / p.sum(1, keepdims=True)
    return model.predict_proba(x)


def ml_context(name, fold, frame, stats, out):
    d = out / f'outer_{fold}' / name
    if (d / 'complete.json').exists(): return
    d.mkdir(parents=True, exist_ok=True)
    fit, val, test, dev = partitions(frame, fold)
    scaler = StandardScaler().fit(stats[fit])
    w = source_class_weights(frame.iloc[fit]); candidates = []
    with threadpool_limits(limits=4):
        for config in GRIDS[name]:
            model = factory(name, config).fit(scaler.transform(stats[fit]),
                        frame.window_target.to_numpy(int)[fit], sample_weight=w)
            vp = ml_proba(model, scaler.transform(stats[val])); vg = table(frame.iloc[val], vp)
            score = source_class_score(vg, vg[[f'prob_{c}' for c in range(3)]].to_numpy().argmax(1))
            candidates.append((score, config, vp))
        score, config, vp = max(candidates, key=lambda v: v[0])
        vg = table(frame.iloc[val], vp)
        rule = calibrate_policy(vg, 'lr', temporal=False)
        selection = dict(validation_score=score, configuration=config,
            candidates=[dict(score=a, configuration=b) for a,b,_ in candidates], rule=rule)
        scaler = StandardScaler().fit(stats[dev])
        model = factory(name, config).fit(scaler.transform(stats[dev]),
            frame.window_target.to_numpy(int)[dev], sample_weight=source_class_weights(frame.iloc[dev]))
        joblib.dump(dict(model=model, scaler=scaler, rule=rule, name=name), d/'model.joblib', compress=3)
        freeze(d, frame, fit, val, test, dev, selection, 'model.joblib')
        p = ml_proba(model, scaler.transform(stats[test]))
    finish(d, frame.iloc[test], p, rule, fold)


def partitions(frame, fold):
    test = np.flatnonzero(frame.fold.eq(fold)); vf = min(set(frame.fold) - {fold})
    val = np.flatnonzero(frame.fold.eq(vf)); fit = np.flatnonzero(~frame.fold.isin([fold, vf]))
    dev = np.flatnonzero(frame.fold.ne(fold)); assert_partitions(frame, fit, val, test)
    return fit, val, test, dev


def normalize(raw, indices, path):
    axes = (0, 1) if raw.ndim == 3 else 0
    mean = raw[indices].mean(axis=axes, keepdims=True)
    std = raw[indices].std(axis=axes, keepdims=True).clip(.001)
    np.savez(path, mean=mean, std=std)
    return ((raw - mean) / std).astype('float32')


def nn_factory(name): return SpectrogramCNN() if name == 'cnn' else AcousticMLP()


def train_nn(name, frame, x, fit, val, device, d, seed, epochs=None):
    torch.manual_seed(seed); model = nn_factory(name).to(device)
    values = torch.tensor(np.array(x[fit]), device=device)
    labels = torch.tensor(frame.window_target.to_numpy(int)[fit], device=device)
    weight = torch.tensor(source_class_weights(frame.iloc[fit]), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'],
                                  weight_decay=SETTINGS['weight_decay'])
    history=[]; best=float('inf'); best_epoch=0; stale=0; started=time.monotonic()
    for epoch in range(1, (epochs or SETTINGS['max_epochs']) + 1):
        model.train(); order=torch.multinomial(weight, len(fit), replacement=True); loss_sum=0
        for start in range(0, len(order), SETTINGS['batch_size']):
            ix=order[start:start+SETTINGS['batch_size']]; optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(values[ix]), labels[ix])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3); optimizer.step()
            loss_sum+=float(loss.detach())*len(ix)
        row=dict(epoch=epoch, train_loss=loss_sum/len(fit))
        if val is not None:
            vp=predict_probabilities(model, x[val], device, batch_size=256)
            vl=source_class_loss(frame.iloc[val], vp); row['validation_loss']=vl
            if vl<best-1e-4:
                best=vl; best_epoch=epoch; stale=0
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, d/'inner_best.pt')
            else: stale+=1
        history.append(row)
        write(d/('refit_history.json' if epochs else 'inner_history.json'), history)
        print('EPOCH',d.parent.name,name,'refit' if epochs else 'select',epoch,
              'best',best_epoch,'seconds',round(time.monotonic()-started,1),flush=True)
        if val is not None and stale>=SETTINGS['patience']: break
    if val is not None: model.load_state_dict(torch.load(d/'inner_best.pt',map_location='cpu',weights_only=True))
    return model, dict(best_epoch=best_epoch if val is not None else epochs,
                      validation_loss=best if val is not None else None)


def neural_context(name, fold, frame, raw, stats, out, device):
    d=out/f'outer_{fold}'/name
    if (d/'complete.json').exists(): return
    d.mkdir(parents=True,exist_ok=True); fit,val,test,dev=partitions(frame,fold)
    features=raw if name=='cnn' else stats
    x=normalize(features,fit,d/'inner_scaler.npz')
    seed=SETTINGS['seed']+fold*101
    model,selection=train_nn(name,frame,x,fit,val,device,d,seed)
    vp=predict_probabilities(model,x[val],device,batch_size=256);vg=table(frame.iloc[val],vp)
    selection['validation_score']=source_class_score(vg,vg[[f'prob_{c}' for c in range(3)]].to_numpy().argmax(1))
    rule=calibrate_policy(vg,'lr',temporal=False); selection['rule']=rule
    del model,x
    x=normalize(features,dev,d/'scaler.npz')
    model,_=train_nn(name,frame,x,dev,None,device,d,seed,epochs=selection['best_epoch'])
    torch.save(dict(config=model.config,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},rule=rule,name=name),d/'model.pt')
    freeze(d,frame,fit,val,test,dev,selection,'model.pt')
    p=predict_probabilities(model,x[test],device,batch_size=256)
    finish(d,frame.iloc[test],p,rule,fold)
    del model,x
    if torch.cuda.is_available():torch.cuda.empty_cache()


def freeze(d,frame,fit,val,test,dev,selection,filename):
    write(d/'selection.json',selection)
    part=frame[['row_id','group','pcm_block','domain','true_label','window_target','fold']].copy()
    part['role']='test';part.loc[fit,'role']='fit';part.loc[val,'role']='validation'
    part.to_csv(d/'partitions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write(d/'freeze.json',dict(before_test_inference=True,model_sha256=digest(d/filename),
        selection_sha256=digest(d/'selection.json'),fit_rows=dev.tolist(),
        scaler_sha256=digest(d/'scaler.npz') if filename.endswith('.pt') else None))


def finish(d,frame,p,rule,fold):
    np.save(d/'window_test_probabilities.npy',p);g=table(frame,p)
    g['argmax_prediction']=g[[f'prob_{c}' for c in range(3)]].to_numpy().argmax(1)
    g['calibrated_prediction']=apply_policy(g,rule);g['fold']=fold
    g.to_csv(d/'group_predictions.csv',index=False)
    write(d/'complete.json',dict(complete=True,argmax=metrics(g,g.argmax_prediction),calibrated=metrics(g,g.calibrated_prediction)))
    print('DONE',fold,d.name,json.dumps(metrics(g,g.argmax_prediction)),flush=True)


def summarize(out):
    out=Path(out)
    if not all((out/f'outer_{f}'/n/'complete.json').exists() for f in range(5) for n in NAMES):return
    expected_groups=int(pd.read_csv(out/'segments.csv.gz',low_memory=False).group.nunique()) if (out/'segments.csv.gz').is_file() else 1794
    all_groups={}; selections=[]
    for name in NAMES:
        g=pd.concat([pd.read_csv(out/f'outer_{f}'/name/'group_predictions.csv') for f in range(5)],ignore_index=True)
        require(len(g)==g.group.nunique()==expected_groups,f'OOF group coverage mismatch for {name}')
        g.to_csv(out/f'{name}_oof.csv',index=False);all_groups[name]=g
    chosen=[]
    for fold in range(5):
        scores={n:json.loads((out/f'outer_{fold}'/n/'selection.json').read_text())['validation_score'] for n in NAMES}
        name=max(NAMES,key=lambda n:scores[n]);chosen.append(all_groups[name][all_groups[name].fold.eq(fold)].assign(selected_model=name))
        selections.append(dict(fold=fold,selected_model=name,validation_scores=scores))
    selected=pd.concat(chosen,ignore_index=True);selected.to_csv(out/'selected_oof.csv',index=False);all_groups['validation_selected']=selected
    report={}
    for name,g in all_groups.items():
        report[name]={rule:{key:metrics(s,s[rule+'_prediction']) for key,s in [('all',g),('original_corpus',g[g.source.ne('torres')]),*list(g.groupby('source'))]} for rule in ['argmax','calibrated']}
    write(out/'metrics.json',report);write(out/'model_selection.json',selections)
    write(out/'complete.json',dict(complete=True,models=5,folds=5,examples_per_model=expected_groups))


def finalize_model(out, frame, raw, stats, device):
    out=Path(out)
    require((out/'complete.json').is_file() and json.loads((out/'complete.json').read_text())['complete'],'Completed 5-fold evaluation is required before finalizing model.')
    scores={name:float(np.mean([json.loads((out/f'outer_{fold}'/name/'selection.json').read_text())['validation_score'] for fold in range(5)])) for name in NAMES}
    name=max(NAMES,key=lambda n:scores[n]);d=out/'final';d.mkdir(parents=True,exist_ok=True)
    if (d/'manifest.json').exists():
        return json.loads((d/'manifest.json').read_text())
    write(d/'selection.json',dict(model=name,mean_inner_validation_scores=scores,
        criterion='source/class-balanced accuracy from inner folds only; retrospective training candidate'))
    oof=pd.read_csv(out/f'{name}_oof.csv')
    calibrated_rule=calibrate_policy(oof,'lr',temporal=False)
    indices=np.arange(len(frame))
    if name in ['mlp','cnn']:
        features=raw if name=='cnn' else stats
        epochs=int(np.median([json.loads((out/f'outer_{fold}'/name/'selection.json').read_text())['best_epoch'] for fold in range(5)]))
        x=normalize(features,indices,d/'scaler.npz');model,_=train_nn(name,frame,x,indices,None,device,d,SETTINGS['seed'],epochs=epochs)
        torch.save(dict(name=name,config=model.config,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},rule=calibrated_rule),d/'model.pt')
        manifest=dict(model=name,model_sha256=digest(d/'model.pt'),scaler_sha256=digest(d/'scaler.npz'),epochs=epochs)
    else:
        candidates=GRIDS[name];averages=[]
        for config in candidates:
            vals=[]
            for fold in range(5):
                selection=json.loads((out/f'outer_{fold}'/name/'selection.json').read_text())
                vals.extend(v['score'] for v in selection['candidates'] if v['configuration']==config)
            averages.append(np.mean(vals))
        config=candidates[int(np.argmax(averages))];scaler=StandardScaler().fit(stats)
        with threadpool_limits(limits=4):
            model=factory(name,config).fit(scaler.transform(stats),frame.window_target.to_numpy(int),sample_weight=source_class_weights(frame))
        joblib.dump(dict(model=model,scaler=scaler,rule=calibrated_rule,name=name),d/'model.joblib',compress=3)
        manifest=dict(model=name,model_sha256=digest(d/'model.joblib'),configuration=config)
    manifest.update(rule='argmax',calibrated_rule=calibrated_rule,training_groups=int(frame.group.nunique()),
        training_windows=len(frame),mean_inner_validation_scores=scores,
        frontend='179time x16 lowfrequencyHTKMel approximate author frontend; stats80forML/MLP',
        final_model_has_no_new_independent_test=True,not_field_validated=True)
    write(d/'manifest.json',manifest)
    return manifest


def main():
    from verify_pattern_classifiers import resolve_document_revision, verify_protected_files
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True,help='Directory with segments.csv.gz, feature_cache.json, classical_features.npy')
    parser.add_argument('--sequences',type=Path,required=True,help='Real common-front-end sequence cache, .npy')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['ml','neural','all'],default='all')
    parser.add_argument('--document-revision',type=Path,help='Explicit receipt for the authorized notebook update; never permits model/data changes.')
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame=pd.read_csv(args.reference/'segments.csv.gz',low_memory=False)
    raw=np.load(args.sequences,mmap_mode='r');stats=np.load(args.reference/'classical_features.npy',mmap_mode='r')
    require(len(frame)==len(raw)==len(stats) and raw.shape[1:]==(179,16),'Input feature shapes do not align with segment table.')
    cache=json.loads((args.reference/'feature_cache.json').read_text());require(digest(args.sequences)==cache['sha256'],'Sequence cache SHA-256 mismatch.')
    require(frame.group.nunique()==1794 and frame.groupby('group').true_label.nunique().max()==1,'Invalid group count or mixed group labels.')
    manifest=dict(settings=SETTINGS,grids=GRIDS,models=NAMES,
        selection='inner source/class-balanced group accuracy; neural epochs by source/class-balanced group log loss',
        primary_rule='argmax fixed before test; calibrated secondary',
        weighting='equal mass per training source/window-label cell with gamma0.5 recording damping',
        frontend=cache['identity'],approximate_torres_frontend=True,synthetic_examples=False,
        segments_sha256=digest(args.reference/'segments.csv.gz'),features_sha256=digest(args.reference/'classical_features.npy'),
        code_sha256={p:digest(ROOT/p) for p in ['pattern_classifiers.py','train_pattern_classifiers.py']},
        protected_sha256=json.loads((args.reference/'protocol.json').read_text())['protected_sha256'])
    revision=resolve_document_revision(args.document_revision)
    verify_protected_files(ROOT,manifest['protected_sha256'],revision)
    if (out/'protocol.json').exists():
        require(json.loads((out/'protocol.json').read_text())==manifest,'Existing protocol.json does not match current manifest.')
    else:
        snapshot=out/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
        for p in ['pattern_classifiers.py','train_pattern_classifiers.py']:
            (snapshot/p).write_bytes((ROOT/p).read_bytes())
        write(out/'protocol.json',manifest)
    if not (out/'segments.csv.gz').exists():
        frame.to_csv(out/'segments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    for fold in range(5):
        for name in (NAMES[:3] if args.stage=='ml' else NAMES[3:] if args.stage=='neural' else NAMES):
            if name in NAMES[:3]:ml_context(name,fold,frame,stats,out)
            else:neural_context(name,fold,frame,raw,stats,out,device)
    summarize(out)
    if (out/'complete.json').exists():
        finalize_model(out,frame,raw,stats,device)
    verify_protected_files(ROOT,manifest['protected_sha256'],revision)


if __name__=='__main__':main()
