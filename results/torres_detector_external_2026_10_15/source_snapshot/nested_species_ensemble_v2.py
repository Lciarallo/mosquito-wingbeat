"""Frozen offline nested v2; historical outputs and embedded code are read-only.

Usage: .venv/bin/python nested_species_ensemble_v2.py --phase train
       .venv/bin/python nested_species_ensemble_v2.py --phase evaluate
"""
import argparse
import hashlib
import json
import math
import time
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import minimize_scalar
from scipy.special import softmax, expit, logit
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from acoustic_features_v2 import SR, FEATURE_NAMES, extract_features, window_features
from improve_device import ROOT, METHODS, model_for, fit_classical, predict_proba, split_calibration, file_hash, atomic_json
from nested_species_ensemble import now, check_partition as legacy_partition, fit_blend
from evaluate_nested_species import verify_freeze as verify_legacy_freeze, pcm_keys
from research_metrics_v2 import classification_metrics

OUT = ROOT/'results/species_nested_2026_10_13'
OLD = ROOT/'results/species_nested_2026_10_09'
SEED = 20261013
MEMBERS = ['svm','extra_trees','hist_gradient_boosting']
SOURCE_FILES = ['acoustic_features_v2.py','research_metrics_v2.py','nested_species_ensemble_v2.py',
                'compare_detection_levels_v2.py','improve_device.py','nested_species_ensemble.py',
                'evaluate_nested_species.py','fetch_species_v2_fresh.py','fetch_humbug.py']


def verify_history(snapshot=None):
    snapshot = snapshot or ROOT/'tmp/v2_2026_10_13/historical_snapshot.json'
    data = json.loads(Path(snapshot).read_text())
    for name, digest in data['files_sha256'].items():
        p = ROOT/name
        if not p.is_file() or file_hash(p) != digest:
            raise ValueError('Historical file changed or disappeared: '+name)
    verify_legacy_freeze(OLD)
    return dict(checked_utc=now(), files=len(data['files_sha256']), unchanged=True,
                snapshot_sha256=file_hash(Path(snapshot)), legacy_verify_freeze='passed')


def verify_freeze(out=OUT):
    data = json.loads((out/'freeze.json').read_text())
    for name,digest in data['artifacts_sha256'].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError('Frozen artifact changed: '+name)
    for name,digest in data['code_sha256'].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError('Frozen inference/selection code changed: '+name)
    return data


def source_pcm_keys(path):
    try:
        return pcm_keys(path)
    except sf.LibsndfileError:
        # Match the original dataset reader's compressed-media decoding at 16 kHz.
        decoded=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar','16000',
                                '-f','f32le','pipe:1'],check=True,capture_output=True).stdout
        wave=resample_poly(np.frombuffer(decoded,dtype='<f4'),1,2)
        pieces=[wave]
        for size in [8000,7936,2400]:
            pieces.extend(wave[i:i+size] for i in range(0,len(wave)-size+1,size))
        keys=set()
        for fragment in pieces:
            if not len(fragment):continue
            for a in [fragment,fragment/max(float(np.abs(fragment).max()),1e-12)*.9]:
                pcm=np.rint(np.clip(a,-1,32767/32768)*32768).astype('<i2')
                if np.ptp(pcm.astype('int32'))>2:
                    keys.add((len(pcm),hashlib.sha256(pcm.tobytes()).hexdigest()))
        return keys


def check_partition(frame, fit, calibration, test):
    audit = legacy_partition(frame,fit,calibration,test)
    if 'collision_group' in frame:
        a,b,c=[set(frame.iloc[np.asarray(ix,int)].collision_group) for ix in (fit,calibration,test)]
        if a&b or a&c or b&c:
            raise ValueError('PCM collision-group leakage between partitions.')
    return audit


def split_collision_calibration(indices,frame,seed):
    sub=frame.iloc[np.asarray(indices,int)]
    if sub.groupby('group').label.nunique().max()==1:
        return split_calibration(indices,frame,seed)
    rng=np.random.default_rng(seed)
    memberships={g:set(f.label) for g,f in sub.groupby('group')}
    bylabel={label:set(sub[sub.label.eq(label)].group) for label in sorted(sub.label.unique())}
    if min(map(len,bylabel.values()))<2:raise ValueError('Insufficient disjoint PCM blocks for calibration.')
    target={label:min(len(gs)-1,max(1,math.ceil(.2*len(gs)))) for label,gs in bylabel.items()}
    chosen=set()
    for label in rng.permutation(list(bylabel)):
        for g in rng.permutation(sorted(bylabel[label])):
            if len(chosen&bylabel[label])>=target[label]:break
            if g in chosen:continue
            if all(len(bylabel[k]-(chosen|{g}))>=1 for k in memberships[g]):chosen.add(g)
    if any(not (chosen&gs) for gs in bylabel.values()):
        raise ValueError('Cannot reserve calibration and fit representatives for every label.')
    ix=np.asarray(indices,int);mask=frame.iloc[ix].group.isin(chosen).to_numpy()
    return ix[~mask],ix[mask]


def marginalize_genus(p, classes):
    p=np.asarray(p,dtype=np.float64)
    membership=np.array([str(c).split()[0] for c in classes])
    genera=np.unique(membership)
    q=np.column_stack([p[:,membership==g].sum(1) for g in genera])
    if not np.allclose(q.sum(1),1,rtol=0,atol=1e-6):
        raise ValueError('Genus probability mass is not conserved.')
    return genera,q


def inductive_probabilities(model, features):
    # Prediction is deliberately performed identically for a standalone row or
    # a batch: neither sklearn/BLAS batching nor a batch-normaliser changes bits.
    if hasattr(model,'n_jobs'):
        model.set_params(n_jobs=1)  # Deterministic tree summation order, also persisted before freeze.
    return np.vstack([predict_proba(model,np.asarray(row)[None,:])[0] for row in features]).astype(np.float64)


def calibrated_member(p, temperature):
    return np.vstack([softmax(np.log(np.maximum(row,1e-12))/temperature) for row in p])


def mix_probabilities(members, parameters):
    w=np.asarray(parameters['weights']);prior=np.asarray(parameters['log_prior'])
    p=np.asarray(members,dtype=float)
    return np.vstack([softmax(np.log(np.maximum(np.sum(p[:,i,:]*w[:,None],axis=0),1e-12))+prior)
                      for i in range(p.shape[1])])


def training_weights(indices, frame, salience, class_balance=True, device_balance=True):
    """Every frequency is calculated from fit indices only, never from a test batch."""
    sub=frame.iloc[np.asarray(indices,int)]
    if 'collision_group' in sub:sub=sub.assign(group=sub.collision_group)
    size=sub.groupby('group').size()
    w=1/sub.group.map(size).to_numpy(float)
    if class_balance:
        count=sub[['label','group']].drop_duplicates().groupby('label').size()
        w /= sub.label.map(count).to_numpy(float)
    if device_balance:
        devices=sub[['label','phone','group']].drop_duplicates()
        joint=devices.groupby(['label','phone']).size()
        totals=devices.groupby('label').size()
        ndevices=devices.groupby('label').phone.nunique()
        multipliers=np.array([math.sqrt(totals[label]/ndevices[label]/joint.loc[(label,phone)])
                              for label,phone in zip(sub.label,sub.phone)])
        w*=multipliers
    w*=np.asarray(salience)[np.asarray(indices,int)]
    return w/max(float(w.mean()),1e-12)


def pool_recording(p, salience, top_fraction=1., quality=None):
    p=np.asarray(p,dtype=float);s=np.asarray(salience,dtype=float)
    if len(p)==0 or len(p)!=len(s):raise ValueError('Empty or misaligned recording bags.')
    sharpness=np.maximum(1+np.sum(p*np.log(np.maximum(p,1e-12)),axis=1)/np.log(p.shape[1]),0)
    # Acoustic rank, not class-prediction rank; deterministic ties by window order.
    count=max(1,int(np.ceil(len(p)*top_fraction)))
    ix=np.argsort(-s,kind='stable')[:count]
    w=s[ix]*np.maximum(sharpness[ix],.01)
    if quality is not None:
        w*=np.asarray(quality,bool)[ix]
    if w.sum()<=0:
        return p.mean(0), dict(selected_windows=count,weighted_windows=0,quality_ok=False)
    result=np.sum(p[ix]*w[:,None],axis=0)/w.sum()
    return result,dict(selected_windows=count,weighted_windows=int((w>0).sum()),quality_ok=True)


def recording_probabilities(frame,p,salience,fraction,quality=None):
    rows=[]; probabilities=[]
    for group,ix in frame.groupby('group',sort=True).indices.items():
        sub=frame.iloc[ix]
        if sub.label.nunique()!=1:raise ValueError('Conflicting weak labels within a recording group.')
        q,info=pool_recording(p[ix],salience[ix],fraction,None if quality is None else quality[ix])
        rows.append(dict(group=group,label=sub.label.iloc[0],
                         collision_group=sub.collision_group.iloc[0] if 'collision_group' in sub else group,**info))
        probabilities.append(q)
    return pd.DataFrame(rows),np.stack(probabilities)


def choose_pooling(frame,p,salience,classes,partition):
    if partition!='calibration':raise ValueError('MIL selection requires calibration groups.')
    candidates=[]
    for fraction in (.25,.5,1.):
        r,q=recording_probabilities(frame,p,salience,fraction)
        metrics=classification_metrics(r.label,np.asarray(classes)[q.argmax(1)])
        candidates.append(dict(top_fraction=fraction,**metrics))
    best=max(candidates,key=lambda z:(z['balanced_accuracy'],z['macro_f1_strict_union'],z['top_fraction']))
    return dict(top_fraction=best['top_fraction'],candidates=candidates,partition='calibration',
                objective='recording balanced accuracy, strict F1, larger fraction as tie-break')


def select_threshold(labels,p,classes,partition,collision_groups=None):
    if partition!='calibration':raise ValueError('Rejection requires calibration groups.')
    labels=np.asarray(labels);p=np.asarray(p);scores=p.max(1);correct=np.asarray(classes)[p.argmax(1)]==labels
    blocks=np.arange(len(labels)) if collision_groups is None else np.asarray(collision_groups)
    for t in np.r_[0.,np.sort(np.unique(scores))]:
        a=scores>=t
        if len(np.unique(blocks[a]))>=20 and correct[a].mean()>=.90:
            return dict(threshold=float(t),calibration_groups=len(labels),calibration_emitted=int(a.sum()),
                        calibration_accuracy=float(correct[a].mean()),target=.90,minimum_emissions=20,
                        calibration_emitted_collision_blocks=int(len(np.unique(blocks[a]))))
    return dict(threshold=1.000001,calibration_groups=len(labels),calibration_emitted=0,
                calibration_accuracy=None,target=.90,minimum_emissions=20)


def collision_audit(frame,out):
    """Audit decoded identity and fixed aligned PCM fragments, before any fitting."""
    parent={g:g for g in frame.group.unique()}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    keys={};collisions=[];files=[]
    for n,(path,sub) in enumerate(frame.groupby('path',sort=True),1):
        if sub.group.nunique()!=1:raise ValueError('One source path has multiple groups.')
        g=sub.group.iloc[0];p=ROOT/path
        # Bound identical *decoded* PCM and common aligned crops, including gain variants.
        signatures=source_pcm_keys(p)
        for key in signatures:
            if key in keys and find(keys[key][0])!=find(g):
                other,other_path=keys[key]
                parent[find(g)]=find(other)
                collisions.append(dict(path=path,other_path=other_path,length_pcm=key[0],pcm_sha256=key[1]))
            else:keys[key]=(g,path)
        files.append(dict(path=path,group=g,file_sha256=file_hash(p),pcm_signatures=len(signatures)))
        if n%100==0:print('PCM AUDIT',n,'files',flush=True)
    frame=frame.copy();frame['collision_group']=frame.group.map(find)
    conflicts=[dict(collision_group=str(group),labels=sorted(sub.label.unique()),
                    original_groups=int(sub.group.nunique()))
               for group,sub in frame.groupby('collision_group') if sub.label.nunique()>1]
    pd.DataFrame(files).to_csv(out/'source_pcm_audit.csv',index=False)
    pd.DataFrame(collisions,columns=['path','other_path','length_pcm','pcm_sha256']).to_csv(out/'pcm_collisions.csv',index=False)
    # Preserve historic folds whenever possible; otherwise assign collision blocks anew.
    changed=frame.groupby('collision_group').fold_species.nunique().max()>1
    if changed:
        folds=np.full(len(frame),-1)
        splitter=StratifiedGroupKFold(3,shuffle=True,random_state=SEED)
        for f,(_,test) in enumerate(splitter.split(np.zeros(len(frame)),frame.label,frame.collision_group)):
            folds[test]=f
        frame['fold_species']=folds
    # A collision block can contain conflicting file-level weak labels (background
    # reused as a negative). Keep those references and recordings; never relabel
    # or delete them. Only the partitioning unit is the connected collision block.
    frame['original_group']=frame.group
    frame.to_csv(out/'segments_v2.csv',index=False)
    atomic_json(out/'collision_audit.json',dict(created_utc=now(),source_files=len(files),
        original_groups=frame.original_group.nunique(),collision_groups=frame.collision_group.nunique(),
        additional_collisions=len(collisions),outer_folds_reassigned=bool(changed),
        conflicting_label_blocks=conflicts,
        limits='Whole decoded PCM and aligned 1.0/.992/.3 s fragments at 8 kHz, raw/gain-normalised, nonconstant. Not arbitrary edits or biological/session independence.'))
    return frame


def external_plan():
    recordings=pd.read_csv(OLD/'external_recording_predictions.csv')
    recordings=recordings[recordings.model.eq('calibrated')].sort_values('group')
    oldplan=json.loads((OLD/'fresh_selection.json').read_text())
    items={x['group']:x for x in oldplan['files']}
    selected=[items[g] for g in recordings.group]
    if len(selected)!=17:raise ValueError('Expected exactly the historical 17 recordings.')
    for item in selected:
        if file_hash(ROOT/item['path'])!=item['sha256']:raise ValueError('Paired audio bytes changed.')
    return dict(created_utc=now(),paired_files=selected,private_file=dict(path='drive_audio.wav',
                sha256=file_hash(ROOT/'drive_audio.wav'),reference_species='Aedes aegypti'),
                retrospective=True,fresh_rule='After freeze, web search official BioDCASE Zenodo, first 2 unused numeric D3 IDs each for Aedes aegypti/albopictus, duration >=1 s. Exclude known bytes/PCM before predictions; never replace by prediction.')


def _temperature(p,y,w):
    def loss(logt):
        q=calibrated_member(p,np.exp(logt))
        return float(np.average(-np.log(np.maximum(q[np.arange(len(y)),y],1e-12)),weights=w))
    result=minimize_scalar(loss,bounds=(-1.6,1.6),method='bounded',options={'xatol':1e-4})
    return float(np.exp(result.x))


def _presence_rule(r,q):
    true=r.label.ne('noise').to_numpy();score=q[:,1]
    candidates=[]
    for t in np.r_[0.,np.sort(np.unique(score)),1.000001]:
        pred=score>=t
        fpr=float(pred[~true].mean()) if (~true).any() else None
        if fpr is not None and fpr<=.05:
            ba=.5*(float(pred[true].mean())+1-fpr)
            candidates.append((ba,-t,t,fpr))
    if not candidates:raise ValueError('Presence calibration has no negative groups.')
    best=max(candidates)
    return dict(threshold=float(best[2]),calibration_balanced_accuracy=best[0],calibration_fpr=best[3],
                positive_groups=int(true.sum()),negative_groups=int((~true).sum()),fpr_target=.05)


def prediction_records(frame,p,salience,classes,settings,presence_p,quality=None,scope='dryad'):
    r,q=recording_probabilities(frame,p,salience,settings['pooling']['top_fraction'],quality)
    pr,pq=recording_probabilities(frame,presence_p,salience,settings['presence_pooling']['top_fraction'],quality)
    if not np.array_equal(r.group,pr.group):raise ValueError('Recording identity differs between tasks.')
    genera,gp=marginalize_genus(q,classes)
    species=np.asarray(classes)[q.argmax(1)]
    r['scope']=scope;r['predicted_species']=species;r['species_score']=q.max(1)
    r['true_genus']=r.label.str.split().str[0];r['genus_posthoc']=pd.Series(species).str.split().str[0].to_numpy()
    r['genus_marginalized']=genera[gp.argmax(1)];r['genus_score']=gp.max(1)
    r['accepted_species']=r.quality_ok & (q.max(1)>=settings['threshold_species']['threshold'])
    r['accepted_genus']=r.quality_ok & (gp.max(1)>=settings['threshold_genus']['threshold'])
    r['presence_score']=pq[:,1]
    r['detected']=pr.quality_ok & (pq[:,1]>=settings['threshold_presence']['threshold'])
    r['accepted_species_gated']=r.accepted_species & r.detected
    r['accepted_genus_gated']=r.accepted_genus & r.detected
    return r,q,gp


def train(out=OUT):
    if out.exists():raise FileExistsError('New experiment directory required; never overwrite a result.')
    before=verify_history()
    out.mkdir(parents=True);(out/'models').mkdir()
    atomic_json(out/'historical_snapshot.json',json.loads((ROOT/'tmp/v2_2026_10_13/historical_snapshot.json').read_text()))
    atomic_json(out/'historical_hashes_before.json',before)
    frame=pd.read_csv(ROOT/'results/segments.csv');frame.phone=frame.phone.fillna('unknown')
    plan=external_plan();atomic_json(out/'external_plan.json',plan)
    classes=np.sort(frame[frame.label.ne('noise')].label.unique())
    if len(classes)!=20:raise ValueError('Expected exactly 20 species.')
    protocol=dict(created_utc=now(),seed=SEED,classes=classes.tolist(),members=MEMBERS,
        primary_model='calibrated',outer_folds=[0,1,2],window_seconds=1,refit_on_calibration=False,
        feature_names=FEATURE_NAMES,feature_normalisation='Per-window gain and spectral tilt removal; PSD median/IQR; static MFCC 0..2 zero, MFCC 3 x .25; all fitted scalers fit-only.',
        weighting='inverse group size x inverse class group count x sqrt(inverse conditional species/device group frequency) x harmonic salience [.1,3]; frequencies fit-only',
        calibration_rule='20% collision groups per species/noise, ceil; mixed-label blocks selected together by seeded greedy union, preserving every fit/calibration class; seed 20261013+fold (final+100)',
        mil_candidates=[.25,.5,1.],mil_weight='salience * max(1-H(p)/log(K), .01)',
        member_temperature_bounds_log=[-1.6,1.6],blend_prior_penalty=.01,
        rejection_target_recording_accuracy=.90,rejection_minimum_emissions=20,
        rejection_minimum_unit='distinct PCM collision blocks among accepted recordings',presence_fpr_target=.05,
        genus_rule='exact sum of calibrated species probabilities; separate calibrated genus rejection',
        calibration_reuse='Temperature, blend, pooling and rejection reuse inner calibration. Optimism there is possible; outer groups are never used for selection.',
        quality_rule='RMS-independent raw peak >=32/32768 and clipping <=1%; invalid windows remain in denominator. Only valid selected windows vote in MIL; all-invalid recordings cannot emit.',
        species_only_primary_no_presence_gate=True,integrated_presence_gate_saved_separately=True,
        no_test_batch_statistics=True,geographic_prior=False,release_allowed=False,arduino_compiled=False,
        prediction_threads=1,prediction_batching='exactly one window; deterministic member summation',
        external_plan_sha256=file_hash(out/'external_plan.json'),code_sha256={p:file_hash(ROOT/p) for p in SOURCE_FILES},
        input_sha256={p:file_hash(ROOT/p) for p in ['results/segments.csv','data/cache/waves.npy','data/cache/features_manifest.json']})
    atomic_json(out/'protocol.json',protocol)
    frame=collision_audit(frame,out)
    waves=np.load(ROOT/'data/cache/waves.npy',mmap_mode='r')
    if len(waves)!=len(frame):raise ValueError('Cache row identity differs.')
    features=np.empty((len(frame),len(FEATURE_NAMES)),np.float32);salience=np.empty(len(frame))
    for i,w in enumerate(waves):
        features[i],salience[i]=window_features(w)
        if (i+1)%2000==0:print('INDUCTIVE FEATURES',i+1,'/',len(frame),flush=True)
    np.savez_compressed(out/'features_v2.npz',features=features,salience=salience,
                        row_indices=np.arange(len(frame)),groups=frame.group.to_numpy(str))
    species_ix=np.flatnonzero(frame.label.ne('noise'))
    labels=np.full(len(frame),-1);labels[species_ix]=np.searchsorted(classes,frame.iloc[species_ix].label)
    presence_labels=frame.label.ne('noise').to_numpy(int)
    oof=np.empty((len(frame),len(classes)));presence_oof=np.empty((len(frame),2))
    base_oof={n:np.zeros((len(frame),len(classes))) for n in MEMBERS}
    parts=[];audits=[];recs=[];fold_settings={}
    for fold in [0,1,2,'final']:
        test=np.flatnonzero(frame.fold_species.eq(fold)) if fold!='final' else np.array([],int)
        outer=np.flatnonzero(~frame.fold_species.eq(fold)) if fold!='final' else np.arange(len(frame))
        partition_frame=frame.copy();partition_frame['group']=frame.collision_group
        # Mixed-label blocks are reserved together. An iterative seeded union
        # selects calibration blocks until every taxon has its target coverage,
        # without moving any block whose removal would lose a fit class.
        fit,cal=split_collision_calibration(outer,partition_frame,SEED+(fold if isinstance(fold,int) else 100))
        audit=check_partition(frame,fit,cal,test);audits.append(dict(fold=fold,**audit,overlap_groups_paths_pcm=0))
        for part,ix in [('fit',fit),('calibration',cal),('test',test)]:
            parts.extend(dict(fold=fold,partition=part,row_index=int(i),group=frame.iloc[i].group,
                              original_group=frame.iloc[i].original_group,collision_group=frame.iloc[i].collision_group,
                              path=frame.iloc[i].path,label=frame.iloc[i].label,content_sha256=frame.iloc[i].content_sha256) for i in ix)
        pd.DataFrame(parts).to_csv(out/'partitions.csv',index=False)
        fit_sp=fit[presence_labels[fit]==1];cal_sp=cal[presence_labels[cal]==1];test_sp=test[presence_labels[test]==1]
        check_partition(frame,fit_sp,cal_sp,test_sp)
        weight=training_weights(fit_sp,frame,salience)
        calw=training_weights(cal_sp,frame,np.ones(len(frame)),device_balance=False)
        models=[];pc=[];pt=[];temps=[];timings=[]
        for k,name in enumerate(MEMBERS):
            print('FOLD',fold,name,'fit',len(fit_sp),'cal',len(cal_sp),'test',len(test),flush=True)
            start=time.monotonic();model=fit_classical(model_for(METHODS[k]),features[fit_sp],labels[fit_sp],weight)
            raw=inductive_probabilities(model,features[cal_sp]);temperature=_temperature(raw,labels[cal_sp],calw)
            pc.append(calibrated_member(raw,temperature));temps.append(temperature)
            if len(test):pt.append(calibrated_member(inductive_probabilities(model,features[test]),temperature))
            timings.append(dict(member=name,seconds=time.monotonic()-start,temperature=temperature))
            modeldata=dict(model=model,classes=classes.tolist(),fit_rows=fit_sp,calibration_rows=cal_sp,
                           test_rows=test_sp,member=name,created_utc=now(),temperature=temperature)
            joblib.dump(modeldata,out/'models'/f'{fold}_{name}.joblib',compress=3);models.append(model)
        parameters=fit_blend(np.stack(pc),labels[cal_sp],calw,partition='calibration')
        cp=mix_probabilities(pc,parameters)
        pooling=choose_pooling(frame.iloc[cal_sp],cp,salience[cal_sp],classes,'calibration')
        cr,cq=recording_probabilities(frame.iloc[cal_sp],cp,salience[cal_sp],pooling['top_fraction'])
        genera,cg=marginalize_genus(cq,classes)
        thresholds=dict(threshold_species=select_threshold(cr.label,cq,classes,'calibration',cr.collision_group),
                        threshold_genus=select_threshold(cr.label.str.split().str[0],cg,genera,'calibration',cr.collision_group))
        print('FOLD',fold,'presence fit',len(fit),flush=True)
        # Positive salience weights; negative groups retain ordinary group/class weights.
        presence_salience=np.where(presence_labels,salience,1.)
        binary_frame=frame.copy();binary_frame['label']=np.where(presence_labels,'mosquito','noise')
        pres=fit_classical(model_for(METHODS[2]),features[fit],presence_labels[fit],
                           training_weights(fit,binary_frame,presence_salience))
        rawpc=inductive_probabilities(pres,features[cal])
        pw=training_weights(cal,binary_frame,np.ones(len(frame)),device_balance=False)
        calibrator=LogisticRegression(C=100,random_state=SEED).fit(logit(np.clip(rawpc[:,1],1e-6,1-1e-6))[:,None],presence_labels[cal],sample_weight=pw)
        def calibrated_presence(x):
            return np.vstack([calibrator.predict_proba([[float(logit(np.clip(row[1],1e-6,1-1e-6)))]])[0] for row in x])
        pcp=calibrated_presence(rawpc)
        # Pooling for presence is independently selected on binary calibration labels.
        pp=choose_pooling(binary_frame.iloc[cal],pcp,salience[cal],np.array(['noise','mosquito']),'calibration')
        pr,pq=recording_probabilities(frame.iloc[cal],pcp,salience[cal],pp['top_fraction'])
        settings=dict(pooling=pooling,presence_pooling=pp,threshold_presence=_presence_rule(pr,pq),
                      temperatures=temps,parameters=parameters,timings=timings,**thresholds)
        fold_settings[str(fold)]=settings;atomic_json(out/f'calibration_{fold}.json',settings)
        np.savez_compressed(out/f'calibration_probabilities_{fold}.npz',row_indices=cal_sp,members=np.stack(pc),
                            blended=cp,presence_row_indices=cal,presence=pcp)
        joblib.dump(dict(model=pres,calibrator=calibrator,fit_rows=fit,calibration_rows=cal,test_rows=test),
                    out/'models'/f'{fold}_presence.joblib',compress=3)
        if len(test):
            p=mix_probabilities(pt,parameters);ps=calibrated_presence(inductive_probabilities(pres,features[test]))
            oof[test]=p;presence_oof[test]=ps
            for name,prob in zip(MEMBERS,pt):base_oof[name][test]=prob
            r,q,gp=prediction_records(frame.iloc[test],p,salience[test],classes,settings,ps)
            r['fold']=fold;recs.append(r)
            np.savez_compressed(out/f'recording_probabilities_{fold}.npz',groups=r.group.to_numpy(str),species=q,genus=gp)
        else:
            bundle=dict(models=models,classes=classes.tolist(),settings=settings,presence_model=pres,
                        presence_calibrator=calibrator,feature_names=FEATURE_NAMES,created_utc=now(),
                        training_hashes=protocol['input_sha256'],fit_rows=fit,calibration_rows=cal)
            joblib.dump(bundle,out/'models/final_bundle.joblib',compress=3)
    rec=pd.concat(recs,ignore_index=True);rec.to_csv(out/'recording_predictions.csv',index=False)
    np.savez_compressed(out/'oof_probabilities.npz',row_indices=np.arange(len(frame)),groups=frame.group.to_numpy(str),
                        folds=frame.fold_species.to_numpy(),calibrated=oof,presence=presence_oof,**base_oof)
    windows=frame[['group','label','start_s','fold_species']].copy()
    windows['row_index']=np.arange(len(frame));windows['salience']=salience
    windows['predicted_species']=classes[oof.argmax(1)];windows['species_score']=oof.max(1)
    genera,gp=marginalize_genus(oof,classes);windows['genus_marginalized']=genera[gp.argmax(1)]
    windows['genus_posthoc']=pd.Series(windows.predicted_species).str.split().str[0]
    windows['presence_score']=presence_oof[:,1]
    windows.to_csv(out/'window_predictions.csv.gz',index=False)
    atomic_json(out/'partition_audit.json',dict(audits=audits,created_utc=now(),no_refit_on_calibration=True))
    atomic_json(out/'manifest.json',dict(completed_utc=now(),protocol_sha256=file_hash(out/'protocol.json'),
                input_sha256=protocol['input_sha256'],code_sha256=protocol['code_sha256'],
                models_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in (out/'models').glob('*.joblib')},
                calibration_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in out.glob('calibration_*.json')},release_allowed=False))
    frozen=list((out/'models').glob('*.joblib'))+list(out.glob('calibration_*.json'))+[
        out/p for p in ['protocol.json','manifest.json','external_plan.json','segments_v2.csv','features_v2.npz',
                       'partitions.csv','partition_audit.json','collision_audit.json']]
    atomic_json(out/'freeze.json',dict(created_utc=now(),artifacts_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in frozen},
                code_sha256={p:file_hash(ROOT/p) for p in SOURCE_FILES},primary_model='calibrated',
                no_v2_external_inference_yet=True,release_allowed=False))
    verify_freeze(out)
    atomic_json(out/'historical_hashes_after_training.json',verify_history(out/'historical_snapshot.json'))
    print('FROZEN',now(),flush=True)


def audio_probabilities(path,bundle):
    wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1);duration=len(wave)/sr
    if sr!=SR:
        d=np.gcd(sr,SR);wave=resample_poly(wave,SR//d,sr//d)
    n=len(wave)//SR
    if n<1:raise ValueError('No complete real 1 s window; no padding/repetition.')
    windows=wave[:n*SR].reshape(n,SR)
    quality=(np.max(np.abs(windows),axis=1)>=32/32768)&(np.mean(np.abs(windows)>=32700/32768,axis=1)<=.01)
    x,s=extract_features(windows)
    members=[calibrated_member(inductive_probabilities(model,x),temperature)
             for model,temperature in zip(bundle['models'],bundle['settings']['temperatures'])]
    p=mix_probabilities(members,bundle['settings']['parameters'])
    raw=inductive_probabilities(bundle['presence_model'],x)
    calibrator=bundle['presence_calibrator']
    presence=np.vstack([calibrator.predict_proba([[float(logit(np.clip(row[1],1e-6,1-1e-6)))]])[0] for row in raw])
    info=dict(duration_s=duration,complete_windows=n,original_sample_rate=int(sr),
              unanalysed_tail_s=duration-n,no_padding_or_repetition=True)
    return p,presence,s,quality,info


def evaluate(out=OUT):
    freeze=verify_freeze(out)
    if (out/'external_evaluation_audit.json').exists():raise FileExistsError('Evaluation already completed; no overwriting.')
    if not (out/'fresh_selection_v2.json').exists():raise ValueError('Acquire fresh audio after freeze before external evaluation.')
    plan=json.loads((out/'external_plan.json').read_text());fresh=json.loads((out/'fresh_selection_v2.json').read_text())
    if fresh['selected_utc']<=freeze['created_utc']:raise ValueError('Fresh search/selection must follow freeze.')
    bundle=joblib.load(out/'models/final_bundle.joblib');classes=np.asarray(bundle['classes'])
    from nested_species_ensemble import audio_probabilities as old_audio
    oldbundle=joblib.load(OLD/'models/final_bundle.joblib')
    allrecs=[];allwindows=[];probabilities={}
    scopes=[('external',plan['paired_files']),('drive',[dict(path='drive_audio.wav',group='private_drive',reference_species='Aedes aegypti',sha256=plan['private_file']['sha256'])]),
            ('fresh',fresh['accepted_files'])]
    for scope,items in scopes:
        for item in items:
            path=ROOT/item['path']
            if file_hash(path)!=item['sha256']:raise ValueError('Evaluation input changed.')
            p,pres,s,q,info=audio_probabilities(path,bundle)
            f=pd.DataFrame(dict(group=[item['group']]*len(p),label=[item['reference_species']]*len(p)))
            r,rp,rg=prediction_records(f,p,s,classes,bundle['settings'],pres,q,scope)
            r['path']=item['path']
            for k,v in info.items():r[k]=v
            r['invalid_windows']=int((~q).sum());r['forced_correct_windows']=int(((classes[p.argmax(1)]==item['reference_species'])&q).sum())
            # Raw private audio is never copied into results. Only predictions are saved.
            window=f.copy();window['scope']=scope;window['start_s']=np.arange(len(p));window['quality_ok']=q
            window['salience']=s;window['predicted_species']=classes[p.argmax(1)];window['species_score']=p.max(1)
            genera,gp=marginalize_genus(p,classes);window['genus_marginalized']=genera[gp.argmax(1)]
            window['genus_posthoc']=pd.Series(window.predicted_species).str.split().str[0]
            window['presence_score']=pres[:,1];window['detected']=q&(pres[:,1]>=bundle['settings']['threshold_presence']['threshold'])
            allwindows.append(window);allrecs.append(r)
            probabilities[item['group']]=dict(species=p,presence=pres,salience=s,quality=q,recording_species=rp,recording_genus=rg)
            if scope in ['drive','fresh']:
                op,oq,oi=old_audio(path,oldbundle);op=op['calibrated'];mean=op.mean(0)
                og,ogp=marginalize_genus(mean[None,:],oldbundle['classes'])
                r['old_predicted_species']=oldbundle['classes'][mean.argmax()]
                r['old_genus_posthoc']=r.old_predicted_species.str.split().str[0]
                r['old_genus_marginalized']=og[ogp.argmax(1)]
                r['old_quality_ok']=bool(oq.all())
                r['old_accepted_species']=bool(oq.all() and mean.max()>=oldbundle['thresholds']['calibrated']['threshold'])
                r['old_accepted_genus']=bool(oq.all())  # historical genus has no calibrated rejection
                r['old_forced_correct_windows']=int(((np.asarray(oldbundle['classes'])[op.argmax(1)]==item['reference_species'])&oq).sum())
            print('FROZEN INFERENCE',scope,Path(item['path']).name,flush=True)
    rec=pd.concat(allrecs,ignore_index=True);win=pd.concat(allwindows,ignore_index=True)
    rec.to_csv(out/'external_recording_predictions.csv',index=False);win.to_csv(out/'external_window_predictions.csv.gz',index=False)
    joblib.dump(probabilities,out/'external_probabilities.joblib',compress=3)
    verify_freeze(out)
    history=verify_history(out/'historical_snapshot.json')
    atomic_json(out/'historical_hashes_after_evaluation.json',history)
    atomic_json(out/'external_evaluation_audit.json',dict(evaluated_utc=now(),frozen_utc=freeze['created_utc'],
                freeze_sha256=file_hash(out/'freeze.json'),models_thresholds_unchanged=True,
                paired_recordings=int(rec.scope.eq('external').sum()),private_recordings=1,
                fresh_recordings=int(rec.scope.eq('fresh').sum()),release_allowed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','evaluate','verify'],required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='train':train()
        elif args.phase=='evaluate':evaluate()
        else:print(verify_freeze());print(verify_history(OUT/'historical_snapshot.json'))
