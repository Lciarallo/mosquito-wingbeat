"""Offline v3. Fit/calibrate every fold, freeze, then generate ANY test prediction.

Commands: --phase train | acquire | evaluate | compare | verify.
Historical files, partitions, reports, firmware and installers are read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.special import logit
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from acoustic_features_v3 import SR,FEATURE_NAMES,extract_features,window_features
from improve_device import ROOT,METHODS,model_for,fit_classical,file_hash,atomic_json
from nested_species_ensemble import now,fit_blend
from evaluate_nested_species import verify_freeze as verify_v1
import nested_species_ensemble_v2 as v2

OUT=ROOT/'results/species_nested_v3_2026_10_14'
V2=ROOT/'results/species_nested_2026_10_13'
SEED=20261014
MEMBERS=['svm','extra_trees','hist_gradient_boosting']
SOURCES=['acoustic_features_v3.py','nested_species_ensemble_v3.py','fetch_species_v3_fresh.py',
         'compare_species_v3.py','nested_species_ensemble_v2.py','acoustic_features_v2.py',
         'research_metrics_v2.py','nested_species_ensemble.py','evaluate_nested_species.py',
         'improve_device.py','fetch_humbug.py']


def verify_history(out=OUT):
    snapshot=json.loads((out/'historical_snapshot.json').read_text())
    for p,h in snapshot['files_sha256'].items():
        if not (ROOT/p).is_file() or file_hash(ROOT/p)!=h:
            raise ValueError('Historical artifact changed: '+p)
    verify_v1(ROOT/'results/species_nested_2026_10_09');v2.verify_freeze(V2)
    return dict(checked_utc=now(),unchanged=True,files=len(snapshot['files_sha256']),
                v1_verify_freeze='passed',v2_verify_freeze='passed',
                snapshot_sha256=file_hash(out/'historical_snapshot.json'))


def verify_freeze(out=OUT):
    freeze=json.loads((out/'freeze.json').read_text())
    for p,h in {**freeze['artifacts_sha256'],**freeze['code_sha256']}.items():
        if file_hash(ROOT/p)!=h:raise ValueError('Frozen artifact/code changed: '+p)
    return freeze


def check_partition(frame,fit,calibration,test):
    audit=v2.check_partition(frame,fit,calibration,test)
    blocks=[set(frame.iloc[np.asarray(ix,int)].pcm_block) for ix in [fit,calibration,test]]
    if blocks[0]&blocks[1] or blocks[0]&blocks[2] or blocks[1]&blocks[2]:
        raise ValueError('pcm_block leakage between fit/calibration/test.')
    return audit


def select_genus_threshold(labels,p,genera,partition,pcm_blocks=None):
    if partition!='calibration':raise ValueError('Genus threshold requires calibration partition.')
    p=np.asarray(p,dtype=float);y=np.asarray(labels).astype(str);genera=np.asarray(genera).astype(str)
    if p.ndim!=2 or len(y)==0 or len(p)!=len(y) or p.shape[1]!=len(genera) or len(genera)<2:
        raise ValueError('Invalid genus calibration dimensions.')
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError('Invalid genus probabilities.')
    floor=1/len(genera)+.05
    scores=p.max(1);correct=genera[p.argmax(1)]==y
    blocks=np.arange(len(y)) if pcm_blocks is None else np.asarray(pcm_blocks)
    if len(blocks)!=len(y):raise ValueError('Misaligned PCM blocks in calibration.')
    curve=[];eligible=[]
    for t in np.unique(np.r_[floor,scores[scores>=floor],1.000001]):
        a=scores>=t;n=int(a.sum());nb=len(np.unique(blocks[a]));cov=float(a.mean())
        accuracy=float(correct[a].mean()) if n else None
        objective=float(accuracy*np.sqrt(cov)) if n else 0.
        row=dict(threshold=float(t),coverage=cov,accepted_accuracy=accuracy,
                 accepted_recordings=n,accepted_pcm_blocks=nb,objective=objective,
                 eligible=bool(n and nb>=20 and accuracy>=.90))
        curve.append(row)
        if row['eligible']:eligible.append(row)
    if eligible:
        best=max(eligible,key=lambda r:(r['objective'],r['accepted_accuracy'],-r['threshold']))
    else:
        best=curve[-1]  # No qualified threshold: reject all, never fall back to zero.
    return dict(threshold=best['threshold'],minimum_confidence=floor,objective_name='accuracy * sqrt(coverage)',
                objective=best['objective'],calibration_accuracy=best['accepted_accuracy'],
                calibration_emitted=best['accepted_recordings'],
                calibration_emitted_pcm_blocks=best['accepted_pcm_blocks'],
                calibration_groups=len(y),target=.90,minimum_pcm_blocks=20,
                qualified=bool(eligible),selection_partition='calibration',curve=curve)


def _feature_chunk(waves):
    with threadpool_limits(limits=1):return extract_features(waves)


def prepare_features(frame,out,workers=4):
    waves=np.load(ROOT/'data/cache/waves.npy',mmap_mode='r')
    if len(waves)!=len(frame):raise ValueError('Cached PCM row identity differs.')
    x=np.empty((len(frame),len(FEATURE_NAMES)),np.float32);s=np.empty(len(frame))
    size=128
    chunks=(np.array(waves[i:i+size]) for i in range(0,len(waves),size))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done=0
        for features,salience in pool.map(_feature_chunk,chunks,chunksize=1):
            n=len(features);x[done:done+n]=features;s[done:done+n]=salience;done+=n
            if done%2048==0 or done==len(frame):print('V3 FEATURES',done,'/',len(frame),flush=True)
    np.savez_compressed(out/'features_v3.npz',features=x,salience=s,row_indices=np.arange(len(frame)),
                        pcm_blocks=frame.pcm_block.to_numpy(str))
    return x,s


def load_inputs(out):
    v2.verify_freeze(V2)
    frame=pd.read_csv(V2/'segments_v2.csv');frame.phone=frame.phone.fillna('unknown')
    frame['pcm_block']=frame.collision_group
    if frame.pcm_block.nunique()!=560:raise ValueError('Expected exactly the 560 audited PCM blocks.')
    for row in pd.read_csv(V2/'source_pcm_audit.csv').itertuples():
        if file_hash(ROOT/row.path)!=row.file_sha256:raise ValueError('Audited source bytes changed: '+row.path)
    parts=pd.read_csv(V2/'partitions.csv',dtype={'fold':str})
    parts['pcm_block']=parts.collision_group
    audits=[]
    for fold in ['0','1','2','final']:
        sub=parts[parts.fold.eq(fold)]
        ix=[sub[sub.partition.eq(p)].row_index.to_numpy(int) for p in ['fit','calibration','test']]
        audit=check_partition(frame,*ix);audits.append(dict(fold=fold,**audit,pcm_block_overlap=0))
    frame.to_csv(out/'segments_v3.csv',index=False);parts.to_csv(out/'partitions.csv',index=False)
    atomic_json(out/'partition_audit.json',dict(created_utc=now(),audits=audits,pcm_blocks=560,
        exact_v2_partition_reuse=True,source_sha256=file_hash(V2/'partitions.csv'),
        limits='Existing whole/aligned 1/.992/.3-second PCM graph; byte identity reverified. Not biological or session independence.'))
    return frame,parts


def presence_probabilities(raw,calibrator):
    return np.vstack([calibrator.predict_proba([[float(logit(np.clip(row[1],1e-6,1-1e-6)))]])[0] for row in raw])


def external_plan(out):
    plan=json.loads((V2/'external_plan.json').read_text())
    fresh=json.loads((V2/'fresh_selection_v2.json').read_text())['accepted_files']
    if len(plan['paired_files'])!=17 or len(fresh)!=2:raise ValueError('Expected exact 17 + 2 paired files.')
    private=dict(path='drive_audio.wav',group='private_drive',reference_species='Aedes aegypti',sha256=plan['private_file']['sha256'])
    for item in plan['paired_files']+fresh+[private]:
        if file_hash(ROOT/item['path'])!=item['sha256']:raise ValueError('Paired audio changed.')
    result=dict(created_utc=now(),retrospective=True,paired_files=plan['paired_files'],
                fresh_previous=fresh,private_file=private,
                new_acquisition_rule='After freeze and web search: first 2 unused numeric D3 Aedes aegypti IDs with real duration >=1 s; scan <=30; reject known byte/aligned PCM collisions before predictions; no replacement after collision exclusion.')
    atomic_json(out/'external_plan.json',result)
    return result


def train(out=OUT,workers=4):
    if (out/'protocol.json').exists():raise FileExistsError('Training protocol exists; never overwrite an experiment.')
    before=verify_history(out);atomic_json(out/'historical_hashes_before.json',before)
    frame,parts=load_inputs(out);plan=external_plan(out)
    classes=np.sort(frame[frame.label.ne('noise')].label.unique())
    if len(classes)!=20:raise ValueError('Expected twenty species.')
    protocol=dict(created_utc=now(),round='v3_2026_10_14',classes=classes.tolist(),members=MEMBERS,
        primary_model='calibrated',pcm_blocks=560,original_recording_groups=613,window_seconds=1,
        partition_rule='Exact v2 fit/calibration/outer test rows and folds; no new split selection.',
        refit_on_calibration=False,predictions_on_outer_or_external_before_freeze=False,
        cepstral_rule='Intra-window scalar median log-mel envelope centring; c0=0; c1/c2 and other means retained. No recording-wide CMN.',
        psd_rule='Subtract local moving median noise floor (31 long-STFT bins spanning 150 Hz); no spectral slope regression; gain-only centring.',
        stft=dict(short_samples=1600,short_hop=160,long_samples=3200,long_hop=320,
                  short_bin_spacing_hz=10,long_bin_spacing_hz=5,no_zero_padding=True),
        feature_names=FEATURE_NAMES,feature_workers=workers,prediction_threads=1,
        feature_inductive=True,arbitrary_channel_invariance_proven=False,
        weights='v2 fit-only class/device/PCM-block weights times v3 per-window salience [.1,3]',
        pooling='v2 MIL; top fractions .25/.5/1 selected only in calibration',
        member_temperature_log_bounds=[-1.6,1.6],blend_prior_regularisation=.01,
        genus_rule='Exact sum of calibrated species probabilities; floor=1/N_genera+.05; maximise accuracy*sqrt(coverage), accuracy>=.90 and >=20 accepted PCM blocks.',
        species_threshold_rule='v2 calibration target .90, minimum 20 accepted PCM blocks',
        presence_rule='v2 HGB and Platt, calibration FPR<=.05; recorded separately',
        calibration_estimator_seed=20261013,base_estimator_seed=42,bootstrap_seed=SEED,
        calibration_reuse='Temperature/blend/pooling/rejection reuse inner calibration; optimism there remains possible. Outer test excluded from all selection.',
        quality_rule='v2 raw peak/clipping check; invalid windows remain in denominator, only valid MIL-selected windows vote; all-invalid recording cannot emit.',
        taxonomic_primary_no_presence_gate=True,additional_presence_gate_saved=True,
        external_retrospective=True,external_plan_sha256=file_hash(out/'external_plan.json'),
        input_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'data/cache/waves.npy',V2/'segments_v2.csv',V2/'partitions.csv',V2/'source_pcm_audit.csv',ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_full.json']},
        code_sha256={p:file_hash(ROOT/p) for p in SOURCES},
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        release_allowed=False,arduino_compiled=False,private_raw_audio_published=False)
    atomic_json(out/'protocol.json',protocol)
    x,s=prepare_features(frame,out,workers)
    labels=np.full(len(frame),-1);positive=frame.label.ne('noise').to_numpy();labels[positive]=np.searchsorted(classes,frame[positive].label)
    (out/'models').mkdir(exist_ok=False)
    for fold in ['0','1','2','final']:
        sub=parts[parts.fold.eq(fold)]
        fit,cal,test=[sub[sub.partition.eq(p)].row_index.to_numpy(int) for p in ['fit','calibration','test']]
        fitsp=fit[positive[fit]];calsp=cal[positive[cal]]
        w=v2.training_weights(fitsp,frame,s)
        cw=v2.training_weights(calsp,frame,np.ones(len(frame)),device_balance=False)
        members=[];pc=[];temperatures=[];timings=[]
        for k,name in enumerate(MEMBERS):
            print('V3 FIT',fold,name,len(fitsp),'fit',len(calsp),'cal; NO TEST PREDICTIONS',flush=True)
            start=time.monotonic();model=fit_classical(model_for(METHODS[k]),x[fitsp],labels[fitsp],w)
            raw=v2.inductive_probabilities(model,x[calsp]);temp=v2._temperature(raw,labels[calsp],cw)
            pc.append(v2.calibrated_member(raw,temp));temperatures.append(temp);members.append(model)
            timings.append(dict(member=name,seconds=time.monotonic()-start))
            joblib.dump(dict(model=model,classes=classes.tolist(),fit_rows=fitsp,calibration_rows=calsp,
                             test_rows=test[positive[test]],temperature=temp),out/'models'/f'{fold}_{name}.joblib',compress=3)
        parameters=fit_blend(np.stack(pc),labels[calsp],cw,partition='calibration')
        cp=v2.mix_probabilities(pc,parameters)
        pooling=v2.choose_pooling(frame.iloc[calsp],cp,s[calsp],classes,'calibration')
        cr,cq=v2.recording_probabilities(frame.iloc[calsp],cp,s[calsp],pooling['top_fraction'])
        genera,cg=v2.marginalize_genus(cq,classes)
        thresholds=dict(threshold_species=v2.select_threshold(cr.label,cq,classes,'calibration',cr.collision_group),
                        threshold_genus=select_genus_threshold(cr.label.str.split().str[0],cg,genera,'calibration',cr.collision_group))
        print('V3 FIT',fold,'presence',len(fit),flush=True)
        binary=frame.copy();binary['label']=np.where(positive,'mosquito','noise')
        pres=fit_classical(model_for(METHODS[2]),x[fit],positive[fit].astype(int),
                           v2.training_weights(fit,binary,np.where(positive,s,1.)))
        rawpres=v2.inductive_probabilities(pres,x[cal])
        pw=v2.training_weights(cal,binary,np.ones(len(frame)),device_balance=False)
        calibrator=LogisticRegression(C=100,random_state=20261013).fit(
            logit(np.clip(rawpres[:,1],1e-6,1-1e-6))[:,None],positive[cal].astype(int),sample_weight=pw)
        pp=presence_probabilities(rawpres,calibrator)
        prespool=v2.choose_pooling(binary.iloc[cal],pp,s[cal],np.array(['noise','mosquito']),'calibration')
        pr,pq=v2.recording_probabilities(frame.iloc[cal],pp,s[cal],prespool['top_fraction'])
        settings=dict(pooling=pooling,presence_pooling=prespool,threshold_presence=v2._presence_rule(pr,pq),
                      temperatures=temperatures,parameters=parameters,timings=timings,**thresholds)
        atomic_json(out/f'calibration_{fold}.json',settings)
        np.savez_compressed(out/f'calibration_probabilities_{fold}.npz',row_indices=calsp,members=np.stack(pc),
                            blended=cp,presence_rows=cal,presence=pp)
        bundle=dict(models=members,classes=classes.tolist(),settings=settings,presence_model=pres,
                    presence_calibrator=calibrator,fit_rows=fit,calibration_rows=cal,test_rows=test,
                    feature_names=FEATURE_NAMES,created_utc=now(),fold=fold)
        joblib.dump(bundle,out/'models'/f'{fold}_bundle.joblib',compress=3)
        print('V3 CALIBRATED',fold,'GENUS THRESHOLD',settings['threshold_genus']['threshold'],flush=True)
    frozen=list((out/'models').glob('*.joblib'))+list(out.glob('calibration_*.json'))+list(out.glob('calibration_probabilities_*.npz'))
    frozen+=[out/p for p in ['protocol.json','features_v3.npz','partitions.csv','partition_audit.json','segments_v3.csv','external_plan.json']]
    atomic_json(out/'manifest.json',dict(completed_training_utc=now(),models_thresholds_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in frozen},
                code_sha256=protocol['code_sha256'],input_sha256=protocol['input_sha256'],no_test_predictions_yet=True,release_allowed=False))
    frozen.append(out/'manifest.json')
    if any(file_hash(ROOT/p)!=h for p,h in protocol['code_sha256'].items()):raise ValueError('Code changed during training.')
    atomic_json(out/'freeze.json',dict(created_utc=now(),artifacts_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in frozen},
                code_sha256=protocol['code_sha256'],all_outer_folds_and_final_frozen=True,no_test_predictions_yet=True,release_allowed=False))
    verify_freeze(out);atomic_json(out/'historical_hashes_after_training.json',verify_history(out))
    print('V3 ALL MODELS FROZEN',now(),flush=True)


def predict_features(x,bundle):
    members=[v2.calibrated_member(v2.inductive_probabilities(m,x),t)
             for m,t in zip(bundle['models'],bundle['settings']['temperatures'])]
    p=v2.mix_probabilities(members,bundle['settings']['parameters'])
    pp=presence_probabilities(v2.inductive_probabilities(bundle['presence_model'],x),bundle['presence_calibrator'])
    return p,pp,np.stack(members)


def audio_probabilities(path,bundle):
    wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1);duration=len(wave)/sr
    if sr!=SR:
        d=np.gcd(sr,SR);wave=resample_poly(wave,SR//d,sr//d)
    n=len(wave)//SR
    if n<1:raise ValueError('No complete real one-second window; no padding.')
    waves=wave[:n*SR].reshape(n,SR)
    quality=(np.abs(waves).max(1)>=32/32768)&(np.mean(np.abs(waves)>=32700/32768,axis=1)<=.01)
    x,s=extract_features(waves);p,pp,_=predict_features(x,bundle)
    return p,pp,s,quality,dict(duration_s=duration,complete_windows=n,original_sample_rate=int(sr),
                             unanalysed_tail_s=duration-n,no_padding_or_repetition=True)


def window_predictions(frame,p,pp,s,quality,bundle,scope):
    r=frame[['group','label']].reset_index(drop=True).copy()
    classes=np.array(bundle['classes']);genera,gp=v2.marginalize_genus(p,classes)
    r['scope']=scope;r['quality_ok']=quality;r['salience']=s
    r['start_s']=frame.start_s.to_numpy() if 'start_s' in frame else np.arange(len(p))
    r['predicted_species']=classes[p.argmax(1)];r['species_score']=p.max(1)
    r['genus_marginalized']=genera[gp.argmax(1)];r['genus_posthoc']=r.predicted_species.str.split().str[0]
    r['presence_score']=pp[:,1];r['detected']=quality&(pp[:,1]>=bundle['settings']['threshold_presence']['threshold'])
    return r


def evaluate(out=OUT):
    freeze=verify_freeze(out)
    if (out/'evaluation_audit.json').exists():raise FileExistsError('Completed evaluations cannot be overwritten.')
    novel=json.loads((out/'fresh_selection_v3.json').read_text())
    if novel['selected_utc']<=freeze['created_utc']:raise ValueError('New acquisition must follow freeze.')
    atomic_json(out/'evaluation_started.json',dict(started_utc=now(),frozen_utc=freeze['created_utc'],freeze_sha256=file_hash(out/'freeze.json')))
    frame=pd.read_csv(out/'segments_v3.csv');parts=pd.read_csv(out/'partitions.csv',dtype={'fold':str})
    a=np.load(out/'features_v3.npz');x,s=a['features'],a['salience']
    oof=np.zeros((len(frame),20));poof=np.zeros((len(frame),2));base=np.zeros((3,len(frame),20));assigned=np.zeros(len(frame),int)
    recs=[];windows=[]
    for fold in ['0','1','2']:
        test=parts[parts.fold.eq(fold)&parts.partition.eq('test')].row_index.to_numpy(int)
        bundle=joblib.load(out/'models'/f'{fold}_bundle.joblib')
        if not np.array_equal(bundle['test_rows'],test):raise ValueError('Frozen test row identity changed.')
        p,pp,member=predict_features(x[test],bundle);oof[test]=p;poof[test]=pp;base[:,test]=member;assigned[test]+=1
        quality=np.ones(len(test),bool)
        r,rp,rg=v2.prediction_records(frame.iloc[test],p,s[test],np.array(bundle['classes']),bundle['settings'],pp,quality,'dryad')
        r['fold']=fold;recs.append(r)
        w=window_predictions(frame.iloc[test],p,pp,s[test],quality,bundle,'dryad');w['row_index']=test;windows.append(w)
        np.savez_compressed(out/f'oof_recording_probabilities_{fold}.npz',groups=r.group.to_numpy(str),species=rp,genus=rg)
        print('V3 FROZEN OOF',fold,len(test),'windows',flush=True)
    if not np.all(assigned==1):raise ValueError('OOF identity missing or duplicated.')
    np.savez_compressed(out/'oof_probabilities.npz',row_indices=np.arange(len(frame)),groups=frame.group.to_numpy(str),
                        folds=frame.fold_species.to_numpy(),calibrated=oof,presence=poof,members=base)
    pd.concat(recs,ignore_index=True).to_csv(out/'recording_predictions.csv',index=False)
    pd.concat(windows,ignore_index=True).to_csv(out/'window_predictions.csv.gz',index=False)
    plan=json.loads((out/'external_plan.json').read_text());bundle=joblib.load(out/'models/final_bundle.joblib')
    recs=[];windows=[];probabilities={}
    scopes=[('external',plan['paired_files']),('fresh',plan['fresh_previous']),('drive',[plan['private_file']]),
            ('postfreeze_new',novel['accepted_files'])]
    for scope,items in scopes:
        for item in items:
            path=ROOT/item['path']
            if file_hash(path)!=item['sha256']:raise ValueError('Evaluation audio changed.')
            p,pp,s,q,info=audio_probabilities(path,bundle)
            f=pd.DataFrame(dict(group=[item['group']]*len(p),label=[item['reference_species']]*len(p)))
            r,rp,rg=v2.prediction_records(f,p,s,np.array(bundle['classes']),bundle['settings'],pp,q,scope)
            r['path']=item['path'];r['invalid_windows']=int((~q).sum())
            r['forced_correct_windows']=int(((np.array(bundle['classes'])[p.argmax(1)]==item['reference_species'])&q).sum())
            for k,value in info.items():r[k]=value
            recs.append(r);windows.append(window_predictions(f,p,pp,s,q,bundle,scope))
            probabilities[item['group']]=dict(species=p,presence=pp,salience=s,quality=q,recording_species=rp,recording_genus=rg)
            print('V3 FROZEN EXTERNAL',scope,path.name,flush=True)
    pd.concat(recs,ignore_index=True).to_csv(out/'external_recording_predictions.csv',index=False)
    pd.concat(windows,ignore_index=True).to_csv(out/'external_window_predictions.csv.gz',index=False)
    joblib.dump(probabilities,out/'external_probabilities.joblib',compress=3)
    verify_freeze(out);history=verify_history(out)
    atomic_json(out/'historical_hashes_after_evaluation.json',history)
    atomic_json(out/'evaluation_audit.json',dict(completed_utc=now(),frozen_utc=freeze['created_utc'],
        freeze_sha256=file_hash(out/'freeze.json'),models_thresholds_unchanged=True,outer_predictions_after_freeze=True,
        paired_external=17,previous_fresh=2,private_recordings=1,postfreeze_new=len(novel['accepted_files']),release_allowed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','acquire','evaluate','compare','verify'],required=True)
    parser.add_argument('--feature-workers',type=int,default=4);args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='train':train(workers=args.feature_workers)
        elif args.phase=='acquire':
            from fetch_species_v3_fresh import acquire
            acquire()
        elif args.phase=='evaluate':evaluate()
        elif args.phase=='compare':
            from compare_species_v3 import main
            main()
        else:print(verify_freeze());print(verify_history())
