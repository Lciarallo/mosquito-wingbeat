"""Offline v4. Fit/calibrate every fold, freeze, then generate ANY test prediction.

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
from scipy.special import logit, softmax, logsumexp
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from acoustic_features_v4 import SR,FEATURE_NAMES,extract_features,window_features
from improve_device import ROOT,METHODS,model_for,fit_classical,file_hash,atomic_json
from nested_species_ensemble import now,fit_blend
from evaluate_nested_species import verify_freeze as verify_v1
import nested_species_ensemble_v2 as v2

OUT=ROOT/'results/species_nested_v4_2026_10_14'
V2=ROOT/'results/species_nested_2026_10_13'
V3=ROOT/'results/species_nested_v3_2026_10_14'
SEED=20261014
MEMBERS=['svm','extra_trees','hist_gradient_boosting']
SOURCES=['acoustic_features_v4.py','nested_species_ensemble_v4.py','fetch_species_v4_fresh.py',
         'compare_species_v4.py','nested_species_ensemble_v2.py','acoustic_features_v2.py',
         'research_metrics_v2.py','nested_species_ensemble.py','evaluate_nested_species.py',
         'improve_device.py','fetch_humbug.py','acoustic_features_v3.py','nested_species_ensemble_v3.py',
         'compare_species_v3.py','fetch_species_v3_fresh.py','research_metrics_v2.py']


def marginalize_genus_v4(species_probs, classes, gamma, tau):
    """A calibrated power aggregation, NOT the exact genus posterior sum.

    gamma=0,tau=1 recovers exact marginalisation. gamma=1 gives a uniform
    genus null under uniform species evidence. Each row is independent.
    """
    p=np.asarray(species_probs,dtype=float)
    classes=np.asarray(classes,str)
    if p.ndim!=2 or p.shape[1]!=len(classes) or not 0<=gamma<=1 or not 1<=tau<=3:
        raise ValueError('Invalid taxonomy/probability dimensions or gamma/tau.')
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-8):
        raise ValueError('Invalid species probability distribution.')
    membership=np.array([c.split()[0] for c in classes]);genera=np.unique(membership)
    masks=[membership==g for g in genera]
    q=np.vstack([softmax(np.array([
        (logsumexp(tau*np.log(np.maximum(row[mask],1e-300)))-gamma*np.log(mask.sum()))/tau
        for mask in masks])) for row in p])
    return genera,q


def select_genus_parameters(labels,p,classes,partition,pcm_blocks=None):
    if partition!='calibration':raise ValueError('Taxonomy selection requires calibration.')
    y=np.asarray([str(label).split()[0] for label in labels])
    genera=np.unique([str(c).split()[0] for c in classes]);floor=1/len(genera)+.05
    blocks=np.arange(len(y)) if pcm_blocks is None else np.asarray(pcm_blocks)
    if len(y)!=len(p) or len(blocks)!=len(y) or not len(y):raise ValueError('Misaligned taxonomy calibration.')
    counts=pd.DataFrame({'label':y,'block':blocks}).drop_duplicates().groupby('label').size()
    sizes=pd.Series(list(zip(y,blocks))).value_counts()
    weights=np.array([1/(counts[g]*sizes[(g,b)]) for g,b in zip(y,blocks)],float)
    yi=np.searchsorted(genera,y);uniform=np.full((1,len(classes)),1/len(classes));curve=[]
    for gamma in (0.,.25,.5,.75,1.):
        for tau in (1.,1.5,2.,3.):
            _,q=marginalize_genus_v4(p,classes,gamma,tau)
            _,null=marginalize_genus_v4(uniform,classes,gamma,tau)
            loss=float(np.average(-np.log(np.maximum(q[np.arange(len(y)),yi],1e-12)),weights=weights))
            ba=float(balanced_accuracy_score(y,genera[q.argmax(1)],sample_weight=weights))
            penalty=float(.05*np.sum(null[0]*np.log(null[0]*len(genera))))
            objective=.5*(ba+np.exp(-loss))-penalty
            curve.append(dict(gamma=gamma,tau=tau,balanced_accuracy=ba,balanced_log_loss=loss,
                uniform_null_max=float(null.max()),uniform_null_distribution=null[0].tolist(),
                uniform_null_penalty=penalty,objective=float(objective),eligible=bool(null.max()<floor)))
    admissible=[r for r in curve if r['eligible']]
    if not admissible:raise ValueError('No taxonomy rule rejects uniform species uncertainty.')
    best=max(admissible,key=lambda r:(r['objective'],r['gamma'],-r['tau']))
    return dict(**best,candidates=curve,selection_partition='calibration',minimum_confidence=floor,
        objective_name='.5*(genus balanced accuracy + exp(-genus balanced log loss)) - .05*KL(uniform-species genus null || uniform genera)',
        exact_probabilistic_marginalization=bool(best['gamma']==0 and best['tau']==1))


def pool_recording(p,salience,top_fraction=.35,quality=None,presence_prob=None):
    p=np.asarray(p,float);s=np.asarray(salience,float)
    n=len(p)
    if not n or p.ndim!=2 or p.shape[1]<2 or len(s)!=n or not 0<top_fraction<=1:
        raise ValueError('Empty/invalid recording bag.')
    if n>4 and top_fraction>.65:raise ValueError('Long bags require genuinely selective MIL, <=65%.')
    pres=np.ones(n) if presence_prob is None else np.asarray(presence_prob,float)
    valid=np.ones(n,bool) if quality is None else np.asarray(quality,bool)
    if pres.shape!=(n,) or valid.shape!=(n,) or not np.isfinite(pres).all() or np.any((pres<0)|(pres>1)):
        raise ValueError('Invalid or misaligned presence/quality.')
    if not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(s<0) or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-8):
        raise ValueError('Invalid MIL probabilities/salience.')
    sharpness=np.maximum(1+np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)/np.log(p.shape[1]),0)
    sharpness[sharpness<1e-12]=0.  # Uniform entropy must not vote through roundoff.
    weights=s*sharpness*(.15+.85*pres)*valid
    count=n if n<=4 else min(n-1,max(1,int(np.ceil(n*top_fraction))))
    ix=np.argsort(-weights,kind='stable')[:count]
    total=float(weights[ix].sum())
    info=dict(selected_windows=count,weighted_windows=int((weights[ix]>0).sum()),quality_ok=bool(total>1e-15),
        selected_weight_sum=total,presence_weighting=presence_prob is not None)
    if total<=1e-15:return p.mean(0),info
    q=np.sum(p[ix]*weights[ix,None],axis=0)/total
    return q/q.sum(),info


def recording_probabilities(frame,p,salience,fraction,quality=None,presence_prob=None):
    p=np.asarray(p);salience=np.asarray(salience)
    if len(frame)!=len(p) or len(salience)!=len(p):raise ValueError('Misaligned recording rows.')
    if quality is not None:quality=np.asarray(quality)
    if presence_prob is not None:presence_prob=np.asarray(presence_prob)
    rows=[];probabilities=[]
    for group,ix in frame.groupby('group',sort=True).indices.items():
        sub=frame.iloc[ix]
        if sub.label.nunique()!=1:raise ValueError('Conflicting recording labels.')
        q,info=pool_recording(p[ix],salience[ix],fraction,
            None if quality is None else quality[ix],None if presence_prob is None else presence_prob[ix])
        rows.append(dict(group=group,label=sub.label.iloc[0],
            collision_group=sub.collision_group.iloc[0] if 'collision_group' in sub else group,**info))
        probabilities.append(q)
    return pd.DataFrame(rows),np.stack(probabilities)


def choose_pooling(frame,p,salience,classes,partition,presence_prob=None):
    if partition!='calibration':raise ValueError('MIL selection requires calibration.')
    from research_metrics_v2 import classification_metrics
    candidates=[]
    for fraction in (.20,.35,.50,.65):
        r,q=recording_probabilities(frame,p,salience,fraction,presence_prob=presence_prob)
        m=classification_metrics(r.label,np.asarray(classes)[q.argmax(1)])
        candidates.append(dict(top_fraction=fraction,**m))
    best=max(candidates,key=lambda r:(r['balanced_accuracy'],r['macro_f1_strict_union'],-r['top_fraction']))
    return dict(top_fraction=best['top_fraction'],candidates=candidates,partition='calibration',
        objective='recording balanced accuracy, strict F1, smaller fraction tie-break',
        long_bag_rule='>4 windows: <=65%, ceil bounded to n-1; <=4 windows: all',presence_weighting=presence_prob is not None)


def mix_probabilities_v4(members,parameters):
    p=np.asarray(members,float);w=np.asarray(parameters['weights']);bias=np.asarray(parameters['log_prior'])
    temperature=float(parameters['blend_temperature'])
    if not temperature>0 or bias.shape!=(p.shape[2],) or w.shape!=(p.shape[0],):raise ValueError('Invalid blend settings.')
    return np.vstack([softmax(np.log(np.maximum(np.sum(p[:,i,:]*w[:,None],axis=0),1e-12))/temperature+bias)
                      for i in range(p.shape[1])])


def fit_blend_v4(p,y,sample_weight,frame,salience,presence_prob,classes,*,partition):
    if partition!='calibration':raise ValueError('Prior/temperature selection requires calibration.')
    p=np.asarray(p,float);y=np.asarray(y,int);w=np.asarray(sample_weight,float)
    if p.ndim!=3 or p.shape[1]!=len(y) or len(frame)!=len(y) or w.shape!=y.shape or np.any(w<=0):
        raise ValueError('Misaligned calibration mixture.')
    w=w/w.sum();m,n,k=p.shape
    def loss_gradient(z):
        mixw=softmax(z);mixture=np.maximum(np.einsum('m,mnc->nc',mixw,p),1e-12)
        loss=-np.sum(w*np.log(mixture[np.arange(n),y]))
        gw=-np.sum(w[None,:]*p[:,np.arange(n),y]/mixture[np.arange(n),y][None,:],axis=1)
        return loss,mixw*(gw-np.dot(gw,mixw))
    fit=minimize(loss_gradient,np.zeros(m),jac=True,method='L-BFGS-B',bounds=[(-6,6)]*m)
    if not fit.success:raise RuntimeError('Calibration mixture optimiser failed: '+fit.message)
    mixw=softmax(fit.x)
    counts=frame[['label','collision_group']].drop_duplicates().groupby('label').size().reindex(classes).to_numpy(float)
    if not np.isfinite(counts).all() or np.any(counts<=0):raise ValueError('Calibration must represent all species.')
    base=-np.log(counts);base=np.clip(base-base.mean(),-.8,.8)
    candidates=[];best=None
    for temperature in (.75,1.,1.25,1.5,2.):
        for alpha in (0.,.25,.5,.75,1.):
            settings=dict(weights=mixw.tolist(),log_prior=(alpha*base).tolist(),blend_temperature=temperature)
            cp=mix_probabilities_v4(p,settings)
            r,q=recording_probabilities(frame,cp,salience,.35,presence_prob=presence_prob)
            target=np.searchsorted(classes,r.label.to_numpy());hot=np.eye(k)[target]
            blockcounts=r[['label','collision_group']].drop_duplicates().groupby('label').size()
            duplicates=r.groupby(['label','collision_group']).size()
            rw=np.array([1/(blockcounts[label]*duplicates.loc[(label,block)]) for label,block in zip(r.label,r.collision_group)])
            rw/=rw.sum()
            ba=float(balanced_accuracy_score(r.label,np.asarray(classes)[q.argmax(1)],sample_weight=rw))
            tp=np.sum(rw[:,None]*hot*q,axis=0);support=np.sum(rw[:,None]*hot,axis=0);predicted=np.sum(rw[:,None]*q,axis=0)
            soft_f1=float(np.mean(2*tp/np.maximum(support+predicted,1e-12)))
            ll=float(np.sum(rw*-np.log(np.maximum(q[np.arange(len(r)),target],1e-12))))
            objective=.5*(ba+soft_f1)
            row=dict(blend_temperature=temperature,prior_strength=alpha,balanced_accuracy=ba,
                     soft_macro_f1=soft_f1,balanced_log_loss=ll,objective=objective)
            candidates.append(row);key=(objective,-ll,-alpha,-abs(temperature-1))
            if best is None or key>best[0]:best=(key,settings,row)
    return dict(**best[1],prior_strength=best[2]['prior_strength'],candidates=candidates,
        calibration_pcm_blocks_per_species=counts.tolist(),base_balanced_log_prior=base.tolist(),
        maximum_absolute_log_prior=.8,selection_partition='calibration',provisional_top_fraction=.35,
        objective='.5*(recording balanced accuracy + PCM/class-weighted soft macro F1)',
        mixture_log_loss=float(fit.fun),converged=True,geographic_prior=False)


def prediction_records(frame,p,salience,classes,settings,presence_p,quality=None,scope='dryad'):
    r,q=recording_probabilities(frame,p,salience,settings['pooling']['top_fraction'],quality,presence_p[:,1])
    pr,pq=recording_probabilities(frame,presence_p,salience,settings['presence_pooling']['top_fraction'],quality)
    if not np.array_equal(r.group,pr.group):raise ValueError('Recording identity differs between tasks.')
    g=settings['genus_parameters'];genera,gp=marginalize_genus_v4(q,classes,g['gamma'],g['tau'])
    _,raw=v2.marginalize_genus(q,classes);species=np.asarray(classes)[q.argmax(1)]
    r['scope']=scope;r['predicted_species']=species;r['species_score']=q.max(1)
    r['true_genus']=r.label.str.split().str[0];r['genus_posthoc']=pd.Series(species).str.split().str[0].to_numpy()
    r['genus_marginalized']=genera[gp.argmax(1)];r['genus_score']=gp.max(1)
    r['genus_raw_sum']=genera[raw.argmax(1)];r['genus_raw_score']=raw.max(1)
    r['accepted_species']=r.quality_ok&(q.max(1)>=settings['threshold_species']['threshold'])
    r['accepted_genus']=r.quality_ok&(gp.max(1)>=settings['threshold_genus']['threshold'])
    r['presence_score']=pq[:,1];r['detected']=pr.quality_ok&(pq[:,1]>=settings['threshold_presence']['threshold'])
    r['accepted_species_gated']=r.accepted_species&r.detected;r['accepted_genus_gated']=r.accepted_genus&r.detected
    return r,q,gp


def verify_history(out=OUT):
    snapshot=json.loads((out/'historical_snapshot.json').read_text())
    for p,h in snapshot['files_sha256'].items():
        if not (ROOT/p).is_file() or file_hash(ROOT/p)!=h:
            raise ValueError('Historical artifact changed: '+p)
    verify_v1(ROOT/'results/species_nested_2026_10_09');v2.verify_freeze(V2)
    from nested_species_ensemble_v3 import verify_freeze as verify_v3
    verify_v3(V3)
    return dict(checked_utc=now(),unchanged=True,files=len(snapshot['files_sha256']),
                v1_verify_freeze='passed',v2_verify_freeze='passed',v3_verify_freeze='passed',
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
            if done%2048==0 or done==len(frame):print('V4 FEATURES',done,'/',len(frame),flush=True)
    np.savez_compressed(out/'features_v4.npz',features=x,salience=s,row_indices=np.arange(len(frame)),
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
    frame.to_csv(out/'segments_v4.csv',index=False);parts.to_csv(out/'partitions.csv',index=False)
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
    for item in plan['paired_files']+fresh+[private]+json.loads((V3/'fresh_selection_v3.json').read_text())['accepted_files']:
        if file_hash(ROOT/item['path'])!=item['sha256']:raise ValueError('Paired audio changed.')
    result=dict(created_utc=now(),retrospective=True,paired_files=plan['paired_files'],
                fresh_previous=fresh,private_file=private,postfreeze_previous=json.loads((V3/'fresh_selection_v3.json').read_text())['accepted_files'],
                new_acquisition_rule='After freeze and web search: first 2 unused numeric D3 Aedes aegypti IDs with real duration >=1 s; scan <=30; reject known byte/aligned PCM collisions before predictions; no replacement after collision exclusion.')
    atomic_json(out/'external_plan.json',result)
    return result


def train(out=OUT,workers=4):
    if (out/'protocol.json').exists():raise FileExistsError('Training protocol exists; never overwrite an experiment.')
    before=verify_history(out);atomic_json(out/'historical_hashes_before.json',before)
    frame,parts=load_inputs(out);plan=external_plan(out)
    classes=np.sort(frame[frame.label.ne('noise')].label.unique())
    if len(classes)!=20:raise ValueError('Expected twenty species.')
    protocol=dict(created_utc=now(),round='v4_2026_10_14',classes=classes.tolist(),members=MEMBERS,
        primary_model='calibrated',pcm_blocks=560,original_recording_groups=613,window_seconds=1,
        partition_rule='Exact v2 fit/calibration/outer test rows and folds; no new split selection.',
        refit_on_calibration=False,predictions_on_outer_or_external_before_freeze=False,
        cepstral_rule='Mel of short power / local floor; dB >=1e-6 Mel floor, temporal-frame median centring within window, c0=0; c1/c2 retained.',
        psd_rule='10log10(power/local_median_power), clip [-6,45] dB, interpolate then median-centre; never log a zero-truncated linear subtraction.',
        stft=dict(short_samples=1600,short_hop=160,long_samples=3200,long_hop=320,
                  short_bin_spacing_hz=10,long_bin_spacing_hz=5,no_zero_padding=True),
        feature_names=FEATURE_NAMES,feature_workers=workers,prediction_threads=1,
        feature_inductive=True,arbitrary_channel_invariance_proven=False,exact_additive_noise_invariance_proven=False,frontend_stress_sha256=file_hash(out/'frontend_stress_test.json'),
        weights='v2 fit-only class/device/PCM-block weights times v4 per-window salience [.1,3]',
        pooling='v4 selective MIL .20/.35/.50/.65; >4-window bags cannot select all; salience*sharpness*(.15+.85*presence) for species, calibrated presence pooled separately.',
        member_temperature_log_bounds=[-1.6,1.6],blend_prior_regularisation=0.,maximum_species_log_prior=.8,species_prior_rule='Calibration-only mixture weights, temperature [.75,1,1.25,1.5,2], inverse PCM-label log prevalence bias capped +/-.8 with strength [0,.25,.5,.75,1]; maximise recording BA + soft macro F1 at provisional top .35, tie balanced LL then lower bias.',
        genus_rule='Tempered cardinality-corrected power aggregation; gamma grid [0,.25,.5,.75,1], tau [1,1.5,2,3], calibration BA+exp(-balancedLL), uniform-null penalty .05*KL, null max < .30; separate rejection.',
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
            print('V4 FIT',fold,name,len(fitsp),'fit',len(calsp),'cal; NO TEST PREDICTIONS',flush=True)
            start=time.monotonic();model=fit_classical(model_for(METHODS[k]),x[fitsp],labels[fitsp],w)
            raw=v2.inductive_probabilities(model,x[calsp]);temp=v2._temperature(raw,labels[calsp],cw)
            pc.append(v2.calibrated_member(raw,temp));temperatures.append(temp);members.append(model)
            timings.append(dict(member=name,seconds=time.monotonic()-start))
            joblib.dump(dict(model=model,classes=classes.tolist(),fit_rows=fitsp,calibration_rows=calsp,
                             test_rows=test[positive[test]],temperature=temp),out/'models'/f'{fold}_{name}.joblib',compress=3)
        print('V4 FIT',fold,'presence',len(fit),flush=True)
        binary=frame.copy();binary['label']=np.where(positive,'mosquito','noise')
        pres=fit_classical(model_for(METHODS[2]),x[fit],positive[fit].astype(int),
                           v2.training_weights(fit,binary,np.where(positive,s,1.)))
        rawpres=v2.inductive_probabilities(pres,x[cal])
        pw=v2.training_weights(cal,binary,np.ones(len(frame)),device_balance=False)
        calibrator=LogisticRegression(C=100,random_state=20261013).fit(
            logit(np.clip(rawpres[:,1],1e-6,1-1e-6))[:,None],positive[cal].astype(int),sample_weight=pw)
        pp=presence_probabilities(rawpres,calibrator)
        prespool=choose_pooling(binary.iloc[cal],pp,s[cal],np.array(['noise','mosquito']),'calibration')
        pr,pq=recording_probabilities(frame.iloc[cal],pp,s[cal],prespool['top_fraction'])
        species_presence=pd.Series(pp[:,1],index=cal).loc[calsp].to_numpy()
        parameters=fit_blend_v4(np.stack(pc),labels[calsp],cw,frame.iloc[calsp],s[calsp],species_presence,classes,partition='calibration')
        cp=mix_probabilities_v4(pc,parameters)
        pooling=choose_pooling(frame.iloc[calsp],cp,s[calsp],classes,'calibration',presence_prob=species_presence)
        cr,cq=recording_probabilities(frame.iloc[calsp],cp,s[calsp],pooling['top_fraction'],presence_prob=species_presence)
        genus_parameters=select_genus_parameters(cr.label,cq,classes,'calibration',cr.collision_group)
        genera,cg=marginalize_genus_v4(cq,classes,genus_parameters['gamma'],genus_parameters['tau'])
        thresholds=dict(threshold_species=v2.select_threshold(cr.label,cq,classes,'calibration',cr.collision_group),
                        threshold_genus=select_genus_threshold(cr.label.str.split().str[0],cg,genera,'calibration',cr.collision_group))
        settings=dict(pooling=pooling,genus_parameters=genus_parameters,presence_pooling=prespool,threshold_presence=v2._presence_rule(pr,pq),
                      temperatures=temperatures,parameters=parameters,timings=timings,**thresholds)
        atomic_json(out/f'calibration_{fold}.json',settings)
        np.savez_compressed(out/f'calibration_probabilities_{fold}.npz',row_indices=calsp,members=np.stack(pc),
                            blended=cp,presence_rows=cal,presence=pp)
        bundle=dict(models=members,classes=classes.tolist(),settings=settings,presence_model=pres,
                    presence_calibrator=calibrator,fit_rows=fit,calibration_rows=cal,test_rows=test,
                    feature_names=FEATURE_NAMES,created_utc=now(),fold=fold)
        joblib.dump(bundle,out/'models'/f'{fold}_bundle.joblib',compress=3)
        print('V4 CALIBRATED',fold,'GENUS THRESHOLD',settings['threshold_genus']['threshold'],flush=True)
    frozen=list((out/'models').glob('*.joblib'))+list(out.glob('calibration_*.json'))+list(out.glob('calibration_probabilities_*.npz'))
    frozen+=[out/p for p in ['protocol.json','features_v4.npz','partitions.csv','partition_audit.json','segments_v4.csv','external_plan.json','frontend_stress_test.json']]
    atomic_json(out/'manifest.json',dict(completed_training_utc=now(),models_thresholds_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in frozen},
                code_sha256=protocol['code_sha256'],input_sha256=protocol['input_sha256'],no_test_predictions_yet=True,release_allowed=False))
    frozen.append(out/'manifest.json')
    if any(file_hash(ROOT/p)!=h for p,h in protocol['code_sha256'].items()):raise ValueError('Code changed during training.')
    atomic_json(out/'freeze.json',dict(created_utc=now(),artifacts_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in frozen},
                code_sha256=protocol['code_sha256'],all_outer_folds_and_final_frozen=True,no_test_predictions_yet=True,release_allowed=False))
    verify_freeze(out);atomic_json(out/'historical_hashes_after_training.json',verify_history(out))
    print('V4 ALL MODELS FROZEN',now(),flush=True)


def predict_features(x,bundle):
    members=[v2.calibrated_member(v2.inductive_probabilities(m,x),t)
             for m,t in zip(bundle['models'],bundle['settings']['temperatures'])]
    p=mix_probabilities_v4(members,bundle['settings']['parameters'])
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
    classes=np.array(bundle['classes']);g=bundle['settings']['genus_parameters'];genera,gp=marginalize_genus_v4(p,classes,g['gamma'],g['tau'])
    r['scope']=scope;r['quality_ok']=quality;r['salience']=s
    r['start_s']=frame.start_s.to_numpy() if 'start_s' in frame else np.arange(len(p))
    r['predicted_species']=classes[p.argmax(1)];r['species_score']=p.max(1)
    r['genus_marginalized']=genera[gp.argmax(1)];r['genus_posthoc']=r.predicted_species.str.split().str[0]
    r['presence_score']=pp[:,1];r['detected']=quality&(pp[:,1]>=bundle['settings']['threshold_presence']['threshold'])
    return r


def evaluate(out=OUT):
    freeze=verify_freeze(out)
    if (out/'evaluation_audit.json').exists():raise FileExistsError('Completed evaluations cannot be overwritten.')
    novel=json.loads((out/'fresh_selection_v4.json').read_text())
    if novel['selected_utc']<=freeze['created_utc']:raise ValueError('New acquisition must follow freeze.')
    atomic_json(out/'evaluation_started.json',dict(started_utc=now(),frozen_utc=freeze['created_utc'],freeze_sha256=file_hash(out/'freeze.json')))
    frame=pd.read_csv(out/'segments_v4.csv');parts=pd.read_csv(out/'partitions.csv',dtype={'fold':str})
    a=np.load(out/'features_v4.npz');x,s=a['features'],a['salience']
    oof=np.zeros((len(frame),20));poof=np.zeros((len(frame),2));base=np.zeros((3,len(frame),20));assigned=np.zeros(len(frame),int)
    recs=[];windows=[]
    for fold in ['0','1','2']:
        test=parts[parts.fold.eq(fold)&parts.partition.eq('test')].row_index.to_numpy(int)
        bundle=joblib.load(out/'models'/f'{fold}_bundle.joblib')
        if not np.array_equal(bundle['test_rows'],test):raise ValueError('Frozen test row identity changed.')
        p,pp,member=predict_features(x[test],bundle);oof[test]=p;poof[test]=pp;base[:,test]=member;assigned[test]+=1
        quality=np.ones(len(test),bool)
        r,rp,rg=prediction_records(frame.iloc[test],p,s[test],np.array(bundle['classes']),bundle['settings'],pp,quality,'dryad')
        r['fold']=fold;recs.append(r)
        w=window_predictions(frame.iloc[test],p,pp,s[test],quality,bundle,'dryad');w['row_index']=test;windows.append(w)
        np.savez_compressed(out/f'oof_recording_probabilities_{fold}.npz',groups=r.group.to_numpy(str),species=rp,genus=rg)
        print('V4 FROZEN OOF',fold,len(test),'windows',flush=True)
    if not np.all(assigned==1):raise ValueError('OOF identity missing or duplicated.')
    np.savez_compressed(out/'oof_probabilities.npz',row_indices=np.arange(len(frame)),groups=frame.group.to_numpy(str),
                        folds=frame.fold_species.to_numpy(),calibrated=oof,presence=poof,members=base)
    pd.concat(recs,ignore_index=True).to_csv(out/'recording_predictions.csv',index=False)
    pd.concat(windows,ignore_index=True).to_csv(out/'window_predictions.csv.gz',index=False)
    plan=json.loads((out/'external_plan.json').read_text());bundle=joblib.load(out/'models/final_bundle.joblib')
    recs=[];windows=[];probabilities={}
    scopes=[('external',plan['paired_files']),('fresh',plan['fresh_previous']),('drive',[plan['private_file']]),
            ('postfreeze_new',plan['postfreeze_previous']),('postfreeze_new_v4',novel['accepted_files'])]
    for scope,items in scopes:
        for item in items:
            path=ROOT/item['path']
            if file_hash(path)!=item['sha256']:raise ValueError('Evaluation audio changed.')
            p,pp,s,q,info=audio_probabilities(path,bundle)
            f=pd.DataFrame(dict(group=[item['group']]*len(p),label=[item['reference_species']]*len(p)))
            r,rp,rg=prediction_records(f,p,s,np.array(bundle['classes']),bundle['settings'],pp,q,scope)
            r['path']=item['path'];r['invalid_windows']=int((~q).sum())
            r['forced_correct_windows']=int(((np.array(bundle['classes'])[p.argmax(1)]==item['reference_species'])&q).sum())
            for k,value in info.items():r[k]=value
            recs.append(r);windows.append(window_predictions(f,p,pp,s,q,bundle,scope))
            probabilities[item['group']]=dict(species=p,presence=pp,salience=s,quality=q,recording_species=rp,recording_genus=rg)
            print('V4 FROZEN EXTERNAL',scope,path.name,flush=True)
    pd.concat(recs,ignore_index=True).to_csv(out/'external_recording_predictions.csv',index=False)
    pd.concat(windows,ignore_index=True).to_csv(out/'external_window_predictions.csv.gz',index=False)
    joblib.dump(probabilities,out/'external_probabilities.joblib',compress=3)
    verify_freeze(out);history=verify_history(out)
    atomic_json(out/'historical_hashes_after_evaluation.json',history)
    atomic_json(out/'evaluation_audit.json',dict(completed_utc=now(),frozen_utc=freeze['created_utc'],
        freeze_sha256=file_hash(out/'freeze.json'),models_thresholds_unchanged=True,outer_predictions_after_freeze=True,
        paired_external=17,previous_fresh=2,private_recordings=1,postfreeze_new=2,postfreeze_new_v4=len(novel['accepted_files']),release_allowed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','acquire','evaluate','compare','verify'],required=True)
    parser.add_argument('--feature-workers',type=int,default=4);args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='train':train(workers=args.feature_workers)
        elif args.phase=='acquire':
            from fetch_species_v4_fresh import acquire
            acquire()
        elif args.phase=='evaluate':evaluate()
        elif args.phase=='compare':
            from compare_species_v4 import main
            main()
        else:print(verify_freeze());print(verify_history())
