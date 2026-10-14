"""Recording-disjoint validation, frozen calibration, and batch-independent inference.

Run --phase fit once, then --phase evaluate. Previously inspected corpora are
historical diagnostics, not new blind tests. Models and reports are preserved.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, io, json, re, zipfile
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.special import expit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from aedes_binary import ROOT, targets, weights, decisions
from aedes_classical import features, raw_scores
from bioacoustic_invariance import extract_batch_bioacoustic_features, BIOACOUSTIC_FEATURE_NAMES
from models_deep import DeepBioacousticHybridNet
from swarm_adapted_ensemble import SwarmAdaptedEnsemble

OUT = ROOT / 'results/corrected_2026_10_01'
PRIVATE = ROOT / 'tmp/corrected_2026_10_01'
SEED = 20261005


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def assert_disjoint(frame):
    if frame.groupby('group').split.nunique().max() != 1:
        raise ValueError('Recording occurs in multiple partitions.')
    if frame.dropna(subset=['content_sha256']).groupby('content_sha256').split.nunique().max() != 1:
        raise ValueError('Byte-identical recordings occur in multiple partitions.')


def select_thresholds(y, p, sample_weight, *, partition):
    if partition != 'calibration':
        raise ValueError('Evaluation labels cannot be used to choose thresholds.')
    y = np.asarray(y); p = np.asarray(p); w = np.asarray(sample_weight)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Calibration needs both labelled classes.')
    high = next(float(t) for t in np.r_[np.linspace(.5, .999, 500), 1.0001]
                if np.average(p[y == 0] >= t, weights=w[y == 0]) <= .05)
    low = 0.
    for t in np.linspace(.001, min(.499, high - 1e-6), 499):
        if np.average(p[y == 1] <= t, weights=w[y == 1]) <= .05:
            low = float(t)
    return low, high


def metrics(y, p, quality, low, high, forced_threshold=.5):
    y = np.asarray(y); p = np.asarray(p); quality = np.asarray(quality, bool)
    known = y >= 0; pos = y == 1; neg = y == 0
    states = decisions(p, quality, low, high)
    emit = states == 'AEDES_AEGYPTI'; reject = states == 'NAO_AEDES_AEGYPTI'
    accepted = emit | reject
    tp = int((emit & pos).sum()); fp = int((emit & neg).sum())
    yp = p >= forced_threshold
    two = len(np.unique(y[known])) == 2
    return dict(windows=len(y), positive_windows=int(pos.sum()), negative_windows=int(neg.sum()),
                unknown_windows=int((~known).sum()), invalid=int((~quality).sum()),
                auc=float(roc_auc_score(y[known], p[known])) if two else None,
                forced_threshold=float(forced_threshold),
                balanced_accuracy=float(balanced_accuracy_score(y[known], yp[known])) if two else None,
                forced_accuracy=float((yp[known] == y[known]).mean()) if known.any() else None,
                aedes_emissions=int(emit.sum()), true_positive_emissions=tp, false_positive_emissions=fp,
                emitted_aedes_recall=tp/int(pos.sum()) if pos.any() else None,
                emitted_false_positive_rate=fp/int(neg.sum()) if neg.any() else None,
                emitted_precision=tp/(tp+fp) if tp+fp else None,
                false_negative_emissions=int((reject & pos).sum()),
                uncertain=int((states == 'INCERTO').sum()),
                known_coverage=float(accepted[known].mean()) if known.any() else None,
                unresolved_aedes_emissions=int((emit & ~known).sum()))


class Predictor:
    def __init__(self):
        torch.set_num_threads(4)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ck = torch.load(ROOT/'results/models_deep/deep_hybrid_net.pt', map_location=self.device, weights_only=False)
        self.net = DeepBioacousticHybridNet(138).to(self.device)
        self.net.load_state_dict(self.ck['state_dict']); self.net.eval()
        self.trees = {n: joblib.load(OUT/f'models_trees/{n}.joblib') for n in
                      ['lightgbm','hist_gradient_boosting','extra_trees','random_forest']}
        self.comm = joblib.load(OUT/'models_committee/committees_bundle.joblib')
        self.baselines = {n: joblib.load(ROOT/f'results/aedes_classical/models/{n}.joblib') for n in
                          ['svm_rbf','extra_trees','gradient_boosting']}

    def tabular(self, x):
        p = {'tree_'+n: expit(m['slope']*raw_scores(m['model'], x)+m['bias']) for n,m in self.trees.items()}
        p.update({'baseline_'+n: expit(m['slope']*raw_scores(m['model'], x[:,:122])+m['bias'])
                  for n,m in self.baselines.items()})
        p['committee_soft_voting'] = self.comm['soft_voting_committee'].predict_proba(x)
        p['committee_stacking'] = self.comm['stacking_committee'].predict_proba(x)
        return p

    def deep(self, wave, x):
        output = []
        with torch.no_grad():
            for i in range(0, len(wave), 96):
                w = np.array(wave[i:i+96], dtype='float32', copy=True)
                if w.shape[1] < 15872:
                    w = np.tile(w, (1,int(np.ceil(15872/w.shape[1]))))[:,:15872]
                b = ((x[i:i+96]-self.ck['scaler_mean'])/self.ck['scaler_scale']).astype('float32')
                z = self.net(torch.from_numpy(w).to(self.device), torch.from_numpy(b).to(self.device)).cpu().numpy()
                output.extend(expit(self.ck['slope']*z+self.ck['bias']))
        return np.asarray(output)

    def frame(self, frame, x, waves):
        p = self.tabular(x); p['deep_hybrid_net'] = np.empty(len(frame))
        for bucket in ['long','short']:
            idx = np.flatnonzero(frame.bucket.eq(bucket))
            if len(idx):
                p['deep_hybrid_net'][idx] = self.deep(waves[bucket][frame.iloc[idx].wave_index.astype(int)],x[idx])
        p['super_committee'] = .5*p['deep_hybrid_net']+.5*p['committee_soft_voting']
        return p

    def audio(self, wave):
        x = np.concatenate([features(wave),extract_batch_bioacoustic_features(wave)],axis=1).astype('float32')
        p = self.tabular(x); p['deep_hybrid_net'] = self.deep(wave,x)
        p['super_committee'] = .5*p['deep_hybrid_net']+.5*p['committee_soft_voting']
        return p, x


def components(p, x):
    return np.column_stack([p['deep_hybrid_net'],p['baseline_svm_rbf'],
                            x[:,122+BIOACOUSTIC_FEATURE_NAMES.index('carrier_energy_ratio')]])


def audio_windows(path):
    w,sr = sf.read(path,dtype='float32',always_2d=True); w=w.mean(1)
    if sr != 16000:
        g=np.gcd(sr,16000); w=resample_poly(w,16000//g,sr//g)
    size=15872 if len(w)>=15872 else 4800
    if len(w)<size: raise ValueError('Less than 0.3s of real audio.')
    w=w[:len(w)//size*size].reshape(-1,size)
    return w,quality(w),size


def quality(w):
    pcm=np.rint(np.clip(w,-1,32767/32768)*32768).astype('int16').astype('int32')
    return (np.abs(pcm).max(1)>=32)&((np.abs(pcm)>=32700).sum(1)*100<=w.shape[1])


def plan_d5():
    """Separate official validation/test lists; exclude all previously evaluated clips."""
    previous=pd.read_csv(ROOT/'tmp/independent_retest_2026_10_01/d5_predictions.csv')
    excluded=set(previous.file.str.removesuffix('.wav'))
    result={'selection_utc':now(),'seed':SEED,'prior_400_ids_excluded':True,
            'method':'Hash-sorted selection before inference; official validation for calibration, official test for evaluation.',
            'independent_individual_identity_not_available':True}
    zpath=ROOT/'data/external/additional/Development_data.zip'
    with zipfile.ZipFile(zpath) as z:
        for split,txt in [('calibration','Validation_ids.txt'),('test','Test_ids.txt')]:
            ids=z.read('Development_data/metadata/'+txt).decode().splitlines();selected=[]
            for sid,count in [(1,200),(3,70),(4,70),(7,60)]:
                pool=[s for s in ids if s.startswith(f'S_{sid}_D_5_') and s not in excluded]
                pool=sorted(pool,key=lambda s:hashlib.sha256(f'{SEED}:{s}'.encode()).hexdigest())
                if len(pool)<count:raise ValueError('Insufficient independent clip IDs.')
                selected.extend(dict(id=s,label=int(sid==1),species_id=sid) for s in pool[:count])
            result[split]=selected
    assert not ({x['id'] for x in result['calibration']}&{x['id'] for x in result['test']})
    write('d5_selection.json',result)
    return result


def read_d5(selection):
    waves=[];meta=[]
    with zipfile.ZipFile(ROOT/'data/external/additional/Development_data.zip') as z:
        for r in selection:
            raw=z.read('Development_data/raw_audio/'+r['id']+'.wav')
            w,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True);w=w.mean(1)
            if sr!=16000:
                g=np.gcd(sr,16000);w=resample_poly(w,16000//g,sr//g)
            if len(w)<4800:raise ValueError('Selected D5 recording is too short; no substitution by result.')
            waves.append(w[:4800]);meta.append(dict(**r,sha256=hashlib.sha256(raw).hexdigest(),
                 pcm_sha256=hashlib.sha256(np.rint(w*32768).astype('int32').tobytes()).hexdigest()))
    return np.array(waves,dtype='float32'),pd.DataFrame(meta)


def fit():
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    if (OUT/'freeze.json').exists():raise FileExistsError('Frozen experiment already exists; do not silently refit after test.')
    f=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);assert_disjoint(f);y=targets(f)
    x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),
                      np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],axis=1).astype('float32')
    waves={b:np.load(ROOT/f'data/cache/multisource/{b}.npy',mmap_mode='r') for b in ['long','short']}
    ix={s:np.flatnonzero(f.split.eq(s)&f.quality_ok&(y>=0)) for s in ['train','calibration']}
    write('protocol.json',dict(seed=SEED,created_utc=now(),source_splits_sha256=sha(ROOT/'results/multisource/segments.csv'),
        models='Four unchanged tree hyperparameters; recording-group OOF; raw OOF inputs at meta inference; existing deep CNN and SVM frozen.',
        selection='No selection by historical/external/private test; all candidates reported.',
        normalization='Weighted empirical CDFs fitted on known training recordings only; no batch ranks.',
        threshold='Trees/committees target 5% weighted calibration FPR; swarm Youden threshold on main calibration only.',
        private_audio_training=False,unknown_species_negative=False,arduino_compiled=False,
        synthetic_original='Abuzz-derived mixtures with clip-level presence labels; exploratory file-level only. Not independent external evidence.',
        old_noise_controls_excluded=True,deep_retrained=False))
    d5plan=plan_d5();pred=Predictor();train=f.iloc[ix['train']];cal=f.iloc[ix['calibration']]
    print('Computing frozen training references',len(train),flush=True)
    ptr=pred.frame(train,x[ix['train']],waves)
    ensemble=SwarmAdaptedEnsemble();ensemble.fit_reference(components(ptr,x[ix['train']]),weights(train),partition='train')
    pca=pred.frame(cal,x[ix['calibration']],waves)
    pca['swarm_corrected']=ensemble.blend_components(components(pca,x[ix['calibration']]))
    t=ensemble.calibrate_threshold(targets(cal),pca['swarm_corrected'],sample_weight=weights(cal),partition='calibration')
    thresholds={'tree_'+n:(m['low'],m['high']) for n,m in pred.trees.items()}
    thresholds.update({'baseline_'+n:(m['low'],m['high']) for n,m in pred.baselines.items()})
    for n,k in [('committee_soft_voting','soft_voting_thresholds'),('committee_stacking','stacking_thresholds')]:
        thresholds[n]=(pred.comm[k]['low'],pred.comm[k]['high'])
    thresholds['deep_hybrid_net']=(pred.ck['low'],pred.ck['high'])
    thresholds['super_committee']=select_thresholds(targets(cal),pca['super_committee'],weights(cal),partition='calibration')
    thresholds['swarm_corrected']=(t,t)
    # Domain-specific calibration is fitted exclusively on the official optical validation clips.
    dw,dm=read_d5(d5plan['calibration']);dp,dx=pred.audio(dw);dp['swarm_corrected']=ensemble.blend_components(components(dp,dx))
    d5thresholds={};q=quality(dw);dy=dm.label.to_numpy()
    for name,p in dp.items():
        ff,tt,th=roc_curve(dy[q],p[q]);d5thresholds[name]=float(th[np.argmax(tt-ff)])
    dm.to_csv(OUT/'d5_calibration_manifest.csv',index=False)
    joblib.dump(dict(ensemble=ensemble,thresholds=thresholds,d5_thresholds=d5thresholds),OUT/'models_committee/corrected_bundle.joblib')
    model_paths=list((OUT/'models_trees').glob('*.joblib'))+list((OUT/'models_committee').glob('*.joblib'))
    model_paths += [ROOT/'results/models_deep/deep_hybrid_net.pt']+list((ROOT/'results/aedes_classical/models').glob('*.joblib'))
    freeze=dict(frozen_utc=now(),models_sha256={str(p.relative_to(ROOT)):sha(p) for p in model_paths},
                protocol_sha256=sha(OUT/'protocol.json'),d5_selection_sha256=sha(OUT/'d5_selection.json'),
                original_weights_unchanged=True,external_test_labels_used_to_fit=False)
    old=json.loads((OUT/'original_weights_sha256.json').read_text());assert all(sha(ROOT/p)==h for p,h in old.items())
    write('freeze.json',freeze)
    print('FROZEN',freeze['frozen_utc'],'swarm threshold',t,flush=True)


def load_frozen():
    audit=json.loads((OUT/'freeze.json').read_text())
    for path,h in audit['models_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Frozen model changed: '+path)
    if sha(OUT/'protocol.json')!=audit['protocol_sha256']:raise ValueError('Protocol changed after freeze.')
    protocol=json.loads((OUT/'protocol.json').read_text())
    if sha(ROOT/'results/multisource/segments.csv')!=protocol['source_splits_sha256']:
        raise ValueError('Source partitions changed after freeze.')
    if sha(OUT/'d5_selection.json')!=audit['d5_selection_sha256']:raise ValueError('D5 selection changed after freeze.')
    return Predictor(),joblib.load(OUT/'models_committee/corrected_bundle.joblib'),audit


def complete_predictions(p,x,bundle):
    p['swarm_corrected']=bundle['ensemble'].blend_components(components(p,x))
    return p


def evaluate():
    pred,bundle,freeze=load_frozen();rows=[]
    f=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);assert_disjoint(f)
    x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),
                      np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],axis=1).astype('float32')
    waves={b:np.load(ROOT/f'data/cache/multisource/{b}.npy',mmap_mode='r') for b in ['long','short']}
    idx=np.flatnonzero(f.split.isin(['test','challenge']));tf=f.iloc[idx].copy()
    p=complete_predictions(pred.frame(tf,x[idx],waves),x[idx],bundle)
    outf=tf[['path','label','group','domain','start_s','quality_ok']].copy()
    for name,scores in p.items():
        outf[name]=scores;l,h=bundle['thresholds'][name]
        threshold=h if name=='swarm_corrected' else .5
        for domain,inds in tf.reset_index(drop=True).groupby('domain').groups.items():
            inds=np.asarray(list(inds));row=metrics(targets(tf.iloc[inds]),scores[inds],tf.iloc[inds].quality_ok,l,h,threshold)
            rows.append(dict(model=name,domain=domain,**row))
    outf.to_csv(OUT/'historical_predictions.csv.gz',index=False);pd.DataFrame(rows).to_csv(OUT/'historical_metrics.csv',index=False)
    # Evaluate only noise from test partitions; never reuse adaptation/training noise.
    noiserows=[]
    for name,scores in p.items():
        l,h=bundle['thresholds'][name]
        for domain in ['dryad','humbug']:
            mask=tf.domain.eq(domain)&tf.label.eq('noise')
            noiserows.append(dict(model=name,domain=domain,**metrics(targets(tf[mask]),scores[mask],tf.loc[mask,'quality_ok'],l,h)))
    pd.DataFrame(noiserows).to_csv(OUT/'heldout_noise_metrics.csv',index=False)
    # Private audio: reference is user-provided Aedes throughout; per-window data stays local.
    w,q,size=audio_windows(ROOT/'tmp/private-audio-test/download.audio');pp,xx=pred.audio(w);pp=complete_predictions(pp,xx,bundle);priv=[]
    pf=pd.DataFrame({'start_s':np.arange(len(w))*size/16000,'quality_ok':q})
    for name,scores in pp.items():
        l,h=bundle['thresholds'][name];priv.append(dict(model=name,**metrics(np.ones(len(w),int),scores,q,l,h,h if name=='swarm_corrected' else .5)));pf[name]=scores
    pd.DataFrame(priv).to_csv(OUT/'private_audio_aggregate.csv',index=False);pf.to_csv(PRIVATE/'private_predictions.csv',index=False)
    # File-level presence only: the original generator inserts separated species chunks.
    from evaluate_new_database import load_and_preprocess_dataset
    sw,files,win=load_and_preprocess_dataset();sp,sx=pred.audio(sw);sp=complete_predictions(sp,sx,bundle)
    synth=[]
    for name,scores in sp.items():
        temp=pd.DataFrame({'filename':win.filename,'score':scores});agg=temp.groupby('filename').score.mean();yy=files.set_index('filename').loc[agg.index,'is_aedes_aegypti'].to_numpy()
        l,h=bundle['thresholds'][name];synth.append(dict(model=name,**metrics(yy,agg.to_numpy(),np.ones(len(yy),bool),l,h,h if name=='swarm_corrected' else .5)))
    pd.DataFrame(synth).to_csv(OUT/'synthetic_file_metrics_exploratory.csv',index=False)
    # The reference transform must give identical answers alone, by file, or in a batch.
    c=components(sp,sx);full=bundle['ensemble'].blend_components(c)
    one=np.concatenate([bundle['ensemble'].blend_components(c[i:i+1]) for i in range(len(c))])
    chunks=np.concatenate([bundle['ensemble'].blend_components(c[i:i+5]) for i in range(0,len(c),5)])
    assert np.allclose(full,one,atol=1e-15) and np.allclose(full,chunks,atol=1e-15)
    write('batch_invariance_audit.json',dict(rows=len(c),max_difference_single=float(abs(full-one).max()),max_difference_by_file=float(abs(full-chunks).max())))
    # New D5 evaluation clips remain outside all model/threshold fitting.
    plan=json.loads((OUT/'d5_selection.json').read_text());dw,dm=read_d5(plan['test']);cal=pd.read_csv(OUT/'d5_calibration_manifest.csv')
    assert not (set(dm.sha256)&set(cal.sha256));assert not (set(dm.pcm_sha256)&set(cal.pcm_sha256))
    dm.to_csv(OUT/'d5_test_manifest.csv',index=False);dp,dx=pred.audio(dw);dp=complete_predictions(dp,dx,bundle);drows=[]
    for name,scores in dp.items():
        l,h=bundle['thresholds'][name];dy=dm.label.to_numpy();q=quality(dw)
        generic=metrics(dy,scores,q,l,h,h if name=='swarm_corrected' else .5)
        dt=bundle['d5_thresholds'][name];adapted=metrics(dy,scores,q,dt,dt,dt)
        drows.extend([dict(model=name,calibration='main_acoustic',**generic),dict(model=name,calibration='separate_D5_validation',**adapted)])
    pd.DataFrame(drows).to_csv(OUT/'d5_test_metrics.csv',index=False)
    write('evaluation_audit.json',dict(completed_utc=now(),frozen_utc=freeze['frozen_utc'],
         main_test_windows=len(tf),heldout_noise_windows=int(tf.label.eq('noise').sum()),
         private_windows=len(w),synthetic_files=len(files),d5_test_files=len(dm),
         d5_calibration_test_exact_byte_and_pcm_overlap=0,refit_after_test=False))
    load_frozen()
    print('CORRECTED EVALUATION COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fit','evaluate'],required=True);args=ap.parse_args()
    if args.phase=='fit':fit()
    else:evaluate()


if __name__=='__main__':main()
