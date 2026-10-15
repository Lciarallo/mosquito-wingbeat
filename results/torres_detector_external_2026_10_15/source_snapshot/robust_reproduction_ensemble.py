"""Round two: choose a conservative ensemble from validation/calibration only.

Motivated by round-one failure on domain shift. Previously inspected evaluations
are now development diagnostics, not blind evidence for this new strategy.
"""
import itertools,json
import numpy as np,pandas as pd
from aedes_binary import ROOT,targets,weights
from train_reproduction import OUT,source_data,wave8_cache,now,sha
from evaluate_reproduction import FrozenPredictor,verify_freeze,summarize


def source_class_recalls(frame,p,cut):
    y=targets(frame);pred=p>=cut;values={}
    for domain in frame.domain.unique():
        for label in [0,1]:
            mask=frame.domain.eq(domain).to_numpy()&(y==label)
            if mask.any():
                w=weights(frame.loc[mask])
                values[f'{domain}|{label}']=float(np.average(pred[mask]==label,weights=w))
    return values


def robust_cut(frame,p,*,partition):
    if partition!='calibration':raise ValueError('No threshold selection on test labels.')
    # Fixed quantile grid, avoiding a threshold search with evaluation data.
    cuts=np.unique(np.r_[0,np.quantile(p,np.linspace(0,1,101)),1.000001])
    y=targets(frame);w=weights(frame)
    correct=(p[None,:]>=cuts[:,None])==y[None,:]
    recalls=[]
    for domain in frame.domain.unique():
        for label in [0,1]:
            mask=frame.domain.eq(domain).to_numpy()&(y==label)
            if mask.any():recalls.append(np.average(correct[:,mask],axis=1,weights=weights(frame.loc[mask])))
    score=np.minimum.reduce(recalls)+.01*np.average(correct,axis=1,weights=w)
    return float(cuts[np.argmax(score)])


def fit():
    if (OUT/'robust_ensemble.json').exists():raise FileExistsError('Round-two parameters already frozen.')
    pred=FrozenPredictor();f,w=source_data();w8=wave8_cache(w)
    x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],axis=1)
    v=f[f.split.eq('validation')&f.quality_ok&(targets(f)>=0)]
    c=f[f.split.eq('calibration')&f.quality_ok&(targets(f)>=0)]
    names=sorted(pred.models);vp=pred.frame(v,w8,x[v.index]);cp=pred.frame(c,w8,x[c.index])
    vv=np.stack([vp[n] for n in names],1);cc=np.stack([cp[n] for n in names],1)
    grid=[np.eye(len(names))[i] for i in range(len(names))]
    for a,b in itertools.combinations(range(len(names)),2):
        for k in [.25,.5,.75]:
            w=np.zeros(len(names));w[a]=k;w[b]=1-k;grid.append(w)
    for ix in itertools.combinations(range(len(names)),3):
        w=np.zeros(len(names));w[list(ix)]=1/3;grid.append(w)
    protocol=dict(created_utc=now(),base_freeze_sha256=sha(OUT/'freeze.json'),
        strategy='Soft voting: 7 singles, 63 weighted pairs, 35 equal triplets.',
        criterion='Maximize minimum validation source/class recording-balanced recall + 0.01 globally balanced validation accuracy.',
        cut='Choose calibration quantile threshold maximizing minimum source/class recall + 0.01 globally balanced calibration accuracy.',
        model_selection_partition='validation',threshold_partition='calibration',
        test_and_private_used_for_fitting=False,prior_evaluations_now_development_diagnostics=True,
        negative_only_fresh_test_cannot_estimate_positive_sensitivity=True,arduino_compiled=False)
    (OUT/'robust_protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for i,w in enumerate(grid):
        cut=robust_cut(c,cc@w,partition='calibration');p=vv@w
        r=source_class_recalls(v,p,cut);ba=float(np.average((p>=cut)==targets(v),weights=weights(v)))
        rows.append(dict(candidate=i,weights=w.tolist(),cut=cut,minimum_validation_source_class_recall=min(r.values()),
                         validation_weighted_ba=ba,validation_source_class_recalls=r,score=min(r.values())+.01*ba))
    chosen=max(rows,key=lambda r:r['score'])
    result=dict(**chosen,model_names=names,frozen_utc=now(),base_model_hashes=pred.freeze['models_sha256'],
                protocol_sha256=sha(OUT/'robust_protocol.json'),inference_source_sha256=sha(ROOT/'robust_reproduction_ensemble.py'))
    (OUT/'robust_validation_search.json').write_text(json.dumps(rows,indent=2))
    (OUT/'robust_ensemble.json').write_text(json.dumps(result,indent=2))
    verify_freeze();print('ROBUST FROZEN',json.dumps(result),flush=True)


def load():
    r=json.loads((OUT/'robust_ensemble.json').read_text());verify_freeze()
    assert sha(OUT/'robust_protocol.json')==r['protocol_sha256']
    assert sha(ROOT/'robust_reproduction_ensemble.py')==r['inference_source_sha256']
    return r


def blend(r,p):return np.stack([p[n] for n in r['model_names']],1)@np.asarray(r['weights'])


def evaluate_diagnostics():
    r=load();cut=r['cut'];df=pd.read_csv(OUT/'historical_predictions.csv.gz');rows=[]
    p=blend(r,{n:df[n].to_numpy() for n in r['model_names']})
    for domain,sub in df.groupby('domain'):
        ix=sub.index;rows.append(dict(domain=domain,**summarize(targets(sub),p[ix],sub.quality_ok,cut)))
    pd.DataFrame(rows).to_csv(OUT/'robust_historical_metrics_exploratory.csv',index=False)
    priv=pd.read_csv(ROOT/'tmp/reproductions_2026_10_01/private_predictions.csv')
    pp=blend(r,{n:priv[n].to_numpy() for n in r['model_names']})
    (OUT/'robust_private_audio_aggregate.json').write_text(json.dumps(summarize(np.ones(len(priv),int),pp,priv.quality_ok,cut),indent=2))
    # Round-one external samples have now informed the strategy, so mark reuse.
    d=pd.read_csv(OUT/'fresh_predictions.csv.gz')
    pivot=d.pivot(index=['path','start_s'],columns='model',values='score')
    meta=d.drop_duplicates(['path','start_s']).set_index(['path','start_s']).loc[pivot.index]
    p=blend(r,{n:pivot[n].to_numpy() for n in r['model_names']});rows=[]
    for source,ix in meta.reset_index(drop=True).groupby('source').groups.items():
        sub=meta.iloc[list(ix)];rows.append(dict(source=source,**summarize(sub.label,p[list(ix)],sub.quality_ok,cut)))
    pd.DataFrame(rows).to_csv(OUT/'robust_prior_fresh_metrics_exploratory.csv',index=False)
    print('ROBUST DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fit','diagnostics'],required=True);args=ap.parse_args()
    fit() if args.phase=='fit' else evaluate_diagnostics()
