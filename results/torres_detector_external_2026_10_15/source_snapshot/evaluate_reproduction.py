"""Inference only: preserve all seven candidates and the validation-selected model.

Historical tests and private recording are diagnostics. Fresh recordings must
have been downloaded after freeze, with source/label rules fixed before scores.
"""
import argparse,json
import numpy as np,pandas as pd,torch,joblib
from scipy.special import expit
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
from aedes_binary import ROOT,targets,weights
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from corrected_validation import sha,metrics,audio_windows
from train_reproduction import OUT,source_data,wave8_cache,infer,now
from reproduction_models import build_model


def verify_freeze():
    f=json.loads((OUT/'freeze.json').read_text())
    for p,h in {**f['models_sha256'],**f['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen artifact changed: '+p)
    for p in ['protocol','selection']:
        if sha(OUT/(p+'.json'))!=f[p+'_sha256']:raise ValueError('Frozen '+p+' changed.')
    return f


def summarize(y,p,q,cut):
    y=np.asarray(y);p=np.asarray(p);q=np.asarray(q,bool)
    r=metrics(y,p,q,cut,cut,cut)
    pos=y==1;neg=y==0;pred=p>=cut
    r['forced_aedes_recall']=float(pred[pos].mean()) if pos.any() else None
    r['forced_false_positive_rate']=float(pred[neg].mean()) if neg.any() else None
    r['emitted_specificity']=float((q[neg]&~pred[neg]).mean()) if neg.any() else None
    r['operational_balanced_accuracy']=.5*(r['emitted_aedes_recall']+r['emitted_specificity']) if pos.any() and neg.any() else None
    r['negative_emissions']=int((q&~pred).sum())
    return r


class FrozenPredictor:
    def __init__(self):
        self.freeze=verify_freeze();torch.set_num_threads(4)
        self.device='cuda' if torch.cuda.is_available() else 'cpu';self.models={};self.cuts={}
        for p in sorted((OUT/'models').iterdir()):
            if p.suffix=='.pt':
                saved=torch.load(p,map_location='cpu',weights_only=True)
                m=build_model(saved['name']).to(self.device);m.load_state_dict(saved['state_dict']);m.eval()
            elif p.suffix=='.joblib':saved=joblib.load(p);m=saved['model']
            else:continue
            self.models[saved['name']]=(m,saved);self.cuts[saved['name']]=saved['cut']
        self.selected=json.loads((OUT/'selection.json').read_text())['selected_model']

    def frame(self,frame,waves,x):
        result={}
        for name,(m,saved) in self.models.items():
            result[name]=(m.predict_proba(x)[:,1] if name.startswith('extra_trees') else
                          expit(saved['slope']*infer(m,name,frame,waves,self.device)+saved['bias']))
        return result

    def audio(self,path):
        w,q,size=audio_windows(path)
        x=np.concatenate([features(w),extract_batch_bioacoustic_features(w)],axis=1).astype('float32')
        bucket='long' if size==15872 else 'short'
        f=pd.DataFrame(dict(bucket=[bucket]*len(w),wave_index=np.arange(len(w))))
        w8=resample_poly(w,1,2,axis=1).astype('float32')
        waves={bucket:w8,'short' if bucket=='long' else 'long':np.empty((0,2400 if bucket=='long' else 7936),dtype='float32')}
        return self.frame(f,waves,x),q,size


def evaluate_main():
    pred=FrozenPredictor();f,w=source_data();w8=wave8_cache(w)
    tf=f[f.split.isin(['test','challenge'])].copy()
    x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],axis=1)
    scores=pred.frame(tf,w8,x[tf.index]);rows=[]
    out=tf[['path','label','group','domain','start_s','quality_ok']].copy()
    for name,p in scores.items():
        out[name]=p
        for domain in tf.domain.unique():
            mask=tf.domain.eq(domain).to_numpy();sub=tf.loc[mask];y=targets(sub)
            r=summarize(y,p[mask],sub.quality_ok,pred.cuts[name])
            known=y>=0
            r['recording_species_source_balanced_accuracy']=float(np.average((p[mask][known]>=pred.cuts[name])==y[known],weights=weights(sub.loc[known]))) if np.unique(y[known]).size==2 else None
            rows.append(dict(model=name,selected=name==pred.selected,domain=domain,recording_groups=sub.group.nunique(),**r))
    out.to_csv(OUT/'historical_predictions.csv.gz',index=False)
    pd.DataFrame(rows).to_csv(OUT/'historical_metrics.csv',index=False)
    p,q,size=pred.audio(ROOT/'tmp/private-audio-test/download.audio');rows=[]
    private=pd.DataFrame(dict(start_s=np.arange(len(q))*size/16000,quality_ok=q))
    for name,score in p.items():
        private[name]=score
        rows.append(dict(model=name,selected=name==pred.selected,**summarize(np.ones(len(q),int),score,q,pred.cuts[name])))
    pd.DataFrame(rows).to_csv(OUT/'private_audio_aggregate.csv',index=False)
    d=ROOT/'tmp/reproductions_2026_10_01';d.mkdir(exist_ok=True)
    private.to_csv(d/'private_predictions.csv',index=False)
    (OUT/'evaluation_audit.json').write_text(json.dumps(dict(completed_utc=now(),frozen_utc=pred.freeze['frozen_utc'],
        selected_before_test=pred.selected,main_windows=len(tf),private_windows=len(q),private_groups=1,
        no_refit=True,no_threshold_tuning=True,unknowns_excluded_from_accuracy=True,arduino_compiled=False),indent=2))
    verify_freeze();print('MAIN EVALUATION COMPLETE',pred.selected,flush=True)


def group_bootstrap(frame,cut,replicates=2000):
    """Stratify by true class, resample recording groups, never treat windows as iid."""
    means=frame.assign(correct=(frame.score.ge(cut)==frame.label)&frame.quality_ok).groupby(['label','group']).correct.mean()
    arrays=[means.loc[label].to_numpy() for label in [0,1] if label in means.index.get_level_values(0)]
    if len(arrays)!=2:return None
    rng=np.random.default_rng(20261008)
    estimates=.5*sum(rng.choice(x,size=(replicates,len(x)),replace=True).mean(1) for x in arrays)
    return dict(recording_macro_balanced_accuracy=float(.5*sum(x.mean() for x in arrays)),
                recording_macro_ba_ci95=np.quantile(estimates,[.025,.975]).tolist(),
                positive_groups=len(arrays[1]),negative_groups=len(arrays[0]),replicates=replicates,
                scope='Conditional on these sources and weak recording-presence labels; no event annotations.')


def evaluate_fresh():
    pred=FrozenPredictor();plan=json.loads((OUT/'fresh_selection.json').read_text())
    assert plan['selection_utc']>pred.freeze['frozen_utc'] and plan['model_hashes']==pred.freeze['models_sha256']
    perfile=[];windows=[]
    for item in plan['files']:
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        p,q,size=pred.audio(path);y=np.full(len(q),int(item['reference_aedes']))
        for name,score in p.items():
            perfile.append(dict(model=name,selected=name==pred.selected,path=item['path'],source=item['source'],
                group=item['group'],reference_species=item['reference_species'],**summarize(y,score,q,pred.cuts[name])))
            windows.extend(dict(model=name,source=item['source'],path=item['path'],group=item['group'],label=int(yy),
                quality_ok=bool(qq),score=float(pp),start_s=i*size/16000,reference_species=item['reference_species'])
                for i,(yy,qq,pp) in enumerate(zip(y,q,score)))
    df=pd.DataFrame(windows);df.to_csv(OUT/'fresh_predictions.csv.gz',index=False)
    pd.DataFrame(perfile).to_csv(OUT/'fresh_file_metrics.csv',index=False);rows=[];bootstrap={}
    for name,sub in df.groupby('model'):
        cut=pred.cuts[name];bootstrap[name]=group_bootstrap(sub,cut)
        for source in ['all_fresh',*sub.source.unique()]:
            s=sub if source=='all_fresh' else sub[sub.source.eq(source)]
            rows.append(dict(model=name,selected=name==pred.selected,source=source,groups=s.group.nunique(),
                             **summarize(s.label,s.score,s.quality_ok,cut)))
    pd.DataFrame(rows).to_csv(OUT/'fresh_metrics.csv',index=False)
    (OUT/'fresh_group_bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
    (OUT/'fresh_evaluation_audit.json').write_text(json.dumps(dict(completed_utc=now(),selection_sha256=sha(OUT/'fresh_selection.json'),
        source_group_novelty_checked=True,duplicate_sha256_excluded=True,files=len(plan['files']),
        selected_model=pred.selected,no_refit=True,pnas_limit='Positive pair presence; not independently annotated per-window flight.'),indent=2))
    verify_freeze();print('FRESH EVALUATION COMPLETE',pred.selected,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['main','fresh'],required=True);a=ap.parse_args()
    evaluate_main() if a.phase=='main' else evaluate_fresh()
