"""Train four model families; select only by validation, calibrate and freeze all."""
from datetime import datetime,timezone
import json,hashlib,time,os
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from scipy.special import expit
from aedes_binary import targets,weights,decisions,ROOT
from aedes_classical import O,features,FEATURE_NAMES,raw_scores
from train_aedes_binary import summary

def write(name,value):(O/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))
def main():
    O.mkdir(parents=True,exist_ok=True);(O/'models').mkdir(exist_ok=True)
    f=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);assert f.groupby('group').split.nunique().max()==1
    y=targets(f);ix={s:np.flatnonzero(f.split.eq(s)&f.quality_ok&(y>=0)) for s in ['train','validation','calibration']}
    assert not f.iloc[ix['train']].domain.eq('ufrgs').any()
    x=np.empty((len(f),len(FEATURE_NAMES)),dtype='float32')
    cached_features=ROOT/'data/cache/aedes_classical_features.npy'
    cache_audit=ROOT/'data/cache/aedes_classical_features.json'
    provenance=dict(segments_sha256=hashlib.sha256((ROOT/'results/multisource/segments.csv').read_bytes()).hexdigest(),
                    feature_code_sha256=hashlib.sha256((ROOT/'aedes_classical.py').read_bytes()).hexdigest())
    if (O/'training_audit.json').exists():
        assert json.loads((O/'training_audit.json').read_text())['input_provenance']==provenance, 'Inputs changed: use a new experiment directory; do not reuse frozen models.'
    if cached_features.exists() and cache_audit.exists() and json.loads(cache_audit.read_text())==provenance:
        x=np.load(cached_features);assert x.shape==(len(f),len(FEATURE_NAMES))
    else:
        for bucket in ['long','short']:
            mask=f.bucket.eq(bucket).to_numpy();wave=np.load(ROOT/f'data/cache/multisource/{bucket}.npy',mmap_mode='r')
            indices=f.loc[mask,'wave_index'].astype(int).to_numpy();x[mask]=features(wave[indices]);print('Features',bucket,int(mask.sum()),flush=True)
        np.save(cached_features,x)
        cache_audit.write_text(json.dumps(provenance,indent=2))
    tw=weights(f.iloc[ix['train']]);vw=weights(f.iloc[ix['validation']]);cw=weights(f.iloc[ix['calibration']]);tr,va,ca=[ix[s] for s in ['train','validation','calibration']]
    configs={
      'logistic':[(dict(C=c),make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000))) for c in [.1,1]],
      'svm_rbf':[(dict(C=c,gamma='scale'),make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale',cache_size=1000))) for c in [1,10]],
      'extra_trees':[(dict(min_samples_leaf=l),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=l,max_features=.7,n_jobs=4,random_state=20261004)) for l in [2,8]],
      'gradient_boosting':[(dict(max_leaf_nodes=l),HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=l,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=20261004)) for l in [15,31]]}
    write('protocol.json',dict(seed=20261004,features=FEATURE_NAMES,feature_fit='No fitted acoustic transformation; scaler fitted on training only',source_weights='Equal binary classes; equal negative labels, domains and groups',
        selection='weighted validation ROC-AUC; ties favor first candidate',calibration='separate old calibration groups; logistic scores and 5% source-weighted FPR target',unknown_excluded=True,input_provenance=provenance,
        previous_test_sets_historical=True,ufrgs_entirely_heldout=True,no_padding=True,arduino_compiled=False,hyperparameters={k:[q for q,m in v] for k,v in configs.items()}))
    history=json.loads((O/'validation_search.json').read_text()) if (O/'validation_search.json').exists() else []
    cached={r['model']:r for r in json.loads((O/'frozen_models.json').read_text())} if (O/'frozen_models.json').exists() else {}
    frozen=[];metrics=[];all_predictions=[];test=np.flatnonzero(f.split.isin(['test','challenge']))
    for name,candidates in configs.items():
        best=-1;bestmodel=None
        for config,model in ([] if name in cached else candidates):
            start=time.time();fitargs={'sample_weight':tw}
            if name in ['logistic','svm_rbf']:fitargs={f'{model.steps[-1][0]}__sample_weight':tw}
            model.fit(x[tr],y[tr],**fitargs);z=raw_scores(model,x[va]);auc=float(roc_auc_score(y[va],z,sample_weight=vw))
            row=dict(model=name,config=config,validation_weighted_auc=auc,seconds=time.time()-start);history.append(row);write('validation_search.json',history);print(json.dumps(row),flush=True)
            if auc>best:best=auc;bestmodel=model;chosen=config
        if name in cached:
            old=joblib.load(O/f'models/{name}.joblib');bestmodel=old['model'];best=cached[name]['validation_weighted_auc'];chosen=cached[name]['config']
        z=raw_scores(bestmodel,x[ca]);clf=LogisticRegression(C=10,max_iter=2000).fit(z[:,None],y[ca],sample_weight=cw)
        slope=float(clf.coef_[0,0]);bias=float(clf.intercept_[0]);assert slope>0
        p=expit(slope*z+bias);high=next(float(t) for t in np.r_[np.linspace(.5,.999,500),1.0001] if np.average(p[y[ca]==0]>=t,weights=cw[y[ca]==0])<=.05)
        low=0.
        for t in np.linspace(.001,min(.499,high-1e-6),499):
            if np.average(p[y[ca]==1]<=t,weights=cw[y[ca]==1])<=.05:low=float(t)
        saved=dict(model=bestmodel,slope=slope,bias=bias,low=low,high=high,features=FEATURE_NAMES,version='aedes-classical-v1',family=name)
        path=O/f'models/{name}.joblib'
        if name in cached:
            assert slope==old['slope'] and bias==old['bias'] and low==old['low'] and high==old['high']
            saved=old
        else:joblib.dump(saved,path)
        item=dict(model=name,validation_weighted_auc=best,config=chosen,low=low,high=high,frozen_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),calibration=summary(f.iloc[ca],p,low,high)[0]);frozen.append(item)
        # Freeze each completed model before looking at its historical test metrics.
        write('frozen_models.json',frozen)
        p=expit(slope*raw_scores(bestmodel,x[test])+bias);tf=f.iloc[test]
        out=tf[['path','label','domain','group','start_s','quality_ok']].copy();out['model']=name;out['aedes_score']=p;out['state']=decisions(p,tf.quality_ok.to_numpy(),low,high);all_predictions.append(out)
        for domain in tf.domain.unique():
            mask=tf.domain.eq(domain).to_numpy();r,_=summary(tf[mask],p[mask],low,high);metrics.append(dict(model=name,domain=domain,**r))
        # Explanation from validation only, not held-out external/test data.
        rng=np.random.default_rng(20261004)
        subset=np.concatenate([rng.choice(np.flatnonzero(y[va]==c),min(128,int((y[va]==c).sum())),replace=False) for c in [0,1]])
        pi=permutation_importance(bestmodel,x[va[subset]],y[va[subset]],scoring='roc_auc',n_repeats=3,random_state=20261004,n_jobs=1,sample_weight=vw[subset])
        pd.DataFrame(dict(feature=FEATURE_NAMES,auc_drop=pi.importances_mean,std=pi.importances_std)).sort_values('auc_drop',ascending=False).to_csv(O/f'explanation_{name}.csv',index=False)
    winner=max(frozen,key=lambda r:r['validation_weighted_auc'])['model']
    write('training_audit.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),models=frozen,selected_by_validation=winner,train_windows=len(tr),validation_windows=len(va),calibration_windows=len(ca),arduino_compiled=False))
    pd.DataFrame(metrics).to_csv(O/'historical_metrics.csv',index=False);pd.concat(all_predictions).to_csv(O/'historical_predictions.csv.gz',index=False)
    print('ALL MODELS FROZEN; winner by validation:',winner,flush=True)
if __name__=='__main__':main()
