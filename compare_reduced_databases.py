"""Retrospective source ablation; calibration never uses evaluation or Drive."""
import json,itertools
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from aedes_binary import ROOT,targets,weights
from corrected_validation import assert_disjoint,audio_windows
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from reliability_metrics import binary_metrics
from train_reproduction import sha,now
OUT=ROOT/'results/reduced_databases_2026_10_02'
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False))
def main():
 OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
 if (OUT/'freeze.json').exists():raise FileExistsError('Frozen experiment exists')
 f=pd.read_csv(ROOT/'results/reliability_2026_10_01/segments.csv',low_memory=False);assert_disjoint(f);y=targets(f)
 x=np.load(ROOT/'data/cache/reliability_2026_10_01/features.npy',mmap_mode='r')
 write('protocol.json',dict(created=now(),training_sources=['dryad','humbug','dryad+humbug'],methods=['logistic','svm_rbf','extra_trees'],selection='validation balanced accuracy; thresholds calibration balanced accuracy only',historical_tests=True,private_audio_training=False,arduino_compiled=False,limitation='Previously inspected tests and Drive are retrospective. Group identities depend on source metadata; no biological independence guarantee.'))
 rows=[];models={}
 for sources in [('dryad',),('humbug',),('dryad','humbug')]:
  name='+'.join(sources);mask=f.domain.isin(sources)&f.quality_ok&(y>=0)
  tr=np.flatnonzero(mask&f.split.eq('train'));ca=np.flatnonzero(mask&f.split.eq('calibration'));va=np.flatnonzero(mask&f.split.eq('validation'))
  assert set(y[tr])==set(y[ca])==set(y[va])=={0,1}
  for method in ['logistic','svm_rbf','extra_trees']:
   print('FIT',name,method,len(tr),flush=True)
   if method=='extra_trees':m=ExtraTreesClassifier(n_estimators=300,max_depth=16,min_samples_leaf=3,n_jobs=4,random_state=20261002)
   else:m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000) if method=='logistic' else SVC(C=1,gamma='scale',cache_size=1000))
   kw={'sample_weight':weights(f.iloc[tr])} if method=='extra_trees' else {m.steps[-1][0]+'__sample_weight':weights(f.iloc[tr])}
   m.fit(x[tr],y[tr],**kw)
   score=lambda a:m.predict_proba(a)[:,1] if method=='extra_trees' else m.decision_function(a)
   pc=score(x[ca]);cuts=np.unique(np.r_[np.quantile(pc,np.linspace(0,1,201)),pc.max()+1e-6]);cy=y[ca];cw=weights(f.iloc[ca]);best=max(cuts,key=lambda t:.5*(np.average(pc[cy==1]>=t,weights=cw[cy==1])+np.average(pc[cy==0]<t,weights=cw[cy==0])))
   key=name+'__'+method;joblib.dump(dict(model=m,cut=float(best),method=method),OUT/'models'/f'{key}.joblib');models[key]=(m,float(best),method)
   vm=binary_metrics(y[va],score(x[va]),np.ones(len(va),bool),best)
   rows.append(dict(model=key,scope='validation',training_windows=len(tr),training_groups=f.iloc[tr].group.nunique(),**vm))
 pd.DataFrame(rows).to_csv(OUT/'validation.csv',index=False)
 chosen=max(rows,key=lambda r:r['balanced_accuracy'])['model'];write('freeze.json',dict(time=now(),chosen_by_validation=chosen,hashes={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'models').glob('*')}))
 # Only now inspect retrospective tests and private positive control.
 for key,(m,cut,method) in models.items():
  score=lambda a:m.predict_proba(a)[:,1] if method=='extra_trees' else m.decision_function(a)
  for domain in ['dryad','humbug','mosquitosong','ufrgs']:
   ix=np.flatnonzero(f.domain.eq(domain)&f.split.isin(['test','challenge'])&(y>=0))
   if len(ix):rows.append(dict(model=key,scope='historical_'+domain,**binary_metrics(y[ix],score(x[ix]),f.iloc[ix].quality_ok,cut)))
 # Cached raw audio features shared across candidates, not trained representations.
 plan=json.loads((ROOT/'results/sex_balanced_reliability_2026_10_01/fresh_selection.json').read_text());audit=pd.read_csv(ROOT/'results/sex_balanced_reliability_2026_10_01/fresh_novelty_audit.csv');allowed=set(audit.loc[~audit.duplicate_file,'path'])
 datasets=[]
 for item in plan['files']:
  if item['path'] in allowed:datasets.append((item['path'],int(item['reference_aedes']),item['source'],item['group']))
 datasets.append(('tmp/private-audio-test/download.audio',1,'drive_positive_only','private'))
 pred=[]
 for i,(path,label,source,group) in enumerate(datasets):
  a,q,size=audio_windows(ROOT/path);xx=np.concatenate([features(a),extract_batch_bioacoustic_features(a)],1)
  for key,(m,cut,method) in models.items():
   p=m.predict_proba(xx)[:,1] if method=='extra_trees' else m.decision_function(xx)
   pred.extend(dict(model=key,path=path,source=source,group=group,label=label,quality_ok=bool(quality),score=float(s)) for s,quality in zip(p,q))
  if i%20==0:print('EVAL',i,len(datasets),flush=True)
 d=pd.DataFrame(pred);d.to_csv(OUT/'predictions.csv.gz',index=False)
 for key,sub in d.groupby('model'):
  scopes={s:a for s,a in sub.groupby('source')};scopes['all_external_retrospective']=sub[sub.source.ne('drive_positive_only')]
  for scope,a in scopes.items():rows.append(dict(model=key,scope=scope,**binary_metrics(a.label,a.score,a.quality_ok,models[key][1])))
 pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
 write('completion.json',dict(completed=now(),selected_before_evaluation=chosen,models=9,no_arduino=True,private_never_fit=True))
 print(pd.DataFrame(rows).query("scope in ['all_external_retrospective','drive_positive_only']") [['model','scope','aedes_precision','aedes_recall','specificity','balanced_accuracy']].to_string(index=False),flush=True)
if __name__=='__main__':main()
