"""Independently reload models/cache and reproduce external-source predictions."""
from pathlib import Path
import json
import numpy as np,pandas as pd,torch,joblib
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits
from train_torres_aedes_transfer import bank,describe
from ternary_protocol import digest,SEGMENTS
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/torres_aedes_other_datasets_2026_10_15';RUN=ROOT/'results/torres_aedes_transfer_2026_10_15'
def extract(frame,frontend):
 front,keep,_=bank(frontend);x=np.zeros((len(frame),len(keep)*5),np.float32)
 for bucket in ['long','short']:
  waves=np.load(ROOT/f'data/cache/reliability_2026_10_01/{bucket}_8k.npy',mmap_mode='r');ids=np.flatnonzero(frame.bucket.eq(bucket))
  for start in range(0,len(ids),128):
   ix=ids[start:start+128];raw=np.asarray(waves[frame.iloc[ix].wave_index.to_numpy(int)],dtype=np.float32)
   with torch.inference_mode():db=front(torch.from_numpy(resample_poly(raw,6,1,axis=1).copy())).numpy()
   x[ix]=describe(db,keep)
 return x
if __name__=='__main__':
 torch.set_num_threads(4);sel=pd.read_csv(OUT/'selected_external_windows.csv');assert sel.loc[sel.domain.ne('figshare'),'bucket'].eq('long').all()
 groups=pd.read_csv(OUT/'group_predictions.csv');clips=pd.read_csv(OUT/'figshare_clip_predictions.csv');full=pd.read_csv(SEGMENTS,low_memory=False);fig=full[full.domain.eq('figshare')].drop_duplicates(['path','start_s']).reset_index(drop=True);receipts=[]
 for name in ['baseline','with_torres']:
  b=joblib.load(RUN/f'{name}_model.joblib');x=extract(sel,b['frontend']);pooled=pd.DataFrame(x);pooled['group']=sel.group;pooled=pooled.groupby('group',sort=True).mean();expected=groups[groups.model.eq(name)&groups.evaluation_scope.eq('external_known_source')].set_index('group').loc[pooled.index]
  with threadpool_limits(limits=4):p=b['model'].predict_proba(pooled.to_numpy())
  assert np.allclose(p,expected[['p0','p1','p2']],atol=1e-12)
  fx=extract(fig,b['frontend']);e=clips[clips.model.eq(name)].set_index('path').loc[fig.path]
  with threadpool_limits(limits=4):p=b['model'].predict_proba(fx)
  assert np.allclose(p[:,2],e.p2,atol=1e-12);assert np.array_equal(p.argmax(1),e.prediction)
  receipts.append(dict(model=name,external_groups_verified=len(expected),figshare_clips_verified=len(e),max_probability_difference=float(np.max(np.abs(p[:,2]-e.p2.to_numpy())))))
 v=json.loads((OUT/'verification.json').read_text())
 for relative,expected in v['protected_sha256'].items():assert digest(ROOT/relative)==expected
 v['reproduction_checks']=receipts;v['verifier_sha256']=digest(Path(__file__));(OUT/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));print(json.dumps(receipts))
