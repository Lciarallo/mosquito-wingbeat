"""Reproduce saved predictions and audit identities for Torres Aedes transfer."""
from pathlib import Path
import json,copy
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from predict_torres_aedes_transfer import predict
from torres_lstm import digest
ROOT=Path(__file__).resolve().parent

def verify(w):
 torch.set_num_threads(4);rows=[]
 for name in ['infected0001.wav','healthy0001.wav']:
  for variant in ['baseline','with_torres']:
   with threadpool_limits(limits=4):r=predict(w,ROOT/'data/external/torres_2025/Healthy and Infected mosquitos'/name,variant)
   rows.append(r)
 (w/'wav_predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 part=pd.read_csv(w/'partitions.csv');pred=pd.read_csv(w/'test_predictions.csv');sel=json.loads((w/'selection_frozen.json').read_text())['selected'];checks=[]
 source=pd.read_csv(ROOT/'results/ternary_lstm_2026_10_15/segments.csv.gz',low_memory=False);truth=source.groupby('group').true_label.max();primary=part[part.origin.eq('primary')]
 assert np.array_equal(primary.target,primary.group.map(truth));assert primary.groupby('target').size().to_dict()=={0:426,1:1008,2:60}
 for name,c in sel.items():
  bundle=joblib.load(w/f'{name}_model.joblib')
  with np.load(w/f"features_{c['frontend']}.npz")as z:
   for origin in ['primary','torres']:
    sub=part[part.origin.eq(origin)].reset_index(drop=True);mask=sub.split.eq('test').to_numpy();expected=pred[pred.model.eq(name)&pred.source.eq(origin)]
    assert sub.group[mask].tolist()==expected.group.tolist()
    with threadpool_limits(limits=4):p=bundle['model'].predict_proba(z[origin][mask])
    assert np.allclose(p,expected[['p0','p1','p2']].to_numpy(),atol=1e-12);checks.append(f'{name}/{origin}: all saved probabilities reproduced')
 for origin in ['primary','torres']:
  sub=part[part.origin.eq(origin)];assert sub.groupby('pcm_block').split.nunique().max()==1
  checks.append(origin+': partition identities disjoint')
 assert set(part.loc[part.origin.eq('torres'),'target'])=={2}
 part.groupby(['origin','split','target']).size().reset_index(name='count').to_csv(w/'partition_counts.csv',index=False)
 v=json.loads((w/'verification.json').read_text())
 for relative,expected in v['protected_hashes'].items():assert digest(ROOT/relative)==expected
 # Preserve the previous external test's checkpoint hashes too.
 old=json.loads((ROOT/'results/torres_detector_external_2026_10_15/summary.json').read_text())
 for relative,expected in old['protected_sha256'].items():assert digest(ROOT/relative)==expected
 v.update(prediction_reproduction=checks,saved_model_sha256={name:digest(w/f'{name}_model.joblib')for name in sel},all_prior_external_test_checkpoints_unchanged=True,primary_event_truth_reproduced=True,inference_script_sha256=digest(ROOT/'predict_torres_aedes_transfer.py'),verifier_script_sha256=digest(Path(__file__)))
 (w/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));return v
if __name__=='__main__':
 w=ROOT/'results/torres_aedes_transfer_2026_10_15';print(json.dumps(verify(w),ensure_ascii=False,indent=2))
