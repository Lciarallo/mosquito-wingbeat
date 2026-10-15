"""Post-freeze fixed unused acoustic CDMSC file test, no tuning."""
import json,struct,zlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from compare_reduced_databases import OUT
from aedes_binary import ROOT
from train_reproduction import now,sha
from corrected_validation import audio_windows
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from reliability_metrics import binary_metrics
import fetch_humbug as fetch
freeze=json.loads((OUT/'freeze.json').read_text())
for path,h in freeze['hashes'].items():assert sha(ROOT/path)==h
index=json.loads((ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_full.json').read_text())
dest=ROOT/'data/external/reduced_databases_fresh_2026_10_02';dest.mkdir(exist_ok=True)
seen={p.name for root in [ROOT/'data/external',ROOT/'tmp'] for p in root.rglob('*.wav') if dest not in p.parents}
chosen=[]
for sid,taxon,n in [(1,'Aedes aegypti',4),(2,'Aedes albopictus',2),(7,'Culex pipiens',2),(8,'Anopheles minimus',2),(9,'Anopheles stephensi',2)]:
 pool=sorted([e for e in index['entries'] if Path(e['name']).name.startswith(f'S_{sid}_D_3_') and e['name'].endswith('.wav') and Path(e['name']).name not in seen],key=lambda e:int(Path(e['name']).stem.split('_')[-1]))[:n]
 chosen.extend(dict(entry=e,label=int(sid==1),taxon=taxon) for e in pool)
plan=dict(selection_utc=now(),freeze=freeze,rule='First unused numeric file IDs, acoustic D3, 4 Aedes and 2 per negative species; no inference before selection.',files=chosen,limitations=['File labels, no temporal flight annotation','Unknown original individuals/sessions','Small preliminary sample'])
(OUT/'unseen_selection.json').write_text(json.dumps(plan,indent=2))
models={p.stem:joblib.load(p) for p in (OUT/'models').glob('*.joblib')};rows=[]
known=set(pd.read_csv(ROOT/'results/reliability_2026_10_01/segments.csv',low_memory=False).content_sha256.dropna())
for item in chosen:
 e=item['entry'];path=dest/Path(e['name']).name
 raw=fetch.byte_range(index['file']['links']['self'],e['offset'],e['offset']+30+len(e['name'])+e['compressed']+1024)
 h=struct.unpack_from('<4s5H3I2H',raw);start=30+h[-2]+h[-1];raw=raw[start:start+e['compressed']];raw=zlib.decompress(raw,-15) if e['method']==8 else raw
 assert zlib.crc32(raw)==e['crc'];path.write_bytes(raw);assert sha(path) not in known
 item.update(path=str(path.relative_to(ROOT)),sha256=sha(path));a,q,_=audio_windows(path);x=np.concatenate([features(a),extract_batch_bioacoustic_features(a)],1)
 for name,s in models.items():
  p=s['model'].predict_proba(x)[:,1] if s['method']=='extra_trees' else s['model'].decision_function(x)
  rows.extend(dict(model=name,group=path.stem,label=item['label'],taxon=item['taxon'],score=float(v),quality_ok=bool(ok)) for v,ok in zip(p,q))
 print('NEW FILE',path.name,flush=True)
(OUT/'unseen_selection.json').write_text(json.dumps(plan,indent=2));d=pd.DataFrame(rows);d.to_csv(OUT/'unseen_predictions.csv.gz',index=False)
metrics=[dict(model=k,scope='new_CDMSC_D3',files=a.group.nunique(),**binary_metrics(a.label,a.score,a.quality_ok,models[k]['cut'])) for k,a in d.groupby('model')];pd.DataFrame(metrics).to_csv(OUT/'unseen_metrics.csv',index=False)
print(pd.DataFrame(metrics)[['model','aedes_precision','aedes_recall','specificity','balanced_accuracy']].to_string(index=False))
