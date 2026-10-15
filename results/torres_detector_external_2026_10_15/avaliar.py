"""Read-only inference of frozen ternary models on Torres example WAVs."""
from pathlib import Path
import sys,json,gzip,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent/"source_snapshot"))
import numpy as np
import pandas as pd
import torch
from predict_ternary import predict_audio
from predict_ternary_lstm import predict
from ternary_protocol import digest,read_audio,pcm_signatures
OUT=Path(__file__).resolve().parent
C=ROOT/'results/ternary_verified_2026_10_14_corrections';N=ROOT/'results/ternary_lstm_2026_10_15'
A=ROOT/'data/external/torres_2025/Healthy and Infected mosquitos'
protected=list((ROOT/'output/ifsc_revisao_oral_2026_10_15').glob('*.pdf'))+[C/'freeze.json',C/'models/final.joblib',N/'freeze.json']+[N/'final'/m/f for m in ['lstm','bilstm']for f in ['model.pt','scaler.npz','freeze.json']]
before={str(p.relative_to(ROOT)):digest(p)for p in protected}
with gzip.open(C/'pcm_signatures.json.gz','rt')as f:known={s for v in json.load(f).values()for s in v}
bytehash=set(pd.read_csv(C/'source_pcm_audit.csv').file_sha256)
(OUT/'analise-lacunas.txt').write_text('Pedido: testar detector ternário nos dados de Aedes infectados, sem alterar apresentação. Teste em WAVs originais disponíveis; 300 tensores mel 188x32 não são compatíveis com frontend ternário 55x64, nem substituem áudio bruto para atributos clássicos. Congelar modelos/limiares e verificar hashes e sobreposição exata de PCM. Dois exemplos WAV (infectado/saudável), sem inferência populacional de desempenho.\n')
torch.set_num_threads(4);rows=[];full={}
for filename,status in [('infected0001.wav','infectado'),('healthy0001.wav','saudável')]:
 path=A/filename;wave=read_audio(path,8000)
 overlap={'file_sha256':digest(path),'training_byte_duplicate':digest(path)in bytehash,'training_pcm_matches':len(pcm_signatures(wave)&known)}
 assert not overlap['training_byte_duplicate']and overlap['training_pcm_matches']==0,overlap
 record={'infection_status':status,'path':str(path.relative_to(ROOT)),'overlap':overlap,'models':{}}
 for name in ['classic','lstm','bilstm']:
  print('INFERENCE',filename,name,flush=True)
  result=predict_audio(C,path)if name=='classic'else predict(N,path,name=name,passes=100,device='cpu')
  record['models'][name]=result
  rows.append({'file':filename,'status':status,'model':name,'expected_class':'aedes_aegypti','prediction':result['decision'],'correct_aedes':result['decision']=='aedes_aegypti','windows':result['windows'],'seconds':result['real_seconds'],'mc_disagreement':result.get('mc_disagreement')})
 full[filename]=record
 (OUT/'predictions.json').write_text(json.dumps(full,ensure_ascii=False,indent=2))
pd.DataFrame(rows).to_csv(OUT/'predictions.csv',index=False)
after={str(p.relative_to(ROOT)):digest(p)for p in protected};assert before==after
summary={'scope':'Teste externo dos dois WAVs exemplificativos disponíveis, sem retreino ou calibração. Não é teste dos 300 tensores mel.','models':{name:{'aedes_detected':sum(r['correct_aedes']for r in rows if r['model']==name),'total':2,'infected_detected':next(r['correct_aedes']for r in rows if r['model']==name and r['status']=='infectado')}for name in ['classic','lstm','bilstm']},'model_and_local_pdf_hashes_unchanged':before==after,'protected_sha256':before,'presentation_remote_modified':False,'training_overlap_detected':False,'source_manifest_sha256':digest(ROOT/'data/external/torres_2025/manifest.json'),'evaluation_script_sha256':digest(Path(__file__))}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(pd.DataFrame(rows).to_string(index=False),flush=True)
