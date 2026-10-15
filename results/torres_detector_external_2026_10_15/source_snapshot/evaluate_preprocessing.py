"""Paired exploratory filter sensitivity: fixed weights, thresholds and methods."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np,pandas as pd,soundfile as sf,torch
from scipy.signal import resample_poly
from scipy.special import softmax,expit
from audio_preprocessing import apply,METHODS
from large_audio_model import load_model,predict,SAMPLES
from train_multisource import weighted_metrics
from improve_device import atomic_json,file_hash
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/preprocessing';PRIVATE=ROOT/'tmp/private-audio-test/preprocessing'

def load_record(path,domain,label):
 wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(axis=1);g=np.gcd(sr,16000)
 if sr!=16000:wave=resample_poly(wave,16000//g,sr//g)
 n=len(wave)//SAMPLES;waves=wave[:n*SAMPLES].reshape(n,SAMPLES)
 pcm=np.rint(np.clip(waves,-1,32767/32768)*32768).astype('int16');absolute=np.abs(pcm.astype('int32'))
 quality=(absolute.max(axis=1)>=32)&((absolute>=32700).sum(axis=1)*100<=SAMPLES)
 frame=pd.DataFrame(dict(path=[str(path)]*n,group=[str(path)]*n,domain=[domain]*n,label=[label]*n,start_s=np.arange(n)*SAMPLES/16000,step_s=[SAMPLES/16000]*n,quality_ok=quality,id=[path.stem]*n))
 return frame,waves

def infer(model,waves,method,saved,device):
 ls,gs=[],[]
 for start in range(0,len(waves),96):
  filtered=apply(np.array(waves[start:start+96]),method)
  assert filtered.shape==waves[start:start+96].shape and np.isfinite(filtered).all()
  l,g=predict(model,filtered,device);ls.append(l);gs.append(g)
 logits=np.concatenate(ls);gates=np.concatenate(gs)
 return softmax(logits/saved['temperature'],axis=1),expit(saved['presence_slope']*gates+saved['presence_bias'])

def main():
 OUT.mkdir(exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
 checkpoint=ROOT/'results/multisource/models/multisource_cnn.pt';sha=file_hash(checkpoint);model,s=load_model(checkpoint,device)
 atomic_json(OUT/'protocol.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),model_sha256=sha,methods=METHODS,methods_defined_before_results=True,weights_and_thresholds_fixed=True,original_quality_guards_preserved=True,model_already_removes_dc_and_normalizes_rms=True,no_retraining=True,no_Arduino_compilation=True,exploratory_previously_examined_tests=True,no_filtered_audio_published=True,noncausal_offline_filters=True,no_waveform_padding_repetition_duration_extension=True))
 base=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);ix=np.flatnonzero(base.split.isin(['test','challenge']));frame=base.iloc[ix].reset_index(drop=True)
 cached={b:np.load(ROOT/f'data/cache/multisource/{b}.npy',mmap_mode='r') for b in ['long','short']}
 private,pw=load_record(ROOT/'tmp/private-audio-test/download.audio','private_user_reference','Aedes aegypti')
 fresh_selection=json.loads((ROOT/'results/multisource/fresh_external/selection.json').read_text());fresh_frames=[];fresh_waves=[]
 for item in fresh_selection['files']:
  f,w=load_record(ROOT/item['path'],'fresh_humbug_retrospective','Aedes aegypti');fresh_frames.append(f);fresh_waves.append(w)
 fresh=pd.concat(fresh_frames,ignore_index=True);fw=np.concatenate(fresh_waves)
 summaries=[];private_summary=[];negatives=[]
 for method in METHODS:
  p=np.zeros((len(frame),20),dtype='float32');g=np.zeros(len(frame),dtype='float32')
  for bucket in ['long','short']:
   mask=frame.bucket.eq(bucket).to_numpy()
   if mask.any():p[mask],g[mask]=infer(model,cached[bucket][frame.loc[mask,'wave_index'].to_numpy().astype(int)],method,s,device)
  rows,_,pred=weighted_metrics(frame,p,g,s['classes'],s['presence_threshold'],s['species_threshold'],method);summaries.extend(rows)
  pred.to_csv(OUT/f'{method}_heldout.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
  n=pred[pred.label.eq('not_aedes')];negatives.append(dict(method=method,windows=len(n),forced_Aedes=int(n.forced_species.eq('Aedes aegypti').sum()),emitted_Aedes=int(n.identified_species.eq('Aedes aegypti').sum())))
  p,g=infer(model,fw,method,s,device);rows,_,pred=weighted_metrics(fresh,p,g,s['classes'],s['presence_threshold'],s['species_threshold'],method);summaries.extend(rows);pred.to_csv(OUT/f'{method}_five_aedes.csv',index=False)
  p,g=infer(model,pw,method,s,device);rows,_,pred=weighted_metrics(private,p,g,s['classes'],s['presence_threshold'],s['species_threshold'],method)
  pred.to_csv(PRIVATE/f'{method}_predictions.csv',index=False)
  emitted=pred.identified_species.ne('-');right=pred.identified_species.eq('Aedes aegypti');present=private.quality_ok & (g>=s['presence_threshold'])
  state=np.where(~private.quality_ok,'AUDIO_INVALIDO',np.where(~present,'SEM_EVIDENCIA',np.where(~emitted,'INCERTO','IDENTIFICACAO_PROVISORIA')))
  ps=dict(method=method,windows=len(private),forced_Aedes=int(pred.forced_species.eq('Aedes aegypti').sum()),correct_identifications=int(right.sum()),wrong_identifications=int((emitted&~right).sum()),presence_candidate_windows=int(present.sum()),emitted_species_counts=pred.loc[emitted,'identified_species'].value_counts().to_dict(),state_counts=pd.Series(state).value_counts().to_dict(),label_source='user_reported_species_not_independently_verified')
  private_summary.append(ps);print(json.dumps(ps,ensure_ascii=False),flush=True)
  pd.DataFrame(summaries).to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(negatives).to_csv(OUT/'ufrgs_negatives.csv',index=False);atomic_json(OUT/'private_audio_summary.json',private_summary)
 assert file_hash(checkpoint)==sha
 atomic_json(OUT/'audit.json',dict(model_unchanged=True,thresholds_unchanged=True,all_methods_evaluated=True,raw_private_audio_unchanged_sha256=file_hash(ROOT/'tmp/private-audio-test/download.audio'),no_retraining=True,no_Arduino_compilation=True))
 print('Finished fixed-filter comparison',flush=True)
if __name__=='__main__':main()
