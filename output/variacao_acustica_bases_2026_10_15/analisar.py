from pathlib import Path
import hashlib,json,math,subprocess,io
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
run=ROOT/'results/ternary_verified_2026_10_14_corrections'
s=pd.read_csv(run/'segments.csv.gz',low_memory=False); e=pd.read_csv(run/'event_predictions.csv');old=pd.read_csv(ROOT/'results/sex_balanced_reliability_2026_10_01/segments.csv',low_memory=False)
records=[]
for _,ev in e.sort_values('group').iterrows():
 source=ev.source; target=int(ev.true_label)
 if target!=2:continue
 for path in sorted(s.loc[s.group.eq(ev.group),'path'].unique()):records.append(dict(path=path,source=source,group=ev.group,target=target,pred=int(ev.activity_gated_log_odds_cascade_event_pred)))
# Fixed negative controls: first 12 lexicographically ordered groups for each source/class.
for (source,target),part in e[e.true_label.ne(2)].groupby(['source','true_label']):
 for _,ev in part.sort_values('group').head(12).iterrows():
  for path in sorted(s.loc[s.group.eq(ev.group),'path'].unique()):records.append(dict(path=path,source=source,group=ev.group,target=int(target),pred=int(ev.activity_gated_log_odds_cascade_event_pred)))
# All source-transfer WAVs with valid retained windows, rather than handpicked examples.
se=pd.read_csv(run/'source_transfer_mosquitosong/event_predictions.csv')
for _,ev in se.iterrows():
 paths=sorted(old.loc[old.group.eq(ev.group),'path'].unique())
 for path in paths:records.append(dict(path=path,source='mosquitosong',group=ev.group,target=int(ev.true_label),pred=int(ev.activity_gated_log_odds_cascade_event_pred)))
# Additional already labelled external examples; no new detector inference.
f=pd.read_csv(ROOT/'results/frequency_comparison_2026_10_03/files.csv')
for _,row in f[f.source.isin(['CDMSC_D3'])].iterrows():records.append(dict(path=row.path,source=row.source,group=row.group,target=2 if row.label==1 else 1,pred=None))
for path in sorted((ROOT/'data/external/additional/ufrgs/dataset').rglob('*.wav')):
 records.append(dict(path=str(path.relative_to(ROOT)),source='ufrgs',group=str(path.relative_to(ROOT)),target=1 if path.parent.name=='No_Aedes_aegypti' else 2,pred=None))
for path in sorted((ROOT/'data/external/torres_2025/Healthy and Infected mosquitos').glob('*.wav')):
 records.append(dict(path=str(path.relative_to(ROOT)),source='Torres (exemplos WAV)',group=path.stem,target=2,pred=None))
# Keep one record per path; source labels and original manifests are preserved.
records=list({r['path']:r for r in records}.values());pd.DataFrame(records).to_csv(OUT/'manifesto.csv',index=False)
rows=[];errors=[]
for k,rec in enumerate(records):
 try:
  p=ROOT/rec['path']
  try:x,sr=sf.read(p,dtype='float64',always_2d=True)
  except RuntimeError:
   raw=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','wav','-acodec','pcm_s16le','pipe:1'],capture_output=True,check=True).stdout
   x,sr=sf.read(io.BytesIO(raw),dtype='float64',always_2d=True)
  x=x.mean(axis=1)
  g=math.gcd(sr,8000);x=signal.resample_poly(x,8000//g,sr//g);x=x-x.mean()
  y=signal.sosfilt(signal.butter(4,[200,2000],btype='bandpass',fs=8000,output='sos'),x)
  n=len(y)//800;rms=np.sqrt(np.mean(y[:n*800].reshape(n,800)**2,axis=1));db=20*np.log10(np.maximum(rms,1e-12))
  row=dict(rec,duration_s=len(x)/8000,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),full_variation_db=float(np.percentile(db,90)-np.percentile(db,10)))
  for block in [10,20]:
   chunks=[float(np.percentile(db[i:i+block],90)-np.percentile(db[i:i+block],10)) for i in range(0,len(db)-block+1,block)]
   row[f'blocks_{block//10}s']=len(chunks);row[f'variation_{block//10}s_db']=float(np.median(chunks)) if chunks else np.nan
  rows.append(row)
 except Exception as ex:errors.append(dict(rec,error=str(ex)))
 if (k+1)%250==0:print('processed',k+1,flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'metricas_por_arquivo.csv',index=False);(OUT/'erros.json').write_text(json.dumps(errors,indent=2))
# Equal weight per recording group, median of file medians, never treat clips as individuals.
groups=df.groupby(['source','target','group'],dropna=False).agg(files=('path','size'),files_ge_1s=('blocks_1s',lambda x:int((x>0).sum())),files_ge_2s=('blocks_2s',lambda x:int((x>0).sum())),variation_1s_db=('variation_1s_db','median'),variation_2s_db=('variation_2s_db','median'),pred=('pred','first')).reset_index();groups.to_csv(OUT/'metricas_por_grupo.csv',index=False)
summary=groups.groupby(['source','target']).agg(groups=('group','size'),files=('files','sum'),files_ge_1s=('files_ge_1s','sum'),files_ge_2s=('files_ge_2s','sum'),median_1s_db=('variation_1s_db','median'),q25_1s_db=('variation_1s_db',lambda x:x.quantile(.25)),q75_1s_db=('variation_1s_db',lambda x:x.quantile(.75)),median_2s_db=('variation_2s_db','median')).reset_index();summary.to_csv(OUT/'resumo.csv',index=False)
positive=groups[(groups.target==2)&groups.pred.notna()].copy();positive['detected']=positive.pred.eq(2)
comparison=positive.groupby(['source','detected']).agg(groups=('group','size'),median_1s_db=('variation_1s_db','median'),median_2s_db=('variation_2s_db','median')).reset_index();comparison.to_csv(OUT/'acertos_erros.csv',index=False)
# Representative median-file example for each source; no model-score or visual selection.
reps=[]
for source,part in df[df.target==2].groupby('source'):
 part=part[part.duration_s>=2].copy();mid=part.variation_1s_db.median();part['distance']=(part.variation_1s_db-mid).abs();reps.append(part.sort_values(['distance','path']).iloc[0])
fig,axes=plt.subplots(len(reps),2,figsize=(14,2.35*len(reps)),constrained_layout=True)
for i,row in enumerate(reps):
 x,sr=sf.read(ROOT/row.path,always_2d=True);x=x.mean(axis=1);g=math.gcd(sr,8000);x=signal.resample_poly(x,8000//g,sr//g);x-=x.mean();start=max(0,(len(x)/8000-2)/2);x=x[int(start*8000):int((start+2)*8000)]
 f,t,z=signal.stft(x,fs=8000,nperseg=1024,noverlap=896,boundary=None,padded=False);z=20*np.log10(np.maximum(abs(z),1e-12));z-=z.max()
 axes[i,0].pcolormesh(t,f,z,shading='auto',cmap='magma',vmin=-65,vmax=0,rasterized=True);axes[i,0].set(ylim=(100,2500),xlim=(0,2),ylabel='Hz',title=f'{row.source} · trecho central · Aedes')
 y=signal.sosfilt(signal.butter(4,[200,2000],btype='bandpass',fs=8000,output='sos'),x);n=len(y)//800;rms=np.sqrt(np.mean(y[:n*800].reshape(n,800)**2,axis=1));db=20*np.log10(np.maximum(rms,1e-12));db-=np.median(db)
 axes[i,1].plot((np.arange(n)+.5)*.1,db,color='#177d55');axes[i,1].set(xlim=(0,2),ylim=(-30,30),ylabel='dB relativos',title='Energia 200–2000 Hz');axes[i,1].grid(alpha=.2)
for ax in axes[-1]:ax.set_xlabel('Tempo (s)')
fig.suptitle('Outras bases · exemplos escolhidos pela mediana de variação, sem seleção por acerto\nEspectros relativos ao pico de cada trecho; escala comum. Torres: somente 2 WAVs disponíveis.',fontsize=13);fig.savefig(OUT/'espectrogramas_bases.png',dpi=140);plt.close(fig)
pd.DataFrame(reps).to_csv(OUT/'exemplos_figura.csv',index=False)
fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
parts=[(source,part) for source,part in groups[groups.target==2].groupby('source') if part.variation_1s_db.notna().any()]
ax.boxplot([part.variation_1s_db.dropna() for _,part in parts],tick_labels=[f'{source}\n{len(part)} grupos' for source,part in parts],showfliers=True)
ax.set(ylabel='Variação de intensidade em blocos de 1 s (dB)',title='Aedes · mediana por grupo · 200–2000 Hz\nGrupos são unidades do corpus, não indivíduos comprovadamente independentes');ax.grid(axis='y',alpha=.2);fig.savefig(OUT/'variacao_por_base.png',dpi=160);plt.close(fig)
print('SUMMARY\n'+summary.to_string(index=False));print('DETECTION\n'+comparison.to_string(index=False));print('files',len(df),'errors',len(errors),flush=True)
