"""Descriptive spectra, not species classification or confirmed fundamental tracking."""
import json,subprocess,io
import numpy as np,pandas as pd,soundfile as sf
from scipy.signal import welch,resample_poly,find_peaks,spectrogram
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aedes_binary import ROOT,targets
from train_reproduction import sha
OUT=ROOT/'results/frequency_comparison_2026_10_03';OUT.mkdir(exist_ok=True)
ASSET=ROOT/'reports/assets';ASSET.mkdir(exist_ok=True)
f=pd.read_csv(ROOT/'results/reliability_2026_10_01/segments.csv',low_memory=False);f['y']=targets(f);items=[]
for domain in ['dryad','humbug','mosquitosong','ufrgs']:
 for label in [0,1]:
  pool=f[f.domain.eq(domain)&f.y.eq(label)].sort_values(['group','path']).drop_duplicates('group').head(12)
  for r in pool.itertuples():items.append(dict(path=r.path,source=domain,label=label,group=r.group))
p=json.load(open(ROOT/'results/reduced_databases_2026_10_02/unseen_selection.json'))
items.extend(dict(path=r['path'],source='CDMSC_D3',label=r['label'],group=r['path']) for r in p['files'])
items.append(dict(path='tmp/private-audio-test/download.audio',source='Drive',label=1,group='Drive'))
rows=[];spectra={};reps={};windows=[]
for i,item in enumerate(items):
 path=ROOT/item['path']
 if not path.exists():continue
 try:a,sr=sf.read(path,always_2d=True,dtype='float32')
 except sf.LibsndfileError:
  raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','wav','-acodec','pcm_f32le','pipe:1'],capture_output=True,check=True).stdout
  a,sr=sf.read(io.BytesIO(raw),always_2d=True,dtype='float32')
 a=a.mean(1);original_sr=sr
 if sr!=16000:g=np.gcd(sr,16000);a=resample_poly(a,16000//g,sr//g)
 size=15872 if len(a)>=15872 else 4800
 if len(a)<size:continue
 chunks=a[:len(a)//size*size].reshape(-1,size);idx=np.unique(np.linspace(0,len(chunks)-1,min(120,len(chunks))).astype(int));chunks=chunks[idx]
 hz,ps=welch(chunks,fs=16000,nperseg=min(4096,size),noverlap=min(2048,size//2),axis=1)
 valid=(np.abs(chunks).max(1)>=32/32768)&((np.abs(chunks)>=32700/32768).mean(1)<=.01);sel=(hz>=200)&(hz<=900);full=(hz>=20)&(hz<=7500);study=(hz>=500)&(hz<=700)
 peaks=hz[sel][ps[:,sel].argmax(1)];mean=ps[valid].mean(0) if valid.any() else ps.mean(0)
 pidx,_=find_peaks(mean);pidx=pidx[(hz[pidx]>=200)&(hz[pidx]<=3000)];strong=pidx[np.argsort(mean[pidx])[::-1][:8]]
 bandshare=ps[:,sel].sum(1)/np.maximum(ps[:,full].sum(1),1e-30);studyshare=ps[:,study].sum(1)/np.maximum(ps[:,sel].sum(1),1e-30)
 for j in range(len(chunks)):windows.append(dict(**item,window=int(idx[j]),quality_ok=bool(valid[j]),band_peak_hz=float(peaks[j]),band_share=float(bandshare[j]),study_share=float(studyshare[j])))
 rows.append(dict(**item,sha256=sha(path),original_sr=original_sr,seconds=len(a)/16000,analyzed_windows=len(chunks),valid_windows=int(valid.sum()),peak_200_900=float(hz[sel][mean[sel].argmax()]),peak_20_7500=float(hz[full][mean[full].argmax()]),peak_window_median=float(np.median(peaks[valid])) if valid.any() else None,share_200_900=float(np.median(bandshare[valid])) if valid.any() else None,share_500_700_within_band=float(np.median(studyshare[valid])) if valid.any() else None,strong_peaks_hz=json.dumps(hz[strong].round(1).tolist())))
 key=(item['source'],item['label']);spectra.setdefault(key,[]).append(mean/np.maximum(mean[full].sum(),1e-30))
 if item['label']==1 and item['source'] not in reps:reps[item['source']]=a
 if i%20==0:print('SPECTRA',i,len(items),flush=True)
r=pd.DataFrame(rows);r.to_csv(OUT/'files.csv',index=False);pd.DataFrame(windows).to_csv(OUT/'windows.csv.gz',index=False)
s=r.groupby(['source','label']).agg(files=('path','size'),peak_median_hz=('peak_200_900','median'),peak_q25=('peak_200_900',lambda v:v.quantile(.25)),peak_q75=('peak_200_900',lambda v:v.quantile(.75)),band_share=('share_200_900','median'),study_share=('share_500_700_within_band','median')).reset_index();s.to_csv(OUT/'summary.csv',index=False)
fig,ax=plt.subplots(figsize=(11,5))
for (source,label),values in spectra.items():
 if label==1:
  curve=np.mean(values,0);ax.plot(hz,10*np.log10(np.maximum(curve,1e-12)),label=source)
ax.axvspan(500,700,alpha=.1,color='green',label='Faixa descrita no TinyML');ax.set(xlim=(100,3000),ylim=(-85,-5),xlabel='Frequência (Hz)',ylabel='PSD relativa (dB)',title='Aedes: espectros normalizados; cada arquivo tem o mesmo peso');ax.legend(ncol=3);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ASSET/'frequency_aedes_sources.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(len(reps),1,figsize=(11,2.3*len(reps)),layout='constrained')
for ax,(source,a) in zip(np.atleast_1d(axes),reps.items()):
 # Fixed first five seconds; no cherry-picking by model or spectrum.
 hz,t,z=spectrogram(a[:80000],fs=16000,nperseg=1024,noverlap=768);z=10*np.log10(np.maximum(z,1e-20));ax.pcolormesh(t,hz,z-z.max(),vmin=-60,vmax=0,shading='auto',cmap='magma');ax.set(ylim=(100,2200),title=source,ylabel='Hz',xlabel='Tempo (s)')
fig.savefig(ASSET/'frequency_spectrograms.png',dpi=140);plt.close(fig)
print(s.to_string(index=False));print(r[r.source.eq('Drive')].to_string(index=False))
