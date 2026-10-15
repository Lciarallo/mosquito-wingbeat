from pathlib import Path
import json
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
items=[('MosquitoSong','mosquitosong_aedes_femea.wav'),('Dryad','dryad_aedes_acerto.wav'),('HumBug','humbug_aedes_acerto.wav')]
data=[]; metrics=[]
for name,filename in items:
 sr,x=wavfile.read(BASE/filename)
 if np.issubdtype(x.dtype,np.integer): x=x.astype(float)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
 else:x=x.astype(float)
 if x.ndim>1:x=x.mean(axis=1)
 from math import gcd
 g=gcd(sr,8000);x=signal.resample_poly(x,8000//g,sr//g);sr=8000
 x=x-x.mean(); y=signal.sosfilt(signal.butter(4,[200,2000],btype='bandpass',fs=sr,output='sos'),x)
 n=800;count=len(y)//n
 rms=np.sqrt(np.mean(y[:count*n].reshape(count,n)**2,axis=1)); db=20*np.log10(np.maximum(rms,1e-12));times=(np.arange(count)+.5)*.1
 f,t,z=signal.stft(x,fs=sr,window='hann',nperseg=1024,noverlap=896,boundary=None,padded=False)
 spec=20*np.log10(np.maximum(abs(z),1e-12));spec-=spec.max()
 metrics.append({'source':name,'duration_s':len(x)/sr,'rms_band_hz':[200,2000],'rms_window_s':.1,'rms_p90_minus_p10_db':float(np.percentile(db,90)-np.percentile(db,10)),'rms_cv':float(rms.std()/rms.mean()),'two_second_block_variation_db':[float(np.percentile(db[i:i+20],90)-np.percentile(db[i:i+20],10)) for i in range(0,len(db)-19,20)]})
 data.append((name,len(x)/sr,f,t,spec,times,db))
for matched in [False,True]:
 fig,axes=plt.subplots(3,2,figsize=(14,9),gridspec_kw={'width_ratios':[1.6,1]},constrained_layout=True)
 for i,(name,duration,f,t,spec,rt,db) in enumerate(data):
  start=max(0,(duration-2)/2) if matched else 0;end=start+2 if matched else duration
  axes[i,0].pcolormesh(t-start,f,spec,shading='auto',vmin=-65,vmax=0,cmap='magma',rasterized=True)
  axes[i,0].set(xlim=(0,end-start),ylim=(100,2500),ylabel='Frequência (Hz)',title=f'{name} · '+('2 s centrais' if matched else f'arquivo completo: {duration:.2f} s'))
  # Relative to the median of the full recording; no audio gain change.
  axes[i,1].plot(rt-start,db-np.median(db),color='#177d55',linewidth=1)
  axes[i,1].set(xlim=(0,end-start),ylim=(-35,25),ylabel='Intensidade relativa (dB)',title='Energia em 200–2000 Hz · janelas de 100 ms')
  axes[i,1].grid(alpha=.2)
 for ax in axes[-1]:ax.set_xlabel('Tempo (s)')
 fig.suptitle('Comparação acústica · espectros relativos ao pico de cada arquivo\nMesma escala de frequência e cor; energia inclui voo e ruído, sem separação de fontes',fontsize=13)
 fig.savefig(OUT/('comparacao_2s.png' if matched else 'comparacao_completa.png'),dpi=160);plt.close(fig)
(OUT/'metricas.json').write_text(json.dumps(metrics,indent=2))
for m in metrics:
 b=m['two_second_block_variation_db']; print(m['source'],'duration',round(m['duration_s'],2),'full P90-P10 dB',round(m['rms_p90_minus_p10_db'],2),'2s-block median',round(float(np.median(b)),2),'2s-block range',round(min(b),2),round(max(b),2))
