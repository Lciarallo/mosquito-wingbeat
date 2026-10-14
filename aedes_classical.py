"""Gain-invariant acoustic features and classical binary ML; computer only."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,soundfile as sf,joblib
from scipy.fft import rfft,dct
from scipy.signal import resample_poly
from aedes_binary import ROOT,decisions

O=ROOT/'results/aedes_classical';FFT=2048;HOP=256;SR=16000
BANDS=[f'band_{125*i}_{125*(i+1)}' for i in range(1,32)]
MEASURE_NAMES=BANDS+[f'cepstrum_{i}' for i in range(1,20)]+['centroid_hz','bandwidth_hz','rolloff85_hz','flatness','entropy','peak_hz','low_peak_hz','harmonic2_ratio','harmonic3_ratio','low_band_ratio','spectral_flux']
FEATURE_NAMES=[f'{m}_{stat}' for stat in ['mean','std'] for m in MEASURE_NAMES]

def features(waves,batch=128):
    waves=np.atleast_2d(waves);size=waves.shape[1]
    if size<4800:raise ValueError('Need at least 0.3 s of real audio; no padding or repetition.')
    starts=np.arange(0,size-FFT+1,HOP);freq=np.fft.rfftfreq(FFT,1/SR);valid=(freq>=125)&(freq<4000);fv=freq[valid]
    window=np.hanning(FFT).astype('float32');result=[]
    for start in range(0,len(waves),batch):
        x=np.array(waves[start:start+batch],dtype='float32',copy=True);x-=x.mean(axis=1,keepdims=True)
        x/=np.sqrt(np.mean(x*x,axis=1,keepdims=True)).clip(1e-6)
        frames=np.stack([x[:,i:i+FFT] for i in starts],axis=1)
        power=np.abs(rfft(frames*window,axis=-1))**2
        p=power[...,valid];p=p/p.sum(axis=-1,keepdims=True).clip(1e-20)
        energy=np.stack([p[..., (fv>=125*i)&(fv<125*(i+1))].sum(axis=-1) for i in range(1,32)],axis=-1)
        log=np.log10(energy.clip(1e-8));cep=dct(log,type=2,norm='ortho',axis=-1)[...,1:20]
        centroid=(p*fv).sum(-1);bw=np.sqrt((p*(fv-centroid[...,None])**2).sum(-1))
        roll=fv[np.argmax(np.cumsum(p,axis=-1)>=.85,axis=-1)]
        flat=np.exp(np.log(p.clip(1e-12)).mean(-1))*len(fv);entropy=-(p*np.log(p.clip(1e-12))).sum(-1)/np.log(len(fv))
        peak=fv[p.argmax(-1)];low=(fv>=200)&(fv<=900);lowp=p[...,low];lowfreq=fv[low][lowp.argmax(-1)]
        peakamp=lowp.max(-1).clip(1e-8)
        h2=(p*(np.abs(fv[None,None,:]-2*lowfreq[...,None])<=16)).sum(-1)/peakamp
        h3=(p*(np.abs(fv[None,None,:]-3*lowfreq[...,None])<=16)).sum(-1)/peakamp
        flux=np.concatenate([np.zeros((*p.shape[:1],1)),np.sqrt(((p[:,1:]-p[:,:-1])**2).mean(-1))],axis=1)
        stat=np.stack([centroid,bw,roll,flat,entropy,peak,lowfreq,h2,h3,lowp.sum(-1),flux],axis=-1)
        allf=np.concatenate([log,cep,stat],axis=-1)
        result.append(np.concatenate([allf.mean(1),allf.std(1)],axis=1).astype('float32'))
    answer=np.concatenate(result);assert answer.shape[1]==len(FEATURE_NAMES) and np.isfinite(answer).all()
    return answer

def raw_scores(model,x):
    if hasattr(model,'decision_function'):return model.decision_function(x)
    p=model.predict_proba(x)[:,1].clip(1e-6,1-1e-6)
    return np.log(p/(1-p))

def calibrated(saved,x):
    from scipy.special import expit
    return expit(saved['slope']*raw_scores(saved['model'],x)+saved['bias'])

def audio_features(path):
    w,sr=sf.read(path,dtype='float32',always_2d=True);w=w.mean(axis=1)
    if sr!=SR:
        g=np.gcd(sr,SR);w=resample_poly(w,SR//g,sr//g)
    size=15872 if len(w)>=15872 else 4800;n=len(w)//size
    if n<1:raise ValueError('Less than 0.3 seconds of actual audio.')
    wave=w[:n*size].reshape(n,size);pcm=np.rint(np.clip(wave,-1,32767/32768)*32768).astype('int16').astype('int32')
    valid=(np.abs(pcm).max(1)>=32)&((np.abs(pcm)>=32700).sum(1)*100<=size)
    return features(wave),valid,size,sr

def main():
    ap=argparse.ArgumentParser();ap.add_argument('audio',type=Path);ap.add_argument('--model',choices=['logistic','svm_rbf','extra_trees','gradient_boosting']);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    selected=a.model or json.loads((O/'training_audit.json').read_text())['selected_by_validation']
    saved=joblib.load(O/f'models/{selected}.joblib');x,q,size,sr=audio_features(a.audio);p=calibrated(saved,x)
    f=pd.DataFrame(dict(start_s=np.arange(len(x))*size/SR,aedes_score=p,quality_ok=q,state=decisions(p,q,saved['low'],saved['high'])))
    a.output.mkdir(parents=True,exist_ok=True);f.to_csv(a.output/'predictions.csv',index=False)
    result=dict(model=selected,windows=len(f),states=f.state.value_counts().to_dict(),experimental=True,arduino_compiled=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
