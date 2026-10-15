"""Computer-only Aedes aegypti binary CNN. Unknown species are not negatives."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch,soundfile as sf
from torch import nn
from scipy.signal import resample_poly
from scipy.special import expit
from large_audio_model import LargeAudioCNN,SAMPLES

ROOT=Path(__file__).resolve().parent
CHECKPOINT=ROOT/'results/aedes_binary/models/aedes_binary.pt'
POSITIVE='Aedes aegypti'

def targets(frame):
    # An unresolved mosquito may be Aedes: never assign it to the negative class.
    return np.where(frame.label.eq('unknown_mosquito'),-1,frame.label.eq(POSITIVE).astype(int))

def weights(frame):
    table=frame[['label','domain','group']].copy()
    counts=table.groupby(['label','domain','group']).size()
    sources=table.drop_duplicates().groupby(['label','domain']).size()
    domains=table[['label','domain']].drop_duplicates().groupby('label').size()
    w=1/counts.reindex(pd.MultiIndex.from_frame(table)).to_numpy()
    w/=sources.reindex(pd.MultiIndex.from_frame(table[['label','domain']])).to_numpy()
    w/=table.label.map(domains).to_numpy()
    y=targets(frame)
    # Half positive mass; other half equally divided among negative labels,
    # then domains and recording groups within each label.
    for c in [0,1]:
        mask=y==c
        if mask.any():w[mask]=w[mask]/w[mask].sum()*.5
    return w/w.mean()

class AedesBinaryCNN(LargeAudioCNN):
    def __init__(self):
        super().__init__();self.species=nn.Linear(512,1)
        for p in self.presence.parameters():p.requires_grad_(False)
    def forward(self,wave,augment=False):
        logits,_=super().forward(wave,augment)
        return logits.squeeze(-1)

def load(checkpoint=CHECKPOINT,device='cpu'):
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    model=AedesBinaryCNN().to(device);model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved

def predict(model,wave,device,batch=96):
    out=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(wave),batch):
            x=torch.tensor(np.array(wave[i:i+batch],dtype='float32',copy=True),device=device)
            out.extend(model(x).cpu().numpy())
    return np.asarray(out)

def decisions(p,quality,low,high):
    # Binary target, with explicit abstention for insufficient evidence.
    return np.where(~np.asarray(quality),'AUDIO_INVALIDO',np.where(p>=high,'AEDES_AEGYPTI',np.where(p<=low,'NAO_AEDES_AEGYPTI','INCERTO')))

def audio_windows(path):
    wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(axis=1)
    if sr!=16000:
        d=np.gcd(sr,16000);wave=resample_poly(wave,16000//d,sr//d)
    n=len(wave)//SAMPLES
    if n<1:raise ValueError('Audio needs at least 0.992 seconds; no padding or repetition.')
    windows=wave[:n*SAMPLES].reshape(n,SAMPLES)
    pcm=np.rint(np.clip(windows,-1,32767/32768)*32768).astype('int16').astype('int32')
    valid=(np.abs(pcm).max(axis=1)>=32)&((np.abs(pcm)>=32700).sum(axis=1)*100<=SAMPLES)
    return windows,valid,sr

def evaluate_audio(path,model,saved,device):
    wave,valid,sr=audio_windows(path)
    score=expit(saved['slope']*predict(model,wave,device)+saved['bias'])
    return pd.DataFrame(dict(start_s=np.arange(len(wave))*.992,quality_ok=valid,aedes_score=score,
        forced_aedes=score>=.5,state=decisions(score,valid,saved['low'],saved['high']))),sr

def main():
    parser=argparse.ArgumentParser(description='Experimental binary detector; does not identify other mosquito species.')
    parser.add_argument('audio',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    model,saved=load(device=device);frame,sr=evaluate_audio(args.audio,model,saved,device)
    args.output.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output/'predictions.csv',index=False)
    summary=dict(windows=len(frame),original_sample_rate=sr,states=frame.state.value_counts().to_dict(),
        mean_aedes_score=float(frame.aedes_score.mean()),experimental=True,arduino_compiled=False,
        warning='Score is not a demonstrated probability of correctness outside calibration conditions; non-Aedes includes noise.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
