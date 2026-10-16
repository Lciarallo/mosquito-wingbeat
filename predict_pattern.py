"""Apply a frozen, non-recurrent common-Mel classifier to real audio."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

from pattern_classifiers import SpectrogramCNN, AcousticMLP, temporal_statistics
from torres_lstm import MelSequence, digest, predict_probabilities
from train_ternary_lstm import GroupPool
from train_pattern_classifiers import ml_proba

LABELS=['ruido','outro_mosquito','aedes_aegypti']


def audio_sequences(path):
    wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
    if not np.isfinite(wave).all():raise ValueError('Audio contains nonfinite values.')
    divisor=np.gcd(sr,48000)
    wave=resample_poly(wave,48000//divisor,sr//divisor) if sr!=48000 else wave
    # Training originals: 7936 real samples at 8 kHz, then resampled to 48 kHz.
    samples=47616;chunks=[wave[i:i+samples] for i in range(0,len(wave),samples)]
    frontend=MelSequence(sample_rate=48000,n_fft=1024,hop=256,bands=32,fmin=0,fmax=24000).eval()
    sequences=[]
    for chunk in chunks:
        if len(chunk)<1024+12*256:continue
        with torch.inference_mode():db=frontend(torch.tensor(chunk[None])).numpy()[:,2:-2,:16]
        if db.shape[1]>179:
            start=(db.shape[1]-179)//2;db=db[:,start:start+179]
        db=np.clip(db-db.max((1,2),keepdims=True),-80,0).astype('float32')
        if db.shape[1]>=9:sequences.append(db)
    if not sequences:raise ValueError('At least about 0.09 s of real audio is needed; no padding/repetition.')
    return sequences


def predict(model_dir,audio):
    model_dir=Path(model_dir);manifest=json.loads((model_dir/'manifest.json').read_text())
    name=manifest['model'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name in ['cnn','mlp']:
        assert digest(model_dir/'model.pt')==manifest['model_sha256']
        checkpoint=torch.load(model_dir/'model.pt',map_location='cpu',weights_only=True)
        model=(SpectrogramCNN if name=='cnn' else AcousticMLP)(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict']);z=np.load(model_dir/'scaler.npz')
        assert digest(model_dir/'scaler.npz')==manifest['scaler_sha256']
    else:
        assert digest(model_dir/'model.joblib')==manifest['model_sha256']
        bundle=joblib.load(model_dir/'model.joblib')
    probabilities=[]
    for sequence in audio_sequences(audio):
        features=sequence if name=='cnn' else temporal_statistics(sequence)
        if name in ['cnn','mlp']:
            x=((features-z['mean'])/z['std']).astype('float32')
            p=predict_probabilities(model,x,device)
        else:p=ml_proba(bundle['model'],bundle['scaler'].transform(features))
        probabilities.append(p[0])
    p=np.asarray(probabilities)
    frame=pd.DataFrame(dict(group='audio',pcm_block='audio',domain='external',true_label=0),index=range(len(p)))
    pooled=GroupPool(frame).probabilities(p)[0];label=int(pooled.argmax())
    return dict(audio=str(audio),model=name,prediction=LABELS[label],prediction_index=label,
                scores=dict(zip(LABELS,pooled.tolist())),windows=len(p),rule='argmax',
                scores_are_calibrated_probabilities=False,
                frontend='approximate common16Mel frontend; short real windows accepted without padding')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--audio',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    print(json.dumps(predict(args.model_dir,args.audio),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
