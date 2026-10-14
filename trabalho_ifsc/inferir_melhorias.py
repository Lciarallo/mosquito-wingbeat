"""Inferência dos modelos revistos, com escolha prévia pela validação."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import softmax
from comparar_melhorias import ROOT, CNNFrequencia, MLPProfunda, frontend, predict_logits


def infer(audio, directory, name=None):
    if name is None:
        name=json.loads((directory/'selection.json').read_text())['selected_model']
    saved=torch.load(directory/f'{name}.pt',map_location='cpu',weights_only=True)
    model=(CNNFrequencia if name=='cnn_frequency' else MLPProfunda)(len(saved['classes']))
    model.load_state_dict(saved['state_dict'])
    wave,sr=sf.read(audio,dtype='float32',always_2d=True);wave=wave.mean(axis=1)
    if sr!=16000:
        div=np.gcd(sr,16000);wave=resample_poly(wave,16000//div,sr//div)
    chunks=np.asarray([wave[s:s+15872] for s in range(0,len(wave)-15872+1,15872)])
    if not len(chunks):raise ValueError('O áudio precisa ter ao menos 0,992 s; não há repetição ou preenchimento.')
    x=frontend(chunks,'cpu')
    if name=='deep_mlp':
        x=np.concatenate([x.mean(axis=3).reshape(len(x),-1),x.std(axis=3).reshape(len(x),-1)],axis=1)
        x=joblib.load(directory/'mlp_scaler.joblib').transform(x).astype('float32')
    probabilities=softmax(predict_logits(model,x,'cpu'),axis=1).mean(axis=0)
    order=np.argsort(probabilities)[-3:][::-1]
    return dict(model=name,audio=str(audio),windows=len(chunks),forced_choice=True,
                presence_not_validated=True,
                top3=[dict(species=saved['classes'][i],score=float(probabilities[i])) for i in order])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio',type=Path)
    p.add_argument('--directory',type=Path,default=ROOT/'results/ifsc_melhorias_2026_10_11')
    p.add_argument('--model',choices=['deep_mlp','cnn_frequency'])
    args=p.parse_args()
    print(json.dumps(infer(args.audio,args.directory,args.model),ensure_ascii=False,indent=2))
