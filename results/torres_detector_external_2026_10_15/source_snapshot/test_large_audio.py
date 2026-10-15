"""Evaluate one unmodified recording with the frozen computer-only CNN."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.special import softmax,expit
from large_audio_model import load_model,predict,SAMPLES
from train_expanded import decide
from improve_device import file_hash,atomic_json

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('audio',type=Path)
    parser.add_argument('--species',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--label-source',default='dataset_authors')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'results/large_cnn/models/large_cnn.pt')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.checkpoint;before=file_hash(checkpoint)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model,saved=load_model(checkpoint,device)
    wave,sr=sf.read(args.audio,dtype='float32',always_2d=True);channels=wave.shape[1];wave=wave.mean(axis=1)
    divisor=np.gcd(sr,16000)
    if sr!=16000:wave=resample_poly(wave,16000//divisor,sr//divisor)
    n=len(wave)//SAMPLES
    if n<1:raise ValueError('No complete window; padding, repetition and concatenation are prohibited.')
    windows=wave[:n*SAMPLES].reshape(n,SAMPLES)
    # Match the recording quality guards; normalization is inside the CNN only.
    pcm=np.rint(np.clip(windows,-1,32767/32768)*32768).astype('int16')
    peak=np.abs(pcm.astype('int32')).max(axis=1);clipped=(np.abs(pcm.astype('int32'))>=32700).sum(axis=1)
    quality=(peak>=32)&(clipped*100<=SAMPLES)
    frame=pd.DataFrame(dict(path=[str(args.audio)]*n,start_s=np.arange(n)*SAMPLES/16000,
        step_s=[SAMPLES/16000]*n,quality_ok=quality))
    logits,gates=predict(model,windows,device)
    p=softmax(logits/saved['temperature'],axis=1)
    g=expit(saved['presence_slope']*gates+saved['presence_bias'])
    _,confirmed=decide(frame,p,g,saved['presence_threshold'],saved['species_threshold'])
    present=quality&(g>=saved['presence_threshold'])
    state=np.where(~quality,'AUDIO_INVALIDO',np.where(~present,'SEM_EVIDENCIA',np.where(confirmed<0,'INCERTO','IDENTIFICACAO_PROVISORIA')))
    frame['state']=state;frame['forced_species']=[saved['classes'][i] for i in p.argmax(axis=1)]
    frame['identified_species']=[saved['classes'][i] if i>=0 else '-' for i in confirmed]
    frame['presence_score']=g;frame['species_score']=p.max(axis=1)
    frame.to_csv(args.output/'predictions.csv',index=False)
    reference=saved['classes'].index(args.species);correct=confirmed==reference;emitted=confirmed>=0
    report=dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),source=args.source,
        reference_species=args.species,label_source=args.label_source,model_sha256=before,
        model_unchanged=before==file_hash(checkpoint),original_audio_sha256=file_hash(args.audio),
        original_sample_rate=sr,original_channels=channels,duration_s=len(wave)/16000,
        complete_windows=n,analyzed_duration_s=n*SAMPLES/16000,unprocessed_tail_s=(len(wave)%SAMPLES)/16000,
        state_counts=frame.state.value_counts().to_dict(),
        emitted_species_counts=frame.loc[emitted,'identified_species'].value_counts().to_dict(),
        correctly_identified_windows=int(correct.sum()),incorrectly_identified_windows=int((emitted&~correct).sum()),
        forced_correct_windows=int((p.argmax(axis=1)==reference).sum()),
        presence_candidate_windows=int(present.sum()),resampling_applied=sr!=16000,
        external_volume_normalization=False,internal_per_window_rms_normalization=True,
        no_padding_or_repetition=True,noise_filtering=False,arduino_compiled=False,physical_board_tested=False)
    atomic_json(args.output/'analysis.json',report);print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
