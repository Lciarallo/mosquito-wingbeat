"""Evaluate a new complete recording with frozen baseline and expanded models."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import arduino_frontend as frontend
from train_expanded import ROOT, WINDOW, infer_portable, decide
from improve_device import file_hash, atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('audio',type=Path)
    parser.add_argument('--species',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--label-source',default='dataset_authors')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    waveform,sr=sf.read(args.audio,dtype='float32',always_2d=True)
    channels=waveform.shape[1]
    waveform=waveform.mean(axis=1)
    divisor=np.gcd(sr,16000)
    if sr!=16000:waveform=resample_poly(waveform,16000//divisor,sr//divisor)
    pcm=np.rint(np.clip(waveform,-1,32767/32768)*32768).astype('int16')
    n=len(pcm)//WINDOW
    if not n:raise ValueError('Recording shorter than one decision window; no padding or repetition is allowed.')
    windows=pcm[:n*WINDOW].reshape(n,WINDOW)
    x=frontend.features(windows,replay_pcm=False)
    peak=np.abs(windows.astype('int32')).max(axis=1)
    clipped=(np.abs(windows.astype('int32'))>=32700).sum(axis=1)
    frame=pd.DataFrame(dict(path=[str(args.audio)]*n,start_s=np.arange(n)*WINDOW/16000,
        step_s=[WINDOW/16000]*n,quality_ok=(peak>=32)&(clipped*100<=WINDOW)))
    outputs,summary=[],[]
    for name,path in [('baseline',ROOT/'results/arduino_species/models/species_fold2.npz'),
                      ('expanded',ROOT/'results/expanded/models/species_fold0.npz')]:
        before=file_hash(path)
        with np.load(path) as z:model={key:z[key].copy() for key in z.files}
        classes=model['classes'].tolist()
        p,g=infer_portable(model,x)
        candidate,confirmed=decide(frame,p,g,model['presence_threshold'],model['confidence_threshold'])
        present=frame.quality_ok.to_numpy()&(g>=model['presence_threshold'])
        state=np.where(~frame.quality_ok,'AUDIO_INVALIDO',np.where(~present,'SEM_EVIDENCIA',
                         np.where(confirmed<0,'INCERTO','IDENTIFICACAO_PROVISORIA')))
        row=frame.copy();row['model']=name;row['state']=state
        row['forced_species']=[classes[i] for i in p.argmax(axis=1)]
        row['identified_species']=[classes[i] if i>=0 else '-' for i in confirmed]
        row['presence_score']=g;row['class_score']=p.max(axis=1)
        outputs.append(row)
        emitted=confirmed>=0
        correct=row.identified_species.eq(args.species).to_numpy()
        summary.append(dict(model=name,checkpoint_sha256=before,model_unchanged=file_hash(path)==before,
            state_counts=row.state.value_counts().to_dict(),
            emitted_species_counts=row.loc[emitted,'identified_species'].value_counts().to_dict(),
            correctly_identified_windows=int(correct.sum()),incorrectly_identified_windows=int((emitted&~correct).sum()),
            forced_correct_windows=int(row.forced_species.eq(args.species).sum()),
            complete_windows=n,reference_species_in_model=args.species in classes))
    pd.concat(outputs,ignore_index=True).to_csv(args.output/'predictions.csv',index=False)
    report=dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),
        source=args.source,reference_species=args.species,label_source=args.label_source,
        original_audio_sha256=file_hash(args.audio),original_sample_rate=sr,original_channels=channels,
        duration_s=len(pcm)/16000,input_pcm_bits=16,complete_windows=n,
        resampling_applied=sr!=16000,volume_normalization=False,noise_filtering=False,
        no_padding_or_repetition=True,physical_board_tested=False,arduino_compiled=False,
        model_results=summary)
    atomic_json(args.output/'analysis.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
