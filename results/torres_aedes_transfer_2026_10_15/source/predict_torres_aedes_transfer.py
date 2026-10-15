"""Inference for the separate exploratory Torres-augmented Aedes detector."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.signal import resample_poly
from ternary_protocol import read_audio,CLASS_NAMES
from train_torres_aedes_transfer import bank,describe

def predict(run,audio,variant='with_torres'):
 bundle=joblib.load(Path(run)/f'{variant}_model.joblib');wave=read_audio(Path(audio),8000);n=len(wave)//7936
 if n<1:raise ValueError('At least0.992s of real audio required; no padding/repetition.')
 windows=wave[:n*7936].reshape(n,7936);rms=np.sqrt(np.mean(windows**2,axis=1));ix=np.argsort(-rms,kind='stable')[:3]
 front,keep,edges=bank(bundle['frontend']);assert np.array_equal(keep,bundle['kept_bands'])
 with torch.inference_mode():db=front(torch.from_numpy(resample_poly(windows[ix],6,1,axis=1).copy())).numpy()
 x=describe(db,keep).mean(0,keepdims=True);p=bundle['model'].predict_proba(x)[0]
 return dict(variant=variant,audio=str(audio),seconds=len(wave)/8000,available_windows=n,selected_window_indices=ix.tolist(),class_order=CLASS_NAMES,probabilities=p.tolist(),decision=CLASS_NAMES[int(p.argmax())],scope='Illustrative WAV inference; relationship to author tensors unknown, not a guaranteed independent holdout')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audio',type=Path);parser.add_argument('--run',type=Path,default=Path('results/torres_aedes_transfer_2026_10_15'));parser.add_argument('--variant',choices=['baseline','with_torres'],default='with_torres');a=parser.parse_args();torch.set_num_threads(4);print(json.dumps(predict(a.run,a.audio,a.variant),ensure_ascii=False,indent=2))
