"""Frozen classical models versus frozen CNN, on fresh clips and private audio."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib,torch,soundfile as sf
from scipy.signal import resample_poly
from scipy.special import expit
from aedes_classical import O,ROOT,audio_features,calibrated
from aedes_binary import load as load_cnn,predict as cnn_predict,decisions,CHECKPOINT
from train_aedes_binary import summary

def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';cnn,cnn_saved=load_cnn(device=device)
    audit=json.loads((O/'training_audit.json').read_text());selection=json.loads((O/'fresh_external/selection.json').read_text())
    assert selection['selection_utc']>audit['frozen_utc']
    models={r['model']:joblib.load(O/f"models/{r['model']}.joblib") for r in audit['models']}
    before={r['model']:hashlib.sha256((O/f"models/{r['model']}.joblib").read_bytes()).hexdigest() for r in audit['models']}
    assert before==selection['models_sha256'];cnn_hash=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    def evaluate(path,reference,group,domain):
        x,quality,size,sr=audio_features(path);w,original_sr=sf.read(path,dtype='float32',always_2d=True);w=w.mean(axis=1)
        if original_sr!=16000:
            d=np.gcd(original_sr,16000);w=resample_poly(w,16000//d,original_sr//d)
        wave=w[:len(x)*size].reshape(len(x),size)
        scores={name:(calibrated(saved,x),saved) for name,saved in models.items()}
        scores['cnn_binary']=(expit(cnn_saved['slope']*cnn_predict(cnn,wave,device)+cnn_saved['bias']),cnn_saved)
        rows=[];frames=[]
        for name,(p,saved) in scores.items():
            frame=pd.DataFrame(dict(label=[reference]*len(p),group=[group]*len(p),domain=[domain]*len(p),start_s=np.arange(len(p))*size/16000,quality_ok=quality,aedes_score=p))
            frame['model']=name;frame['state']=decisions(p,quality,saved['low'],saved['high']);r,_=summary(frame,p,saved['low'],saved['high'])
            rows.append(dict(model=name,reference_species=reference,source_group=group,**r));frames.append(frame)
        return rows,frames
    rows=[];frames=[]
    for item in selection['files']:
        path=ROOT/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        r,f=evaluate(path,item['reference_species'],item['group'],'fresh_humbug');rows.extend(r);frames.extend(f)
    pd.DataFrame(rows).to_csv(O/'fresh_external/per_file.csv',index=False);full=pd.concat(frames,ignore_index=True);full.to_csv(O/'fresh_external/predictions.csv',index=False)
    agg=[]
    for name,f in full.groupby('model'):
        saved=cnn_saved if name=='cnn_binary' else models[name];r,_=summary(f,f.aedes_score.to_numpy(),saved['low'],saved['high']);agg.append(dict(model=name,**r))
    pd.DataFrame(agg).to_csv(O/'fresh_external/metrics.csv',index=False)
    private=ROOT/'tmp/private-audio-test/download.audio';r,f=evaluate(private,'Aedes aegypti','private_user_audio','private_swarm')
    dest=ROOT/'tmp/private-audio-test/aedes_classical';dest.mkdir(exist_ok=True)
    summaries=[]
    for frame in f:
        name=frame.model.iloc[0];frame.to_csv(dest/f'{name}_predictions.csv',index=False)
        summaries.append(dict(model=name,windows=len(frame),states=frame.state.value_counts().to_dict(),reference_source='user, swarm species not independently confirmed; no event annotations',training_used=False))
    (O/'private_audio_summary.json').write_text(json.dumps(summaries,indent=2))
    assert before=={n:hashlib.sha256((O/f'models/{n}.joblib').read_bytes()).hexdigest() for n in models}
    assert cnn_hash==hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    (O/'evaluation_audit.json').write_text(json.dumps(dict(models_unchanged=True,fresh_selection_after_all_models_frozen=True,fresh_recordings=len(selection['files']),new_domain=False,private_audio_used_for_training=False,arduino_compiled=False),indent=2))
    print(pd.DataFrame(agg).to_string(index=False));print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
