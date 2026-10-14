"""Evaluate frozen binary model on post-freeze selections; never alter weights."""
import json,hashlib
import numpy as np,pandas as pd,torch
from aedes_binary import ROOT,CHECKPOINT,load,evaluate_audio
from train_aedes_binary import summary

def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';model,saved=load(device=device)
    sha=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest();out=ROOT/'results/aedes_binary/fresh_external'
    selection=json.loads((out/'selection.json').read_text());assert sha==selection['checkpoint_sha256'];assert selection['selection_utc']>selection['frozen_utc']
    rows=[];all_frames=[]
    for item in selection['files']:
        path=ROOT/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        f,sr=evaluate_audio(path,model,saved,device);f['label']=item['reference_species'];f['group']=item['group'];f['id']=item['id'];f['domain']='fresh_humbug'
        row,_=summary(f,f.aedes_score.to_numpy(),saved['low'],saved['high'])
        rows.append(dict(id=item['id'],reference_species=item['reference_species'],**row));all_frames.append(f)
    all_frames=pd.concat(all_frames,ignore_index=True);all_frames.to_csv(out/'predictions.csv',index=False);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    result,_=summary(all_frames,all_frames.aedes_score.to_numpy(),saved['low'],saved['high'])
    result.update(recordings=len(rows),positive_recordings=sum(x['reference_aedes'] for x in selection['files']),selection_utc=selection['selection_utc'],frozen_utc=selection['frozen_utc'],model_unchanged=sha==hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),new_domain=False,shifted_waveform_overlap_audit_complete=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
