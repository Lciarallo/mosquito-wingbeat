"""Evaluate fixed post-freeze recordings; never fit or choose models here."""
import json
import numpy as np
import pandas as pd
from corrected_validation import ROOT,OUT,audio_windows,load_frozen,complete_predictions,metrics,sha,now


def main():
    pred,bundle,freeze=load_frozen();selection=json.loads((OUT/'fresh_selection.json').read_text())
    assert selection['selection_utc']>freeze['frozen_utc'];assert selection['model_hashes']==freeze['models_sha256']
    rows=[];all_rows=[]
    for item in selection['files']:
        path=ROOT/item['path'];assert sha(path)==item['sha256'];w,q,size=audio_windows(path)
        p,x=pred.audio(w);p=complete_predictions(p,x,bundle);y=np.full(len(w),int(item['reference_aedes']))
        for name,scores in p.items():
            l,h=bundle['thresholds'][name];r=metrics(y,scores,q,l,h,h if name=='swarm_corrected' else .5)
            rows.append(dict(model=name,path=item['path'],reference_species=item['reference_species'],group=item['group'],**r))
            for i,score in enumerate(scores):all_rows.append(dict(model=name,path=item['path'],group=item['group'],reference_species=item['reference_species'],label=int(y[i]),quality_ok=bool(q[i]),score=float(score),start_s=i*size/16000))
    pd.DataFrame(rows).to_csv(OUT/'fresh_file_metrics.csv',index=False)
    frame=pd.DataFrame(all_rows);frame.to_csv(OUT/'fresh_predictions.csv.gz',index=False);aggregate=[]
    for (name,source),sub in frame.assign(source=np.where(frame.reference_species.eq('environmental_noise'),'fresh_noise','fresh_other_mosquito')).groupby(['model','source']):
        l,h=bundle['thresholds'][name];aggregate.append(dict(model=name,source=source,**metrics(sub.label,sub.score,sub.quality_ok,l,h,h if name=='swarm_corrected' else .5)))
    pd.DataFrame(aggregate).to_csv(OUT/'fresh_metrics.csv',index=False)
    (OUT/'fresh_evaluation_audit.json').write_text(json.dumps(dict(completed_utc=now(),files=len(selection['files']),
        original_group_exclusion=True,byte_duplicate_exclusion=True,positive_groups=selection['positive_groups_available'],
        inference_only=True,limitations=selection['positive_limitation']),indent=2))
    load_frozen();print('FRESH EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()
