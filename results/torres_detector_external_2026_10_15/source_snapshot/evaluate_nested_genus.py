"""Post-hoc genus aggregation of frozen species probabilities, without tuning."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from nested_species_ensemble import ROOT, NAMES, audio_probabilities, now, file_hash
from evaluate_nested_species import verify_freeze


def aggregate(p,classes):
    genus=np.array([c.split()[0] for c in classes])
    taxa=np.unique(genus)
    return taxa,np.column_stack([p[:,genus==g].sum(1) for g in taxa])


def main():
    out=ROOT/'results/species_nested_2026_10_09'
    dest=out/'genus_analysis'
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Genus analysis already exists.')
    freeze=verify_freeze(out)
    dest.mkdir(exist_ok=True)
    a=np.load(out/'oof_probabilities.npz')
    classes=json.loads((out/'protocol.json').read_text())['classes']
    frame=pd.read_csv(ROOT/'results/segments.csv').iloc[a['row_indices']].copy()
    metrics=[];predictions=[]

    def record(scope,unit,name,truth,p,taxa,groups,quality):
        predicted=np.where(quality,taxa[p.argmax(1)],'INVALID')
        correct=predicted==truth
        observed=np.unique(truth)
        metrics.append(dict(scope=scope,unit=unit,model=name,n=len(truth),groups=len(np.unique(groups)),
            correct=int(correct.sum()),accuracy=float(accuracy_score(truth,predicted)),
            balanced_accuracy=float(balanced_accuracy_score(truth,predicted)),
            macro_f1=float(f1_score(truth,predicted,labels=observed,average='macro',zero_division=0)),
            forced_choice=True,genus_rejection_calibrated=False))
        cm=confusion_matrix(truth,predicted,labels=taxa)
        pd.DataFrame(cm,index=taxa,columns=taxa).to_csv(dest/f'{scope}_{unit}_{name}_confusion.csv')
        predictions.extend(dict(scope=scope,unit=unit,model=name,group=str(g),true_genus=str(t),
            predicted_genus=str(pred),score=float(score),quality_ok=bool(q))
            for g,t,pred,score,q in zip(groups,truth,predicted,p.max(1),quality))

    for name in NAMES:
        if a[name].shape[1]!=len(classes):raise ValueError('Class dimensions do not match protocol.')
        taxa,p=aggregate(a[name],classes)
        truth=frame.label.str.split().str[0].to_numpy()
        record('dryad','window',name,truth,p,taxa,frame.group,np.ones(len(p),bool))
        rec=pd.DataFrame(p,index=frame.group).groupby(level=0).mean()
        truth_rec=frame.groupby('group').label.first().str.split().str[0].reindex(rec.index).to_numpy()
        record('dryad','recording',name,truth_rec,rec.to_numpy(),taxa,rec.index,np.ones(len(rec),bool))

    bundle=joblib.load(out/'models/final_bundle.joblib')
    if bundle['classes']!=classes:raise ValueError('Final class order differs from OOF protocol.')
    plan=json.loads((out/'fresh_selection.json').read_text())
    audit=pd.read_csv(out/'novelty_audit.csv')
    excluded=set(audit[audit.excluded].path)
    external={name:[] for name in NAMES}
    metadata=[]
    for item in plan['files']:
        if item['path'] in excluded:continue
        if file_hash(ROOT/item['path'])!=item['sha256']:raise ValueError('Audio changed.')
        scores,quality,info=audio_probabilities(ROOT/item['path'],bundle)
        for name,p in scores.items():
            taxa,q=aggregate(p,bundle['classes'])
            external[name].append((q,quality))
        metadata.append(item)
    for name,parts in external.items():
        p=np.concatenate([z[0] for z in parts]);quality=np.concatenate([z[1] for z in parts])
        truth=np.concatenate([np.repeat(item['reference_species'].split()[0],len(part[0])) for item,part in zip(metadata,parts)])
        groups=np.concatenate([np.repeat(item['group'],len(part[0])) for item,part in zip(metadata,parts)])
        record('external','window',name,truth,p,taxa,groups,quality)
        rec=np.stack([z[0].mean(0) for z in parts]);quality_rec=np.array([z[1].all() for z in parts])
        record('external','recording',name,np.array([item['reference_species'].split()[0] for item in metadata]),
            rec,taxa,np.array([item['group'] for item in metadata]),quality_rec)
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(dest/'predictions.csv.gz',index=False)
    verify_freeze(out)
    (dest/'audit.json').write_text(json.dumps(dict(evaluated_utc=now(),
        inference_code_sha256=file_hash(Path(__file__)),models_thresholds_unchanged=True,
        species_probabilities_summed_by_genus=True,genus_specific_training=False,
        genus_rejection_calibrated=False,analysis_post_hoc=True,
        scope='Known Dryad corpus and the same 17 external clips; not a new independent acquisition.',
        frozen_utc=freeze['frozen_utc']),indent=2)+'\n')
    print(pd.DataFrame(metrics).query("model=='calibrated'").to_string(index=False))
    print(pd.read_csv(dest/'external_recording_calibrated_confusion.csv').to_string(index=False))


if __name__=='__main__':main()
