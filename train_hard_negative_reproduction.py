"""Round three: explicit hard-negative feedback; round-one controls become TRAIN.

Requires an entirely new test after the new model freeze. Earlier test outcomes
are development diagnostics and must not be called blind for this round.
"""
import json,joblib
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from aedes_binary import ROOT,targets,weights
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from corrected_validation import audio_windows,assert_disjoint
from train_reproduction import OUT,source_data,choose_binary_cut,now,sha
from evaluate_reproduction import verify_freeze


def main():
    verify_freeze();dest=OUT/'hard_negative';dest.mkdir(exist_ok=True)
    if (dest/'freeze.json').exists():raise FileExistsError('Never refit a frozen model.')
    f,_=source_data();x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],1)
    plan=json.loads((OUT/'fresh_selection.json').read_text());rows=[];extra=[]
    for item in plan['files']:
        assert not item['reference_aedes'];assert sha(ROOT/item['path'])==item['sha256']
        w,q,size=audio_windows(ROOT/item['path']);extra.append(np.concatenate([features(w),extract_batch_bioacoustic_features(w)],1))
        rows.extend(dict(label=item['reference_species'],domain='humbug',group='humbug:'+item['group'],
              split='train',quality_ok=bool(quality),content_sha256=item['sha256']) for quality in q)
    e=pd.DataFrame(rows);combined=pd.concat([f,e],ignore_index=True);assert_disjoint(combined)
    extra=np.concatenate(extra);ti=np.flatnonzero(f.split.eq('train')&f.quality_ok&(targets(f)>=0))
    et=np.flatnonzero(e.quality_ok);train=pd.concat([f.iloc[ti],e.iloc[et]],ignore_index=True)
    xx=np.concatenate([x[ti],extra[et]],0);y=targets(train);w=weights(train)
    v=f[f.split.eq('validation')&f.quality_ok&(targets(f)>=0)];c=f[f.split.eq('calibration')&f.quality_ok&(targets(f)>=0)]
    protocol=dict(created_utc=now(),strategy='Hard-negative retraining of ExtraTrees; new negative species and real background recordings.',
        original_training_windows=len(ti),additional_training_windows=len(et),additional_training_recording_groups=e.group.nunique(),
        additional_source_selection_sha256=sha(OUT/'fresh_selection.json'),
        original_group_partitions_still_disjoint=True,unknowns_excluded=True,private_in_training=False,
        configurations=[dict(max_depth=12,min_samples_leaf=3),dict(max_depth=None,min_samples_leaf=3)],
        selection='Weighted validation BA at probability .5 + 0.001 validation AUC.',
        threshold_selection='Weighted calibration Youden threshold, requiring calibration sensitivity and specificity >=.6 if feasible.',
        earlier_evaluations_status='Exploratory development diagnostics; 19 round-one negatives are now training.',arduino_compiled=False)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train[['label','domain','group','split','quality_ok','content_sha256']].to_csv(dest/'training_manifest.csv.gz',index=False)
    records=[]
    for depth in [12,None]:
        name='extra_trees_'+('regularized' if depth is not None else 'deep')
        m=ExtraTreesClassifier(n_estimators=500,max_depth=depth,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=20261009)
        m.fit(xx,y,sample_weight=w);p=m.predict_proba(x[v.index])[:,1];pc=m.predict_proba(x[c.index])[:,1]
        cut=choose_binary_cut(targets(c),pc,weights(c),partition='calibration')
        path=dest/(name+'.joblib');joblib.dump(dict(model=m,cut=cut,name=name),path)
        record=dict(name=name,validation_weighted_ba=float(np.average((p>=.5)==targets(v),weights=weights(v))),
            validation_weighted_auc=float(roc_auc_score(targets(v),p,sample_weight=weights(v))),cut=cut,sha256=sha(path))
        records.append(record);print(name,json.dumps(record),flush=True)
    selected=max(records,key=lambda r:r['validation_weighted_ba']+.001*r['validation_weighted_auc'])
    (dest/'selection.json').write_text(json.dumps(dict(selected_model=selected['name'],selected_before_new_test=True,records=records),indent=2))
    (dest/'freeze.json').write_text(json.dumps(dict(frozen_utc=now(),model_hashes={r['name']:r['sha256'] for r in records},
        source_sha256=sha(ROOT/'train_hard_negative_reproduction.py'),protocol_sha256=sha(dest/'protocol.json'),selection_sha256=sha(dest/'selection.json')),indent=2))
    verify_freeze();print('HARD NEGATIVE FROZEN',selected['name'],flush=True)


if __name__=='__main__':main()
