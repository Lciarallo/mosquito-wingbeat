"""Round-three inference: final test selection frozen before scores are computed."""
import json,hashlib,joblib
import numpy as np,pandas as pd
from train_reproduction import ROOT,OUT,now,sha,source_data
from aedes_binary import targets
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from corrected_validation import audio_windows
from evaluate_reproduction import summarize,verify_freeze,group_bootstrap

HARD=OUT/'hard_negative'


def load():
    verify_freeze();f=json.loads((HARD/'freeze.json').read_text())
    assert sha(ROOT/'train_hard_negative_reproduction.py')==f['source_sha256']
    for p in ['selection','protocol']:assert sha(HARD/(p+'.json'))==f[p+'_sha256']
    for n,h in f['model_hashes'].items():assert sha(HARD/(n+'.joblib'))==h
    selected=json.loads((HARD/'selection.json').read_text())['selected_model']
    return {n:joblib.load(HARD/(n+'.joblib')) for n in f['model_hashes']},selected,f


def pcmsha(w):
    x=np.rint(np.clip(w,-1,32767/32768)*32768).astype('<i2')
    return hashlib.sha256(x.tobytes()).hexdigest()


def main():
    models,selected,freeze=load();f,waves=source_data()
    x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],1)
    tf=f[f.split.isin(['test','challenge'])];rows=[]
    for n,m in models.items():
        p=m['model'].predict_proba(x[tf.index])[:,1]
        for domain in tf.domain.unique():
            mask=tf.domain.eq(domain).to_numpy();s=tf.loc[mask]
            rows.append(dict(model=n,selected=n==selected,domain=domain,**summarize(targets(s),p[mask],s.quality_ok,m['cut'])))
    pd.DataFrame(rows).to_csv(HARD/'historical_metrics_exploratory.csv',index=False)
    private=ROOT/'tmp/private-audio-test/download.audio';w,q,size=audio_windows(private)
    xx=np.concatenate([features(w),extract_batch_bioacoustic_features(w)],1)
    rows=[]
    for n,m in models.items():
        p=m['model'].predict_proba(xx)[:,1];rows.append(dict(model=n,selected=n==selected,**summarize(np.ones(len(q),int),p,q,m['cut'])))
    pd.DataFrame(rows).to_csv(HARD/'private_audio_aggregate_exploratory.csv',index=False)
    plan=json.loads((HARD/'final_selection.json').read_text())
    assert plan['selection_utc']>freeze['frozen_utc'] and plan['model_hashes']==freeze['model_hashes']
    # Check exact decoded/resampled windows against every original split, and
    # against all 19 recordings newly used for hard-negative training.
    known={pcmsha(w) for bucket in waves.values() for w in bucket}
    prior=json.loads((OUT/'fresh_selection.json').read_text())
    for item in prior['files']:
        w,q,size=audio_windows(ROOT/item['path']);known.update(pcmsha(v) for v in w)
    allrows=[];perfile=[];novelty=[];skip=[]
    for item in plan['files']:
        path=ROOT/item['path'];assert sha(path)==item['sha256'];w,q,size=audio_windows(path)
        hashes=[pcmsha(v) for v in w];overlap=sum(h in known for h in hashes)
        novelty.append(dict(path=item['path'],windows=len(w),exact_pcm_windows_seen=overlap,original_recording_group_available=item['source'].startswith('HumBug')))
        if overlap:
            # No substitution and no prediction-based filtering: preserve the
            # planned sample, report and exclude the entire duplicate recording.
            skip.append(item['path']);continue
        xx=np.concatenate([features(w),extract_batch_bioacoustic_features(w)],1);yy=np.full(len(w),int(item['reference_aedes']))
        for n,m in models.items():
            p=m['model'].predict_proba(xx)[:,1]
            perfile.append(dict(model=n,selected=n==selected,path=item['path'],source=item['source'],group=item['group'],
                reference_species=item['reference_species'],**summarize(yy,p,q,m['cut'])))
            allrows.extend(dict(model=n,path=item['path'],source=item['source'],group=item['group'],label=int(y),
                reference_species=item['reference_species'],quality_ok=bool(quality),score=float(score),start_s=i*size/16000)
                for i,(y,quality,score) in enumerate(zip(yy,q,p)))
    pd.DataFrame(novelty).to_csv(HARD/'final_novelty_audit.csv',index=False)
    d=pd.DataFrame(allrows);d.to_csv(HARD/'final_predictions.csv.gz',index=False);pd.DataFrame(perfile).to_csv(HARD/'final_file_metrics.csv',index=False)
    rows=[];bootstrap={}
    for n,s in d.groupby('model'):
        for source in ['all_final',*s.source.unique()]:
            sub=s if source=='all_final' else s[s.source.eq(source)]
            rows.append(dict(model=n,selected=n==selected,source=source,groups=sub.group.nunique(),**summarize(sub.label,sub.score,sub.quality_ok,models[n]['cut'])))
        bootstrap[n]=group_bootstrap(s,models[n]['cut'])
    pd.DataFrame(rows).to_csv(HARD/'final_metrics.csv',index=False)
    (HARD/'final_file_bootstrap_exploratory.json').write_text(json.dumps(bootstrap,indent=2))
    (HARD/'evaluation_audit.json').write_text(json.dumps(dict(completed_utc=now(),final_selection_sha256=sha(HARD/'final_selection.json'),
        excluded_planned_duplicate_recordings=skip,no_substitutions=True,all_evaluation_models_frozen=True,
        final_file_novelty_only_for_cdmsc=True,recording_group_exclusion_for_humbug=True,no_test_label_optimization=True,
        previous_tests_and_private_exploratory=True,arduino_compiled=False),indent=2))
    load();print('FINAL EVALUATION COMPLETE',selected,flush=True)


if __name__=='__main__':main()
