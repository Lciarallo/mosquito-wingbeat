"""Paired recording/PCM analysis of frozen notebook-transfer results; no model tuning."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from notebook_transfer_features import SR, NAMES, LFCC_NAMES, load_reference_functions, relative_lfcc_features
from acoustic_features_v4 import FEATURE_NAMES
from reproduce_notebook_transfer import OUT, ROOT, digest, now, verify_freeze
from research_metrics_v2 import classification_metrics

ANALYSIS=ROOT/'results/notebook_transfer_analysis_2026_10_14_2050'


def weighted_metrics(y,pred,accepted,indices):
    y=y[indices];pred=pred[indices];accepted=accepted[indices]
    labels=np.union1d(np.unique(y),np.unique(pred));a=np.searchsorted(labels,y);b=np.searchsorted(labels,pred)
    cm=np.bincount(a*len(labels)+b,minlength=len(labels)**2).reshape(len(labels),len(labels))
    support=cm.sum(1);emits=cm.sum(0);tp=np.diag(cm);f1=2*tp/np.maximum(support+emits,1)
    correct=y==pred
    return np.array([correct.mean(),np.mean(tp[support>0]/support[support>0]),f1[(support+emits)>0].mean(),accepted.mean(),correct[accepted].mean() if accepted.any() else np.nan])


def paired_bootstrap(frame,a,b,repetitions=2000):
    left=frame[(frame.variant==a[0])&(frame.family==a[1])].set_index('group')
    right=frame[(frame.variant==b[0])&(frame.family==b[1])].set_index('group').reindex(left.index)
    assert np.array_equal(left.true_label,right.true_label)
    y=left.true_label.to_numpy();pa=left.predicted_label.to_numpy();pb=right.predicted_label.to_numpy();aa=left.accepted.to_numpy(bool);ab=right.accepted.to_numpy(bool)
    blocks={k:np.flatnonzero(left.pcm_block.to_numpy()==k) for k in left.pcm_block.unique()}
    strata={}
    for k,ix in blocks.items():strata.setdefault(tuple(sorted(set(left.iloc[ix].source_label))),[]).append(k)
    rng=np.random.default_rng(20261014);draws=[]
    for _ in range(repetitions):
        ids=[]
        for candidates in strata.values():
            for j in rng.integers(0,len(candidates),len(candidates)):ids.extend(blocks[candidates[j]])
        ix=np.array(ids);draws.append(weighted_metrics(y,pb,ab,ix)-weighted_metrics(y,pa,aa,ix))
    draws=np.array(draws);idx=np.arange(len(y));delta=weighted_metrics(y,pb,ab,idx)-weighted_metrics(y,pa,aa,idx)
    names=['accuracy','balanced_accuracy','macro_f1_strict_union','coverage','accepted_accuracy'];result={}
    for i,name in enumerate(names):
        finite=draws[:,i][np.isfinite(draws[:,i])]
        result[name]={'difference':float(delta[i]) if np.isfinite(delta[i]) else None,'ci95':np.quantile(finite,[.025,.975]).tolist() if len(finite) else None}
    return dict(n=len(y),pcm_clusters=len(blocks),unit='recording, clustered by audited PCM block, stratified by observed source-label set',repetitions=repetitions,metrics=result,
                limits='Known recordings/corpus; biological/session independence and acquisition selection are not corrected by bootstrap.')


def main():
    verify_freeze()
    if ANALYSIS.exists():raise FileExistsError('Never overwrite a completed analysis.')
    ANALYSIS.mkdir();table=pd.read_csv(OUT/'recording_predictions.csv')
    calibration=json.loads((OUT/'calibration.json').read_text());selected_rows=[];choices=[]
    # Compute choices solely from the already frozen internal FIT-CV scores.
    for task in ['binary','species']:
        for fold in ['0','1','2']:
            s=max([s for s in calibration if s['task']==task and s['fold']==fold],key=lambda s:(s['cv_macro_f1'],s['variant']=='v4_308',s['family']=='lr'))
            choices.append({k:s[k] for k in ['task','fold','variant','family','cv_macro_f1']})
    (ANALYSIS/'selection_from_frozen_cv.json').write_text(json.dumps(dict(created_utc=now(),choices=choices,outer_metrics_used=False,source_calibration_sha256=digest(OUT/'calibration.json')),indent=2)+'\n')
    for s in choices:
        sub=table[(table.scope=='dryad')&(table.task==s['task'])&(table.fold.astype(str)==s['fold'])&(table.variant==s['variant'])&(table.family==s['family'])].copy()
        sub['variant']='nested_fit_cv_choice';sub['family']='per_fold';selected_rows.append(sub)
    selected=pd.concat(selected_rows);selected.to_csv(ANALYSIS/'nested_selection_predictions.csv',index=False)
    summaries=[]
    for task,g in selected.groupby('task'):
        m=classification_metrics(g.true_label,g.predicted_label,g.accepted);m.update(scope='dryad',task=task,variant='nested_fit_cv_choice',family='per_fold');summaries.append(m)
    # Secondary genus is explicitly post-hoc, without a newly fitted genus threshold.
    genus=[]
    for (scope,variant,family),g in table[table.task=='species'].groupby(['scope','variant','family']):
        y=g.true_label.str.split().str[0];pred=g.predicted_label.str.split().str[0];m=classification_metrics(y,pred)
        m.update(scope=scope,variant=variant,family=family,task='genus_posthoc',separate_genus_rejection_calibrated=False);genus.append(m)
    pd.DataFrame(genus).to_csv(ANALYSIS/'genus_posthoc_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(ANALYSIS/'nested_selection_metrics.csv',index=False)
    intervals={}
    for scope in ['dryad','external17']:
        f=table[(table.scope==scope)&(table.task=='species')]
        intervals[scope+'_lfcc_rf_vs_v4_rf']=paired_bootstrap(f,('v4_308','rf'),('v4_lfcc334','rf'))
    (ANALYSIS/'paired_intervals.json').write_text(json.dumps(intervals,indent=2)+'\n')
    # Stationary-noise and gain stress cases are diagnostics, not extra model tuning.
    t=np.arange(SR)/SR;signal=np.sin(2*np.pi*570*t)+.3*np.sin(2*np.pi*1140*t)+.15*np.sin(2*np.pi*1710*t)
    rng=np.random.default_rng(42);noise=rng.normal(size=SR);noise/=np.sqrt(np.mean(noise**2));rms=np.sqrt(np.mean(signal**2))
    waves={snr:signal+rms*10**(-snr/20)*noise for snr in [25,12]}
    vectors={snr:relative_lfcc_features(w) for snr,w in waves.items()}
    cosine=lambda a,b:float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
    ref=load_reference_functions(cache_filterbank=False)
    raw1=signal.astype(np.float32);raw2=.1*rng.normal(size=SR).astype(np.float32)
    one=ref['extract_features_1s'](ref['pre_processar_audio'](np.r_[raw1,raw2],SR)[:SR],SR)
    changed=ref['extract_features_1s'](ref['pre_processar_audio'](np.r_[raw1,100*raw2],SR)[:SR],SR)
    (ANALYSIS/'frontend_diagnostics.json').write_text(json.dumps(dict(lfcc_noise_25_to_12_db_mean_cosine=cosine(vectors[25][1:13],vectors[12][1:13]),
        lfcc_noise_25_to_12_db_std_cosine=cosine(vectors[25][13:],vectors[12][13:]),
        notebook_whole_recording_normalization_changes_first_window=[k for k in one if one[k]!=changed[k]],
        changing_other_window_only=True,noise_invariance_claimed=False),indent=2)+'\n')
    chosen=json.loads((OUT/'selected_models.json').read_text())['species']
    bundle=joblib.load(OUT/'models'/f"final_species_{chosen['variant']}_{chosen['family']}.joblib")
    if chosen['family']=='rf':
        names=NAMES if chosen['variant']=='notebook61' else list(FEATURE_NAMES)+(LFCC_NAMES if chosen['variant']=='v4_lfcc334' else [])
        importance=pd.DataFrame({'feature':names,'importance':bundle['model'].named_steps['model'].feature_importances_}).sort_values('importance',ascending=False)
        importance.to_csv(ANALYSIS/'fit_only_rf_importance.csv',index=False)
    m=pd.read_csv(OUT/'metrics.csv');fig,ax=plt.subplots(1,2,figsize=(12,4.6));variants=['notebook61','v4_308','v4_lfcc334'];colors=['#687C90','#006B54','#B36640']
    for i,(variant,color) in enumerate(zip(variants,colors)):
        values=[100*float(m[(m.task=='species')&(m.family=='rf')&(m.variant==variant)&(m.scope==scope)].accuracy.iloc[0]) for scope in ['dryad','external17']]
        bars=ax[0].bar(np.arange(2)+(i-1)*.22,values,width=.2,color=color,label=variant)
        for b in bars:ax[0].text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.1f}%',ha='center',fontsize=9)
    ax[0].set_xticks([0,1],['Dryad OOF (569)','D3 pareado (17)']);ax[0].set_ylim(0,80);ax[0].set_ylabel('Acurácia de espécie por gravação (%)');ax[0].legend(fontsize=9);ax[0].set_title('Random Forest: LFCC melhora o interno, não o externo')
    b=m[(m.task=='binary')&(m.variant=='notebook61')&(m.scope=='dryad')].set_index('family').reindex(['lr','tree','rf','knn'])
    ax[1].bar(np.arange(4)-.17,b.accuracy*100,.32,label='Acurácia global',color='#687C90');ax[1].bar(np.arange(4)+.17,b.detection_sensitivity*100,.32,label='Sensibilidade Ae. com corte calibrado',color='#006B54');ax[1].set_xticks(np.arange(4),['Logística','Árvore','RF','kNN']);ax[1].set_ylim(0,105);ax[1].legend(fontsize=9);ax[1].set_title('Acurácia alta pode esconder baixa sensibilidade')
    for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.tight_layout();fig.savefig(ANALYSIS/'comparison.png',dpi=200);plt.close(fig)
    (ANALYSIS/'analysis_complete.json').write_text(json.dumps(dict(created_utc=now(),source_freeze_sha256=digest(OUT/'freeze.json'),outer_data_used_to_tune=False),indent=2)+'\n')
    print(json.dumps({'nested_selection':summaries,'paired_intervals':intervals},indent=2))


if __name__=='__main__':main()
