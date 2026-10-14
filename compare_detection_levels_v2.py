"""Read frozen outputs only; paired recording bootstrap; no fitting or retuning."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from improve_device import ROOT,atomic_json,file_hash
from nested_species_ensemble import now
from research_metrics_v2 import classification_metrics,bootstrap_metrics

SPECIES=ROOT/'results/species_nested_2026_10_13'
OLD=ROOT/'results/species_nested_2026_10_09'
OUT=ROOT/'results/three_levels_2026_10_13'


def _old_frame(scope,new):
    if scope=='dryad':
        old=pd.read_csv(OLD/'recording_predictions.csv').query('model=="calibrated"')
        old=old.set_index('group').reindex(new.group)
        species=old.predicted_species.to_numpy();quality=np.ones(len(old),bool)
        a=old.accepted.to_numpy(bool)
    elif scope=='external':
        old=pd.read_csv(OLD/'external_recording_predictions.csv').query('model=="calibrated"')
        old=old.set_index('group').reindex(new.group)
        species=old.forced_species.to_numpy();quality=old.quality_ok.to_numpy(bool);a=old.accepted.to_numpy(bool)
    else:
        return dict(species=np.where(new.old_quality_ok,new.old_predicted_species,'INVALID'),
                    genus=np.where(new.old_quality_ok,new.old_genus_marginalized,'INVALID'),
                    posthoc=np.where(new.old_quality_ok,new.old_genus_posthoc,'INVALID'),
                    species_accepted=new.old_accepted_species.to_numpy(bool),genus_accepted=new.old_accepted_genus.to_numpy(bool))
    genus=pd.read_csv(OLD/'genus_analysis/predictions.csv.gz')
    genus=genus.query('model=="calibrated" and unit=="recording" and scope==@scope').set_index('group').reindex(new.group)
    if genus.predicted_genus.isna().any() or old.isna().any().any():raise ValueError('Paired historical identity missing.')
    return dict(species=np.where(quality,species,'INVALID'),genus=genus.predicted_genus.to_numpy(),
                posthoc=np.where(quality,[s.split()[0] for s in species],'INVALID'),
                species_accepted=a,genus_accepted=quality)


def main():
    from nested_species_ensemble_v2 import verify_freeze,verify_history
    verify_freeze(SPECIES)
    if OUT.exists():raise FileExistsError('Completed comparisons cannot be overwritten.')
    OUT.mkdir(parents=True)
    rows=[];paired=[];intervals={};per=[];joined=[]
    internal=pd.read_csv(SPECIES/'recording_predictions.csv')
    external=pd.read_csv(SPECIES/'external_recording_predictions.csv')
    for scope,new in [('dryad',internal)]+[(s,f) for s,f in external.groupby('scope')]:
        new=new[new.label.ne('noise')].sort_values('group').reset_index(drop=True)
        if new.empty:continue
        if new.group.duplicated().any():raise ValueError('Bootstrap inputs must be one row per group.')
        old=_old_frame(scope,new);joined.append(new)
        for task,true,oldpred,newpred,olda,newa in [
            ('species',new.label,old['species'],new.predicted_species,old['species_accepted'],new.accepted_species),
            ('genus_marginalized',new.true_genus,old['genus'],new.genus_marginalized,old['genus_accepted'],new.accepted_genus),
            ('genus_posthoc',new.true_genus,old['posthoc'],new.genus_posthoc,np.ones(len(new),bool),np.ones(len(new),bool))]:
            newpred=np.where(new.quality_ok,newpred,'INVALID')
            models={'09/10/2026':np.asarray(oldpred),'13/10/2026 v2':newpred}
            accepts={'09/10/2026':np.asarray(olda,bool),'13/10/2026 v2':np.asarray(newa,bool)}
            ci=bootstrap_metrics(true,models,accepts,clusters=new.collision_group)
            intervals[scope+'|'+task]=ci
            for version,pred in models.items():
                m=classification_metrics(true,pred,accepts[version]);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
                rows.append(dict(scope=scope,task=task,version=version,unit='recording',**m))
                labels=np.union1d(np.unique(true),np.unique(pred))
                cm=confusion_matrix(true,pred,labels=labels)
                pd.DataFrame(cm,index=labels,columns=labels).to_csv(OUT/f'{scope}_{task}_{"v1" if version.startswith("09") else "v2"}_confusion.csv')
                for label in labels:
                    hit=(np.asarray(true)==label)&(pred==label);support=int((np.asarray(true)==label).sum());emitted=int((pred==label).sum())
                    per.append(dict(scope=scope,task=task,version=version,taxon=label,support=support,predicted=emitted,
                                    correct=int(hit.sum()),recall=float(hit.sum()/support) if support else None,
                                    precision=float(hit.sum()/emitted) if emitted else None))
            paired.extend(dict(scope=scope,task=task,group=g,label=t,prediction_v1=a,prediction_v2=b,
                               accepted_v1=bool(c),accepted_v2=bool(d)) for g,t,a,b,c,d in zip(new.group,true,oldpred,newpred,olda,newa))
            # Operational gate is an additional analysis, not a retrofit of old thresholds.
            if task in ['species','genus_marginalized']:
                a=new.accepted_species_gated if task=='species' else new.accepted_genus_gated
                m=classification_metrics(true,newpred,a);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
                rows.append(dict(scope=scope,task=task+'_presence_gate',version='13/10/2026 v2',unit='recording',**m))
    # New presence includes all positive/noise groups, same recording unit.
    for scope,f in [('dryad',internal)]+[(s,g) for s,g in external.groupby('scope')]:
        y=np.where(f.label.eq('noise'),'noise','mosquito');pred=np.where(f.detected,'mosquito','noise')
        pred=np.where(f.quality_ok,pred,'INVALID')
        m=classification_metrics(y,pred);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
        neg=f.label.eq('noise').to_numpy();pos=~neg
        m.update(positive_groups=int(pos.sum()),negative_groups=int(neg.sum()),
                 sensitivity=float(np.mean(pred[pos]=='mosquito')) if pos.any() else None,
                 specificity=float(np.mean(pred[neg]=='noise')) if neg.any() else None,
                 precision_informative=bool(pos.any() and neg.any()))
        rows.append(dict(scope=scope,task='presence',version='13/10/2026 v2',unit='recording',**m))
        intervals[scope+'|presence']=bootstrap_metrics(y,{'13/10/2026 v2':pred},{'13/10/2026 v2':np.ones(len(f),bool)},clusters=f.collision_group)
        labels=np.union1d(y,pred)
        pd.DataFrame(confusion_matrix(y,pred,labels=labels),index=labels,columns=labels).to_csv(OUT/f'{scope}_presence_confusion.csv')
    # Preserve the historical detector's unit/protocol, without calling it paired.
    hist=pd.read_csv(ROOT/'results/arduino/predictions.csv.gz')
    hist=hist[hist.model.eq('Escolha por validação interna') & hist['mode'].eq('Calibração FPR 5%')]
    if len(hist):
        m=classification_metrics(np.where(hist.label.eq('noise'),'noise','mosquito'),np.where(hist.detected,'mosquito','noise'))
        m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
        rows.append(dict(scope='historical_corpus_unpaired',task='presence',version='09/10/2026',unit='window',**m))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(per).to_csv(OUT/'per_taxon.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_recording_predictions.csv',index=False)
    atomic_json(OUT/'bootstrap_intervals.json',intervals)
    # Bootstrap of windows in a single swarm would be pseudoreplication.
    drive=external[external.scope.eq('drive')]
    drive.to_csv(OUT/'private_audio_recording_metrics.csv',index=False)
    windows=pd.read_csv(SPECIES/'external_window_predictions.csv.gz');windows=windows[windows.scope.eq('drive')]
    quality=windows.quality_ok.to_numpy(bool);true=windows.label.to_numpy()
    pd.DataFrame([dict(task='species',unit='window_descriptive_only',**classification_metrics(true,np.where(quality,windows.predicted_species,'INVALID'))),
                  dict(task='genus_marginalized',unit='window_descriptive_only',**classification_metrics(pd.Series(true).str.split().str[0],np.where(quality,windows.genus_marginalized,'INVALID')))]).to_csv(OUT/'private_window_diagnostics.csv',index=False)
    history=verify_history(SPECIES/'historical_snapshot.json');verify_freeze(SPECIES)
    atomic_json(OUT/'audit.json',dict(completed_utc=now(),source_freeze_sha256=file_hash(SPECIES/'freeze.json'),
                historical_files_unchanged=history,no_retuning=True,external_reliability_achieved=False,
                limitations=['Paired files and private swarm are retrospective.','Post-freeze unused files are from a familiar corpus with unknown biological/session ancestry.',
                             'External presence has no absence/noise controls; specificity/precision cannot establish reliability.',
                             'One private recording has no independent recording CI.','Historical genus already marginalizes probabilities and has no rejection threshold.',
                             'Historical presence is window-based and not protocol-matched.']))
    print(pd.DataFrame(rows).query('task in ["species","genus_marginalized","presence"]')[["scope","task","version","n","accuracy","balanced_accuracy","macro_f1_strict_union","coverage","accepted_accuracy"]].to_string(index=False))


if __name__=='__main__':main()
