"""Frozen inference with confusion matrices, precision/recall and group CIs."""
import json,hashlib
import joblib,numpy as np,pandas as pd,torch
from scipy.signal import resample_poly
from scipy.special import expit,logit
from train_reproduction import ROOT,now,sha
from fetch_reliability_data import OUT
from prepare_reliability_data import CACHE
from reliability_models import JointDetector
from reliability_metrics import binary_metrics,precision_at_prevalence
from aedes_binary import targets
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from corrected_validation import audio_windows,assert_disjoint


def verify():
    f=json.loads((OUT/'freeze.json').read_text())
    for p,h in {**f['models_sha256'],**f['source_sha256']}.items():assert sha(ROOT/p)==h,p
    for p in ['selection','protocol']:assert sha(OUT/(p+'.json'))==f[p+'_sha256']
    for p,h in json.loads((OUT/'original_weights_sha256.json').read_text()).items():assert sha(ROOT/p)==h
    return f


def infer_stages(model,name,frame,waves,x,device,batch=64):
    """The frozen rule, exposing presence and conditional scores for inspection.

    Raw CNN long-input stage scores are averages over three physical crops;
    its joint score is the mean of crop products, not product of head means.
    """
    model.eval();out=np.empty(len(frame));presence=np.empty(len(frame));species=np.empty(len(frame))
    with torch.no_grad():
        for bucket in ['long','short']:
            ix=np.flatnonzero(frame.bucket.eq(bucket));wi=frame.iloc[ix].wave_index.astype(int).to_numpy()
            for start in range(0,len(ix),batch):
                index=ix[start:start+batch];w=torch.tensor(np.array(waves[bucket][wi[start:start+batch]]),device=device)
                feat=torch.tensor(np.asarray(x[index],dtype='float32'),device=device)
                crop=name=='raw_expanded' and w.shape[1]>2400
                inputs=torch.cat([w[:,off:off+2400] for off in [0,2400,4800]]) if crop else w
                joint,zp,zs=model(inputs,feat.repeat(3,1) if crop else feat)
                a,b=zp.sigmoid().cpu().numpy(),zs.sigmoid().cpu().numpy()
                if crop:
                    z=logit(np.clip(joint.sigmoid().cpu().numpy().reshape(3,len(w)).mean(0),1e-6,1-1e-6));a=a.reshape(3,len(w)).mean(0);b=b.reshape(3,len(w)).mean(0)
                else:z=joint.cpu().numpy()
                out[index]=z;presence[index]=a;species[index]=b
    return out,presence,species


class Predictor:
    def __init__(self):
        self.freeze=verify();torch.set_num_threads(4);self.device='cuda' if torch.cuda.is_available() else 'cpu';self.models={};self.cuts={}
        self.selected=json.loads((OUT/'selection.json').read_text())['selected_model']
        for path in self.freeze['models_sha256']:
            p=ROOT/path
            if p.suffix=='.pt':
                s=torch.load(p,map_location='cpu',weights_only=True)
                m=JointDetector(s['name'],np.zeros(138,dtype='float32'),np.ones(138,dtype='float32')).to(self.device)
                m.load_state_dict(s['state_dict']);m.eval()
            else:s=joblib.load(p);m=None
            self.models[s['name']]=(m,s);self.cuts[s['name']]=s['cut']
        # Paired comparison with the model chosen BEFORE round-three test.
        # It cannot compete in this round's selection, and is never refitted.
        from evaluate_hard_negative_reproduction import load as load_previous
        previous,old_selected,old_freeze=load_previous();saved=previous[old_selected]
        self.comparison_name='previous_round3_'+old_selected
        self.comparison_sha256=old_freeze['model_hashes'][old_selected]
        self.models[self.comparison_name]=(None,dict(legacy_model=saved['model'],cut=saved['cut']))
        self.cuts[self.comparison_name]=saved['cut']
        from aedes_classical import FEATURE_NAMES
        from bioacoustic_invariance import BIOACOUSTIC_FEATURE_NAMES
        names=FEATURE_NAMES+BIOACOUSTIC_FEATURE_NAMES;importance=[]
        for name,(_,saved) in self.models.items():
            if name.startswith('trees'):
                for stage in ['presence_model','species_model']:
                    importance.extend(dict(model=name,stage=stage,feature=feature,impurity_importance=float(value)) for feature,value in zip(names,saved[stage].feature_importances_))
        pd.DataFrame(importance).to_csv(OUT/'classical_feature_importance_exploratory.csv',index=False)

    def frame(self,f,w,x):
        output={};self.stage_scores={}
        for name,(m,s) in self.models.items():
            if 'legacy_model' in s:
                output[name]=s['legacy_model'].predict_proba(x)[:,1];continue
            if m is not None:
                z,presence,species=infer_stages(m,name,f,w,x,self.device);self.stage_scores[name]=(presence,species)
            else:
                presence=s['presence_model'].predict_proba(x)[:,1];species=s['species_model'].predict_proba(x)[:,1]
                z=logit(np.clip(presence*species,1e-6,1-1e-6));self.stage_scores[name]=(presence,species)
            output[name]=expit(s['slope']*z+s['bias'])
        return output

    def audio(self,path):
        a,q,size=audio_windows(path);bucket='long' if size==15872 else 'short'
        f=pd.DataFrame(dict(bucket=[bucket]*len(a),wave_index=np.arange(len(a))))
        w={bucket:resample_poly(a,1,2,axis=1).astype('float32'),'short' if bucket=='long' else 'long':np.empty((0,2400 if bucket=='long' else 7936),dtype='float32')}
        x=np.concatenate([features(a),extract_batch_bioacoustic_features(a)],1).astype('float32')
        return self.frame(f,w,x),q,size,w[bucket]


def pcmsha(w):
    return hashlib.sha256(np.rint(np.clip(w,-1,32767/32768)*32768).astype('<i2').tobytes()).hexdigest()


def bootstrap_values(frame,cut,replicates=2000):
    y=frame.label.to_numpy();pred=np.where(frame.quality_ok,frame.score>=cut,2)
    count=pd.DataFrame(dict(group=frame.group,source=frame.source,label=y,tp=(y==1)&(pred==1),fp=(y==0)&(pred==1),tn=(y==0)&(pred==0),positive=y==1,negative=y==0,fn=(y==1)&(pred==0)))
    g=count.groupby(['source','group'],sort=True)[['tp','fp','tn','positive','negative','fn']].sum()
    rng=np.random.default_rng(20261013);total=np.zeros((replicates,6))
    for source,part in g.groupby(level=0):
        a=part.to_numpy();ix=rng.integers(len(a),size=(replicates,len(a)));total+=a[ix].sum(1)
    tp,fp,tn,pos,neg,fn=total.T
    precision=tp/np.maximum(tp+fp,1);recall=tp/np.maximum(pos,1);specificity=tn/np.maximum(neg,1)
    positive_f1=2*tp/np.maximum(tp+fp+pos,1);negative_f1=2*tn/np.maximum(tn+fn+neg,1)
    values=dict(aedes_precision=precision,aedes_recall=recall,aedes_f1=positive_f1,specificity=specificity,macro_f1=(positive_f1+negative_f1)/2,balanced_accuracy=(recall+specificity)/2)
    return values,len(g)


def intervals(frame,cut,replicates=2000):
    """Stratified recording-group resampling; CD files are not known individuals."""
    values,groups=bootstrap_values(frame,cut,replicates)
    return dict(method='2000 bootstrap replicates, original recording groups stratified by source; CDMSC file identity only, not independent individuals.',groups=groups,metrics={k:dict(lower95=float(np.quantile(v,.025)),upper95=float(np.quantile(v,.975))) for k,v in values.items()})


def metric_row(model,selected,scope,sub,cut):
    return dict(model=model,selected=model==selected,scope=scope,groups=sub.group.nunique(),**binary_metrics(sub.label,sub.score,sub.quality_ok,cut))


def main():
    p=Predictor();f=pd.read_csv(OUT/'segments.csv',low_memory=False);assert_disjoint(f)
    w={b:np.load(CACHE/(b+'_8k.npy'),mmap_mode='r') for b in ['long','short']};x=np.load(CACHE/'features.npy',mmap_mode='r')
    tf=f[f.split.isin(['test','challenge'])];scores=p.frame(tf,w,x[tf.index]);rows=[]
    for name,score in scores.items():
        sub=tf[['group','domain','quality_ok']].copy();sub['label']=targets(tf);sub['score']=score
        for domain,a in sub.groupby('domain'):rows.append(metric_row(name,p.selected,domain,a,p.cuts[name]))
    pd.DataFrame(rows).to_csv(OUT/'historical_metrics_exploratory.csv',index=False)
    score,q,_,_=p.audio(ROOT/'tmp/private-audio-test/download.audio');rows=[]
    for name,v in score.items():rows.append(dict(model=name,selected=name==p.selected,**binary_metrics(np.ones(len(q),int),v,q,p.cuts[name])))
    pd.DataFrame(rows).to_csv(OUT/'private_audio_aggregate_exploratory.csv',index=False)
    plan=json.loads((OUT/'fresh_selection.json').read_text());assert plan['selection_utc']>p.freeze['frozen_utc'] and plan['models_sha256']==p.freeze['models_sha256']
    assert sha(OUT/'fresh_protocol.json')==plan['fresh_protocol_sha256']
    fresh_protocol=json.loads((OUT/'fresh_protocol.json').read_text());assert fresh_protocol['created_utc']<p.freeze['frozen_utc']
    known={pcmsha(a) for bucket in w.values() for a in bucket};groups=set(f.group);known_files=set(f.content_sha256.dropna())
    previous=json.loads((ROOT/'results/reproduction_2026_10_01/hard_negative/final_selection.json').read_text())
    for item in previous['files']:
        if item['reference_aedes']:
            a,q0,size=audio_windows(ROOT/item['path']);known.update(pcmsha(v) for v in resample_poly(a,1,2,axis=1));known_files.add(item['sha256'])
    allrows=[];audits=[];excluded=[];seen_fresh=set()
    for i,item in enumerate(plan['files']):
        path=ROOT/item['path'];assert sha(path)==item['sha256'];assert item['group'] not in groups
        a,q0,size=audio_windows(path);a8=resample_poly(a,1,2,axis=1);overlap=sum(pcmsha(v) in known for v in a8)
        duplicate=item['sha256'] in known_files or item['sha256'] in seen_fresh or overlap>0
        seen_fresh.add(item['sha256'])
        audits.append(dict(path=item['path'],windows=len(a),exact_pcm_windows_seen=overlap,duplicate_file=duplicate,original_recording_identity_available=item['source']!='CDMSC2026_D3'))
        if duplicate:excluded.append(item['path']);continue
        scores,q,size,_=p.audio(path)
        for name,v in scores.items():
            stages=p.stage_scores.get(name,(np.full(len(v),np.nan),np.full(len(v),np.nan)))
            allrows.extend(dict(model=name,path=item['path'],source=item['source'],group=item['group'],reference_species=item['reference_species'],label=int(item['reference_aedes']),quality_ok=bool(quality),score=float(s),presence_stage_score=float(stages[0][j]),conditional_species_score=float(stages[1][j]),start_s=j*size/16000) for j,(s,quality) in enumerate(zip(v,q)))
        if (i+1)%10==0:print('Fresh inference',i+1,'of',len(plan['files']),flush=True)
    d=pd.DataFrame(allrows);d.to_csv(OUT/'fresh_predictions.csv.gz',index=False);pd.DataFrame(audits).to_csv(OUT/'fresh_novelty_audit.csv',index=False)
    d.groupby(['model','reference_species'])[['presence_stage_score','conditional_species_score','score']].median().to_csv(OUT/'stage_explanations.csv')
    rows=[];file_rows=[];ci={}
    for name,s in d.groupby('model'):
        scopes={'all_fresh':s,'mosquito_only':s[s.reference_species.ne('noise')],**{source:sub for source,sub in s.groupby('source')}}
        for scope,sub in scopes.items():rows.append(metric_row(name,p.selected,scope,sub,p.cuts[name]))
        for species,sub in s.groupby('reference_species'):rows.append(metric_row(name,p.selected,'taxon:'+species,sub,p.cuts[name]))
        for path,sub in s.groupby('path'):file_rows.append(dict(path=path,reference_species=sub.reference_species.iloc[0],**metric_row(name,p.selected,sub.source.iloc[0],sub,p.cuts[name])))
        ci[name]={scope:intervals(sub,p.cuts[name]) for scope,sub in scopes.items() if scope in ['all_fresh','mosquito_only','CDMSC2026_D3']}
    pd.DataFrame(rows).to_csv(OUT/'fresh_metrics.csv',index=False);pd.DataFrame(file_rows).to_csv(OUT/'fresh_file_metrics.csv',index=False)
    (OUT/'fresh_group_bootstrap.json').write_text(json.dumps(ci,indent=2))
    new=d[d.model.eq(p.selected)];old=d[d.model.eq(p.comparison_name)]
    assert list(zip(new.path,new.start_s))==list(zip(old.path,old.start_s))
    delta={}
    for scope in ['all_fresh','mosquito_only','CDMSC2026_D3']:
        a=new if scope=='all_fresh' else new[new.reference_species.ne('noise')] if scope=='mosquito_only' else new[new.source.eq(scope)]
        b=old if scope=='all_fresh' else old[old.reference_species.ne('noise')] if scope=='mosquito_only' else old[old.source.eq(scope)]
        nv,_=bootstrap_values(a,p.cuts[p.selected]);ov,_=bootstrap_values(b,p.cuts[p.comparison_name])
        nm=binary_metrics(a.label,a.score,a.quality_ok,p.cuts[p.selected]);om=binary_metrics(b.label,b.score,b.quality_ok,p.cuts[p.comparison_name])
        delta[scope]={k:dict(change=float(nm[k]-om[k]),lower95=float(np.quantile(v-ov[k],.025)),upper95=float(np.quantile(v-ov[k],.975))) for k,v in nv.items()}
    (OUT/'paired_improvement_exploratory.json').write_text(json.dumps(dict(new_model=p.selected,previous_model=p.comparison_name,method='Paired identical recording-group bootstrap draws, stratified by source; exploratory, individual identities unknown.',differences=delta),indent=2))
    selected=[r for r in rows if r['selected']];overall=next(r for r in selected if r['scope']=='all_fresh');mosq=next(r for r in selected if r['scope']=='mosquito_only')
    failures=[]
    same_domain=next(r for r in selected if r['scope']=='CDMSC2026_D3')
    for scope,row in [('all_fresh',overall),('mosquito_only',mosq),('CDMSC2026_D3',same_domain)]:
        for m in ['aedes_precision','aedes_recall','macro_f1','specificity']:
            if row[m] is None or row[m]<.6:failures.append(f'{scope}: {m} below .60')
            if ci[p.selected][scope]['metrics'][m]['lower95']<.6:failures.append(f'{scope}: {m} lower95 below .60')
    for r in selected:
        if r['scope'].startswith('taxon:') and r['scope']!='taxon:noise':
            m='aedes_recall' if r['scope']=='taxon:Aedes aegypti' else 'specificity'
            if r[m] is None or r[m]<.6:failures.append(r['scope']+': '+m+' below .60')
    decision=dict(completed_utc=now(),selected_model=p.selected,offline_minimum_criteria_passed=not failures,
        failure_reasons=failures,reliable_for_arduino=False,target_sensor_negative_recordings_available=False,
        classification='Insufficient evidence for deployment; target-sensor independent positive and negative recordings still required.',
        source_confound='HumBug has fresh negatives only; same-domain CDMSC includes positives AND negatives, but original sessions remain unknown.',positive_individual_mapping_unknown=True,private_audio_not_blind=True,
        fresh_files=len(plan['files'])-len(excluded),excluded_duplicate_recordings=excluded,no_prediction_filtering=True,no_substitutions=True,
        prevalence_scenarios_not_observed_precision={str(prevalence):precision_at_prevalence(mosq['aedes_recall'],mosq['emitted_false_positive_rate'],prevalence) for prevalence in [.01,.05,.2,.5]},
        evaluation_source_sha256=sha(ROOT/'evaluate_reliability.py'),fresh_protocol_sha256=sha(OUT/'fresh_protocol.json'),
        previous_selected_comparison_model=p.comparison_name,previous_comparison_model_sha256=p.comparison_sha256,
        article='https://www.dio.me/articles/por-que-a-acuracia-accuracy-nao-e-a-melhor-metrica-para-avaliar-o-desempenho-de-classificadores',arduino_compiled=False)
    device=json.loads((OUT/'device_files_before_sha256.json').read_text());assert all(sha(ROOT/path)==h for path,h in device.items())
    decision['historical_device_files_preserved']=len(device)
    (OUT/'reliability_decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False));verify()
    print('RELIABILITY EVALUATION COMPLETE',json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
