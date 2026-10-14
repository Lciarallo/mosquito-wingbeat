"""Predeclared expanded-data search with gated mosquito/species classifiers.

No test/private selection, no Arduino work. Freeze all seven candidates before
obtaining the next external WAVs. Previous tests remain historical diagnostics.
"""
import json,time,copy
import numpy as np,pandas as pd,torch,joblib
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from train_reproduction import ROOT,now,sha
from fetch_reliability_data import OUT
from prepare_reliability_data import CACHE
from aedes_binary import targets,weights
from corrected_validation import assert_disjoint
from reliability_models import JointDetector,channel_augmentation
from reproduction_models import normalize
from reliability_metrics import binary_metrics
from robust_reproduction_ensemble import source_class_recalls

CONFIGS=[dict(name='residual_expanded',epochs=40,lr=.0002),dict(name='pcen_large',epochs=40,lr=.0002),
         dict(name='pcen_hybrid_large',epochs=40,lr=.0002),dict(name='raw_expanded',epochs=40,lr=.0001)]


def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))


def choose_cut(frame,p,*,partition):
    if partition!='calibration':raise ValueError('Only calibration labels may choose a threshold.')
    y=targets(frame);cut=np.unique(np.r_[0,np.quantile(p,np.linspace(0,1,201)),1.000001])
    pred=p[None,:]>=cut[:,None];correct=pred==y[None,:];cells=[]
    for d in frame.domain.unique():
        for label in [0,1]:
            mask=frame.domain.eq(d).to_numpy()&(y==label)
            if mask.any():cells.append(np.average(correct[:,mask],axis=1,weights=weights(frame.loc[mask])))
    tp=(pred&y[None,:].astype(bool)).sum(1);fp=(pred&~y[None,:].astype(bool)).sum(1)
    precision=tp/np.maximum(tp+fp,1);recall=tp/max(1,int((y==1).sum()))
    f1=2*tp/np.maximum(2*tp+fp+(y==1).sum()-tp,1)
    minimum=np.minimum.reduce(cells);feasible=(precision>=.6)&(recall>=.6)&(minimum>=.6)
    score=minimum+.05*f1
    i=np.flatnonzero(feasible)[np.argmax(score[feasible])] if feasible.any() else int(np.argmax(score))
    return float(cut[i]),dict(calibration_target_feasible=bool(feasible.any()),minimum_source_class_recall=float(minimum[i]),
                             precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]))


def infer(model,name,frame,waves,x,device,batch=64):
    model.eval();out=np.empty(len(frame))
    with torch.no_grad():
        for bucket in ['long','short']:
            ix=np.flatnonzero(frame.bucket.eq(bucket));wi=frame.iloc[ix].wave_index.astype(int).to_numpy()
            for start in range(0,len(ix),batch):
                index=ix[start:start+batch];w=torch.tensor(np.array(waves[bucket][wi[start:start+batch]]),device=device)
                features=torch.tensor(np.asarray(x[index],dtype='float32'),device=device)
                if name=='raw_expanded' and w.shape[1]>2400:
                    crops=torch.cat([w[:,off:off+2400] for off in [0,2400,4800]])
                    p=model(crops,features.repeat(3,1))[0].sigmoid().cpu().numpy().reshape(3,len(w)).mean(0)
                    z=logit(np.clip(p,1e-6,1-1e-6))
                else:z=model(w,features)[0].cpu().numpy()
                out[index]=z
    return out


def selection_score(frame,p):
    recalls=source_class_recalls(frame,p,.5)
    metric=binary_metrics(targets(frame),p,np.ones(len(frame),bool),.5)
    # The lowest source/class recall cannot be hidden by majority-class accuracy.
    score=min(recalls.values())+.05*metric['macro_f1']+.001*metric['average_precision']
    return float(score),dict(validation_source_class_recalls=recalls,validation_minimum_recall=min(recalls.values()),
                             validation_precision=metric['aedes_precision'],validation_recall=metric['aedes_recall'],
                             validation_macro_f1=metric['macro_f1'],validation_ap=metric['average_precision'])


def main():
    OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    if (OUT/'freeze.json').exists():raise FileExistsError('Already frozen; do not retune a tested experiment.')
    f=pd.read_csv(OUT/'segments.csv',low_memory=False);assert_disjoint(f)
    w={b:np.load(CACHE/(b+'_8k.npy'),mmap_mode='r') for b in ['long','short']};x=np.load(CACHE/'features.npy',mmap_mode='r')
    frames={s:f[f.split.eq(s)&f.quality_ok&(targets(f)>=0)] for s in ['train','validation','calibration']}
    tr,v,c=[frames[s] for s in ['train','validation','calibration']];ti=tr.index.to_numpy();tw=weights(tr)
    mean=np.average(x[ti],axis=0,weights=tw).astype('float32');scale=np.sqrt(np.average((x[ti]-mean)**2,axis=0,weights=tw)).clip(1e-4).astype('float32')
    original=json.loads((ROOT/'results/corrected_2026_10_01/original_weights_sha256.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in original.items());write('original_weights_sha256.json',original)
    protocol=dict(created_utc=now(),seed=20261012,configs=CONFIGS,
        target='Aedes versus known other taxa and environmental sounds',unknown_mosquito_excluded=True,
        training='Group/source/negative-taxon balancing; batches 48 Aedes,48 other mosquitoes,32 noise.',
        improvements=['Real hard negatives and ESC50 environmental sounds','Separate mosquito-presence and conditional Aedes heads',
            '9M-parameter residual PCEN networks','Simulated microphone equalization','SNR5-30dB real training-only noise mixture',
            'Training-only same-binary-class waveform mixtures, at most 50%; labeled simulations, not independent observations'],
        architecture='PCEN causal EMA and relative log-mel/harmonic contrast; hybrid clean physical features with 50% branch dropout paired to augmented waveform, same underlying label.',
        selection='Maximum minimum source/class validation recording-balanced recall at p=.5 + .05 validation macro-F1 + .001 validation AP.',
        threshold='Calibration only; maximize minimum source/class recall + .05 positive F1; require precision/recall/minimum recall>=.6 where feasible.',
        reliability_gate=dict(precision_min=.6,recall_min=.6,macro_f1_min=.6,minimum_source_class_recall=.6,
           source_group_bootstrap_lower95_min=.6,target_sensor_independent_recordings_required=True),
        previous_external_tests='Now development diagnostics; next test selected/downloaded only after freeze.',
        private_in_training=False,all_cdmsc_aedes_in_training=False,no_waveform_extension_at_inference=True,arduino_compiled=False,
        segments_sha256=sha(OUT/'segments.csv'),early_stop='Patience10, minimum12 epochs; max40 each predeclared configuration.')
    write('protocol.json',protocol);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    wc={b:torch.tensor(np.array(a),device=device) for b,a in w.items()}
    xt=torch.tensor(np.array(x[ti]),device=device);target=torch.tensor(targets(tr),device=device,dtype=torch.float32)
    present=torch.tensor(tr.label.ne('noise').to_numpy(),device=device,dtype=torch.float32)
    sample=torch.tensor(tw,device=device);bucket=tr.bucket.to_numpy();wi=tr.wave_index.astype(int).to_numpy()
    def indices(size):
        eligible=torch.tensor(bucket=='long',device=device) if size>2400 else torch.ones(len(tr),device=device)
        return torch.cat([torch.multinomial(sample*eligible*mask,n,replacement=True) for mask,n in
               [(target.eq(1),48),(target.eq(0)&present.eq(1),48),(present.eq(0),32)]])
    def wave(ix,size):
        ii=ix.cpu().numpy();out=torch.empty(len(ix),size,device=device)
        for b,a in wc.items():
            loc=np.flatnonzero(bucket[ii]==b)
            if len(loc):
                r=a[torch.tensor(wi[ii[loc]],device=device)];off=torch.randint(r.shape[1]-size+1,(len(loc),1),device=device)
                out[torch.tensor(loc,device=device)]=r.gather(1,off+torch.arange(size,device=device)[None,:])
        return out
    records=[]
    for ci,config in enumerate(CONFIGS):
        name=config['name'];path=OUT/f'models/{name}.pt'
        if path.exists():records.append(json.loads((OUT/(name+'_training.json')).read_text()));continue
        torch.manual_seed(20261012+ci);np.random.seed(20261012+ci)
        model=JointDetector(name,mean,scale).to(device);opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,config['epochs'],eta_min=config['lr']/10)
        best=-np.inf;stale=0;history=[];start=time.time()
        print('TRAIN',name,len(tr),'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,config['epochs']+1):
            model.train();losses=[]
            for step in range((len(tr)+127)//128):
                size=2400 if name=='raw_expanded' or step%2==0 else 7936
                ix=indices(size);a=channel_augmentation(wave(ix,size));yy=target[ix];pres=present[ix]
                ni=indices(size)[-32:];ni=ni[torch.randint(len(ni),(len(ix),),device=device)]
                noise=normalize(wave(ni,size));snr=5+25*torch.rand(len(ix),device=device)
                a+=noise*(10**(-snr/20)*(torch.rand(len(ix),device=device)<.8)).unsqueeze(1)
                if step%2==0:
                    # Simulated superposition; never mix true Aedes with other species.
                    for lo,hi in [(0,48),(48,96)]:
                        use=(torch.rand(hi-lo,device=device)<.5).unsqueeze(1)
                        a[lo:hi]+=a[lo:hi].roll(1,0)*use*(.2+.8*torch.rand(hi-lo,1,device=device))
                opt.zero_grad(set_to_none=True);joint,zp,zs=model(a,xt[ix])
                joint_loss=torch.nn.functional.binary_cross_entropy_with_logits(joint,yy,weight=torch.where(yy==1,4/3,.8))
                presence_loss=torch.nn.functional.binary_cross_entropy_with_logits(zp,pres,weight=torch.where(pres==1,2/3,2.))
                mosquito=pres.eq(1);species_loss=torch.nn.functional.binary_cross_entropy_with_logits(zs[mosquito],yy[mosquito])
                loss=joint_loss+.5*presence_loss+.25*species_loss
                if not torch.isfinite(loss):raise ValueError('Non-finite loss.')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            scheduler.step();z=infer(model,name,v,w,x[v.index],device);score,detail=selection_score(v,expit(z))
            row=dict(epoch=epoch,loss=float(np.mean(losses)),selection_score=score,elapsed_s=time.time()-start,**detail)
            history.append(row);write(name+'_history.json',history);print(name,json.dumps(row),flush=True)
            if score>best+1e-4:
                best=score;bestepoch=epoch;state=copy.deepcopy({k:a.detach().cpu().clone() for k,a in model.state_dict().items()});stale=0
            else:stale+=1
            if epoch>=12 and stale>=10:break
        model.load_state_dict(state);zc=infer(model,name,c,w,x[c.index],device)
        cal=LogisticRegression(C=10,max_iter=2000).fit(zc[:,None],targets(c),sample_weight=weights(c))
        slope,bias=float(cal.coef_[0,0]),float(cal.intercept_[0]);assert slope>0
        cut,calinfo=choose_cut(c,expit(slope*zc+bias),partition='calibration')
        saved=dict(name=name,state_dict=state,slope=slope,bias=bias,cut=cut)
        torch.save(saved,path);record=dict(name=name,**history[bestepoch-1],cut=cut,calibration=calinfo,parameters=sum(p.numel() for p in model.parameters()),sha256=sha(path))
        write(name+'_training.json',record);records.append(record);del model,opt;torch.cuda.empty_cache()
    # Alternative two-stage classical models with the same engineering features.
    for name in ['trees_regularized','trees_deep','hist_gradient_boosting']:
        if name.startswith('trees'):
            make=lambda:ExtraTreesClassifier(n_estimators=500,max_depth=16 if name=='trees_regularized' else None,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=20261012)
        else:make=lambda:HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,l2_regularization=3,learning_rate=.05,early_stopping=False,random_state=20261012)
        yi=targets(tr);pi=tr.label.ne('noise').astype(int).to_numpy();pw=tw.copy()
        for label in [0,1]:pw[pi==label]/=pw[pi==label].sum()
        pm=make().fit(x[ti],pi,sample_weight=pw/pw.mean());mask=pi==1;sw=weights(tr.loc[mask]);sm=make().fit(x[ti][mask],yi[mask],sample_weight=sw)
        def predict(indices):return np.clip(pm.predict_proba(x[indices])[:,1]*sm.predict_proba(x[indices])[:,1],1e-6,1-1e-6)
        pv=predict(v.index);pc=predict(c.index);score,detail=selection_score(v,pv)
        cal=LogisticRegression(C=10,max_iter=2000).fit(logit(pc)[:,None],targets(c),sample_weight=weights(c))
        slope,bias=float(cal.coef_[0,0]),float(cal.intercept_[0]);assert slope>0
        cut,calinfo=choose_cut(c,expit(slope*logit(pc)+bias),partition='calibration')
        path=OUT/f'models/{name}.joblib';joblib.dump(dict(name=name,presence_model=pm,species_model=sm,slope=slope,bias=bias,cut=cut),path)
        record=dict(name=name,selection_score=score,**detail,cut=cut,calibration=calinfo,sha256=sha(path));write(name+'_training.json',record);records.append(record)
        print('CLASSICAL',json.dumps(record),flush=True)
    chosen=max(records,key=lambda r:r['selection_score']);write('validation_search.json',records)
    write('selection.json',dict(selected_model=chosen['name'],selection_score=chosen['selection_score'],selected_before_new_test=True,
        validation_minimum_recall=chosen['validation_minimum_recall']))
    assert all(sha(ROOT/p)==h for p,h in original.items())
    write('freeze.json',dict(frozen_utc=now(),models_sha256={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'models').iterdir()},
        source_sha256={p:sha(ROOT/p) for p in ['reliability_models.py','train_reliability.py','reliability_metrics.py','results/reliability_2026_10_01/segments.csv']},
        protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=sha(OUT/'selection.json'),arduino_compiled=False))
    print('RELIABILITY FROZEN',json.dumps(chosen),flush=True)


if __name__=='__main__':main()
