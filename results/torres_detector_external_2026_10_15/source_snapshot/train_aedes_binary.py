"""Binary fine-tuning with unchanged group splits; validation/calibration only selection."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,copy,hashlib
import numpy as np,pandas as pd,torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from aedes_binary import AedesBinaryCNN,targets,weights,predict,decisions,ROOT,POSITIVE

O=ROOT/'results/aedes_binary';PARENT=ROOT/'results/multisource/models/multisource_cnn.pt'
def write(name,value):(O/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))
def infer(model,frame,waves,device):
    z=np.empty(len(frame))
    for bucket in ['long','short']:
        mask=frame.bucket.eq(bucket).to_numpy()
        if mask.any():z[mask]=predict(model,waves[bucket][frame.loc[mask,'wave_index'].astype(int)],device)
    return z

def summary(f,p,low,high):
    y=targets(f);known=y>=0;valid=f.quality_ok.to_numpy();yp=p>=.5
    state=decisions(p,valid,low,high);positive=state=='AEDES_AEGYPTI';negative=state=='NAO_AEDES_AEGYPTI';accepted=positive|negative
    tp=int((positive&(y==1)).sum());fp=int((positive&(y==0)).sum())
    result=dict(windows=len(f),positive_windows=int((y==1).sum()),negative_windows=int((y==0).sum()),unknown_windows=int((y<0).sum()),
        forced_accuracy=float((yp[known]==y[known]).mean()) if known.any() else None,
        forced_aedes_recall=float(yp[y==1].mean()) if (y==1).any() else None,
        forced_false_positive_rate=float(yp[y==0].mean()) if (y==0).any() else None,
        balanced_accuracy=float(balanced_accuracy_score(y[known],yp[known])) if len(np.unique(y[known]))==2 else None,
        auc=float(roc_auc_score(y[known],p[known])) if len(np.unique(y[known]))==2 else None,
        positive_emissions=tp+fp,true_positive_emissions=tp,false_positive_emissions=fp,
        emitted_aedes_precision=tp/(tp+fp) if tp+fp else None,
        emitted_aedes_recall=tp/int((y==1).sum()) if (y==1).any() else None,
        emitted_false_positive_rate=fp/int((y==0).sum()) if (y==0).any() else None,
        known_coverage=float(accepted[known].mean()) if known.any() else None,
        accepted_accuracy=float(((positive[known]==y[known])[accepted[known]]).mean()) if accepted[known].any() else None,
        false_negative_emissions=int((negative&(y==1)).sum()),
        unresolved_positive_emissions=int((positive&(y<0)).sum()))
    return result,state

def main():
    O.mkdir(parents=True,exist_ok=True);(O/'models').mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261003);np.random.seed(20261003)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    f=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);assert f.groupby('group').split.nunique().max()==1
    y=targets(f);known=y>=0
    train=f[f.split.eq('train')&known&f.quality_ok].copy();val=f[f.split.eq('validation')&known&f.quality_ok].copy();cal=f[f.split.eq('calibration')&known&f.quality_ok].copy()
    assert not train.domain.eq('ufrgs').any()
    waves={b:np.load(ROOT/f'data/cache/multisource/{b}.npy',mmap_mode='r') for b in ['long','short']}
    model=AedesBinaryCNN().to(device);parent=torch.load(PARENT,map_location='cpu',weights_only=True)
    state={k:v for k,v in parent['state_dict'].items() if not k.startswith('species.')};model.load_state_dict(state,strict=False)
    with torch.no_grad():
        a=parent['classes'].index(POSITIVE);other=[i for i in range(20) if i!=a]
        model.species.weight.copy_(parent['state_dict']['species.weight'][a:a+1]-parent['state_dict']['species.weight'][other].mean(0,keepdim=True))
        model.species.bias.zero_()
    cache={b:torch.tensor(np.array(waves[b]),device=device) for b in waves}
    bucket=train.bucket.to_numpy();wi=train.wave_index.astype(int).to_numpy();ty=torch.tensor(targets(train),device=device,dtype=torch.float32)
    tw=weights(train);vw=weights(val);sample=torch.tensor(tw,device=device)
    longsample=sample*torch.tensor(bucket=='long',device=device)
    def batch(ix,size):
        ix=ix.cpu().numpy();out=torch.empty(len(ix),size,device=device)
        for b in cache:
            loc=np.flatnonzero(bucket[ix]==b)
            if len(loc):out[torch.tensor(loc,device=device)]=cache[b][torch.tensor(wi[ix[loc]],device=device)][:,:size]
        return out
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.01)
    write('protocol.json',dict(target='Aedes aegypti versus other labelled species, grouped non-Aedes and noise',unknown_mosquito_excluded_from_binary_loss=True,
        initial_checkpoint_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),split_sha256=hashlib.sha256((ROOT/'results/multisource/segments.csv').read_bytes()).hexdigest(),
        max_epochs=15,patience=4,selection='source-balanced validation binary cross entropy',sampling='50% Aedes; 50% non-Aedes equally divided among labels, domains and source groups',
        heldout_ufrgs=True,calibration_target_false_positive_rate=.05,short_windows_no_padding=True,private_audio_used_in_training=False,arduino_compiled=False))
    print('Binary training',len(train),'windows',device,flush=True)
    history=[];bestloss=float('inf');stale=0;started=time.time()
    for epoch in range(1,16):
        model.train();losses=[]
        for step in range((len(train)+95)//96):
            size=4800 if step%2==0 else 15872
            ix=torch.multinomial(sample if size==4800 else longsample,96,replacement=True)
            x=batch(ix,size);target=ty[ix]
            x-=x.mean(1,keepdim=True);x/=x.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)
            # Real noise augmentation only from training; no species-changing mixtures.
            noise=np.flatnonzero(train.label.eq('noise').to_numpy()&((bucket=='long') if size==15872 else True))
            if len(noise):
                ni=torch.tensor(np.random.choice(noise,len(ix)),device=device);n=batch(ni,size);n-=n.mean(1,keepdim=True);n/=n.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)
                use=(torch.rand(len(ix),device=device)<.5)&target.eq(1);snr=5+25*torch.rand(len(ix),device=device)
                x+=n*(10**(-snr/20)*use).unsqueeze(1)
            optim.zero_grad(set_to_none=True);z=model(x,augment=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();losses.append(float(loss.detach()))
        z=infer(model,val,waves,device);vy=targets(val);vl=float(np.average(np.logaddexp(0,z)-vy*z,weights=vw))
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=vl,validation_balanced_accuracy=float(balanced_accuracy_score(vy,z>=0)),elapsed_s=time.time()-started)
        history.append(row);write('history.json',history);print(json.dumps(row),flush=True)
        if vl<bestloss-1e-4:bestloss=vl;best=copy.deepcopy({k:v.cpu().clone() for k,v in model.state_dict().items()});bestepoch=epoch;stale=0
        else:stale+=1
        if stale>=4:break
    model.load_state_dict(best);z=infer(model,cal,waves,device);cy=targets(cal);cw=weights(cal)
    clf=LogisticRegression(C=10,max_iter=2000).fit(z[:,None],cy,sample_weight=cw);slope=float(clf.coef_[0,0]);bias=float(clf.intercept_[0]);assert slope>0
    p=expit(slope*z+bias);grid=np.r_[np.linspace(.001,.999,999),1.0001]
    high=next(float(t) for t in grid if t>=.5 and np.average(p[cy==0]>=t,weights=cw[cy==0])<=.05)
    low=0.
    for t in np.linspace(.001,min(.499,high-1e-6),499):
        if np.average(p[cy==1]<=t,weights=cw[cy==1])<=.05:low=float(t)
    saved=dict(state_dict=best,slope=slope,bias=bias,low=low,high=high,best_epoch=bestepoch,version='aedes-binary-cnn-v1')
    ck=O/'models/aedes_binary.pt';torch.save(saved,ck);sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    write('training_audit.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=sha,best_epoch=bestepoch,
        slope=slope,bias=bias,low=low,high=high,parameters=sum(p.numel() for p in model.parameters()),
        train_windows=len(train),train_aedes_windows=int((targets(train)==1).sum()),train_negative_windows=int((targets(train)==0).sum()),
        calibration=summary(cal,p,low,high)[0],seconds=time.time()-started,arduino_compiled=False))
    # All these domains have already been examined in earlier experiments: historical evaluation.
    test=f[f.split.isin(['test','challenge'])].copy();z=infer(model,test,waves,device);p=expit(slope*z+bias)
    rows=[];output=test[['path','label','group','domain','start_s','quality_ok']].copy();output['aedes_score']=p;output['forced_aedes']=p>=.5;output['state']=decisions(p,test.quality_ok.to_numpy(),low,high)
    for domain in test.domain.unique():
        mask=test.domain.eq(domain).to_numpy();r,_=summary(test[mask],p[mask],low,high);rows.append(dict(domain=domain,**r))
    pd.DataFrame(rows).to_csv(O/'historical_metrics.csv',index=False);output.to_csv(O/'historical_predictions.csv.gz',index=False)
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==sha
    print('FROZEN',sha,flush=True);print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
