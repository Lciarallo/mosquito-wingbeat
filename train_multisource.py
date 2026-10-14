"""Train a multi-million-parameter audio CNN, calibrate, freeze and test.

Computer evaluation only. No Arduino headers, packages, builds or uploads.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import time
import copy
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.nn import functional as F
from scipy.signal import resample_poly
from scipy.special import expit,softmax,logsumexp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from large_audio_model import LargeAudioCNN,predict,VERSION,SAMPLES,load_model
from train_expanded import source_weights,decide,infer_portable
from improve_device import atomic_json,file_hash

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/multisource'
CACHE=ROOT/'data/cache/multisource'
SEED=20261002


def bucket_predict(model,frame,indices,waves,device):
    logits=np.zeros((len(indices),20),dtype='float32');gates=np.zeros(len(indices),dtype='float32')
    for bucket in ['long','short']:
        mask=frame.iloc[indices].bucket.eq(bucket).to_numpy()
        if not mask.any():continue
        wi=frame.iloc[indices[mask]].wave_index.to_numpy().astype(int)
        p,g=predict(model,waves[bucket][wi],device)
        logits[mask]=p;gates[mask]=g
    return logits,gates


def weighted_metrics(frame,p,g,classes,pt,st,model_name):
    target=frame.label.map({name:i for i,name in enumerate(classes)}).fillna(-1).to_numpy().astype(int)
    _,confirmed=decide(frame,p,g,pt,st)
    confirmed[(frame.start_s+frame.step_s).to_numpy()<2.976-1e-7]=-1
    guessed=p.argmax(axis=1);emitted=confirmed>=0;correct=confirmed==target
    rows,per_class=[],[]
    output=frame[['path','group','domain','label','start_s','quality_ok','id']].copy()
    output['model']=model_name;output['forced_species']=[classes[i] for i in guessed]
    output['identified_species']=[classes[i] if i>=0 else '-' for i in confirmed]
    output['presence_score']=g;output['class_score']=p.max(axis=1)
    for domain in frame.domain.unique():
        mask=frame.domain.eq(domain).to_numpy();positive=mask&(target>=0)
        noise=mask&frame.label.eq('noise').to_numpy();unknown=mask&frame.label.eq('unknown_mosquito').to_numpy()
        source_p=pd.DataFrame(p[positive],index=frame.loc[positive,'group']).groupby(level=0).mean()
        source_y=frame.loc[positive].groupby('group').label.first().map({c:i for i,c in enumerate(classes)}).reindex(source_p.index)
        recalls=[(guessed[positive&(target==i)]==i).mean() for i in np.unique(target[positive])]
        rows.append(dict(model=model_name,domain=domain,known_windows=int(positive.sum()),known_groups=int(frame.loc[positive,'group'].nunique()),
            forced_window_accuracy=float((guessed[positive]==target[positive]).mean()),forced_balanced_accuracy=float(np.mean(recalls)),
            forced_recording_accuracy=float((source_p.to_numpy().argmax(axis=1)==source_y.to_numpy()).mean()),
            accepted_known_windows=int((emitted&positive).sum()),known_coverage=float(emitted[positive].mean()),
            accepted_known_accuracy=float(correct[positive&emitted].mean()) if (positive&emitted).any() else None,
            correct_known_fraction=float((correct&emitted)[positive].mean()),noise_windows=int(noise.sum()),
            noise_false_species_rate=float(emitted[noise].mean()) if noise.any() else None,
            presence_known_recall=float((g[positive]>=pt).mean()),presence_noise_false_positive_rate=float((g[noise]>=pt).mean()) if noise.any() else None,
            unknown_windows=int(unknown.sum()),unresolved_species_emission_rate=float(emitted[unknown].mean()) if unknown.any() else None))
        for i,name in enumerate(classes):
            truth=positive&(target==i)
            if not truth.any():continue
            per_class.append(dict(model=model_name,domain=domain,species=name,windows=int(truth.sum()),groups=int(frame.loc[truth,'group'].nunique()),
                forced_recall=float((guessed[truth]==i).mean()),coverage=float(emitted[truth].mean()),correctly_identified_fraction=float((correct&emitted)[truth].mean()),
                accepted_accuracy=float(correct[truth&emitted].mean()) if (truth&emitted).any() else None))
    return rows,per_class,output


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=False
    frame=pd.read_csv(OUT/'segments.csv',low_memory=False)
    assert frame.groupby('group').split.nunique().max()==1
    classes=np.load(ROOT/'results/expanded/models/species_fold0.npz')['classes'].tolist()
    y=frame.label.map({name:i for i,name in enumerate(classes)}).fillna(-1).to_numpy().astype('int64')
    presence=frame.label.ne('noise').to_numpy().astype('float32')
    def indices(split):return np.flatnonzero(frame.split.eq(split)&frame.quality_ok)
    train,val,cal,test=[indices(split) for split in ['train','validation','calibration','test']]
    # Test keeps invalid windows as rejected decisions in the comparison.
    test=np.flatnonzero(frame.split.isin(['test','challenge']))
    waves={b:np.load(CACHE/f'{b}.npy',mmap_mode='r') for b in ['long','short']}
    model,parent=load_model(ROOT/'results/large_cnn/models/large_cnn.pt',device)
    parameters=sum(p.numel() for p in model.parameters())
    train_wave={b:torch.tensor(np.array(waves[b]),device=device) for b in ['long','short']}
    train_bucket=frame.iloc[train].bucket.to_numpy();train_wi=frame.iloc[train].wave_index.to_numpy().astype(int)
    train_y=torch.tensor(y[train],device=device)
    train_presence=torch.tensor(presence[train],device=device)
    # Sampling balances species and sources, with noise and unresolved positives
    # each assigned a bounded total batch share, rather than treating all as a class.
    sampling=np.zeros(len(train),dtype='float64')
    known=y[train]>=0;noise=presence[train]==0;unknown=~known&~noise
    for mask,target,mass in [(known,y[train],.75),(noise,presence[train],.15),(unknown,presence[train],.10)]:
        if mask.any():
            values=source_weights(frame.iloc[train[mask]],target[mask]);sampling[mask]=values/values.sum()*mass
    sampling_numpy=sampling.copy()
    sampling_short=torch.tensor(sampling,device=device)
    sampling=torch.tensor(sampling*(train_bucket=='long'),device=device)
    class_indices={int(i):torch.nonzero(train_y.eq(int(i)),as_tuple=True)[0] for i in np.unique(y[train][known])}
    noise_indices=torch.nonzero(train_presence.eq(0),as_tuple=True)[0]
    def training_waves(ix,size):
        arr=ix.detach().cpu().numpy();output=torch.empty((len(arr),size),device=device)
        for b in ['long','short']:
            loc=np.flatnonzero(train_bucket[arr]==b)
            if len(loc):output[torch.tensor(loc,device=device)]=train_wave[b][torch.tensor(train_wi[arr[loc]],device=device)][:,:size]
        return output
    val_weights=source_weights(frame.iloc[val],np.where(y[val]>=0,y[val],np.where(presence[val]==0,20,21)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=.00001)
    best_loss=float('inf');best=None;best_epoch=0;stale=0;history=[];started=time.time()
    atomic_json(OUT/'protocol.json',dict(version=VERSION,seed=SEED,parameters=parameters,architecture='Residual spectrogram CNN, 2 channels, 32/64/128/256, dense512, species20+presence1',
        fft=2048,hop=256,mel_bands=128,min_frequency=80,max_frequency=4000,samples=SAMPLES,
        max_epochs=30,patience=8,batch_size=96,selection='weighted validation cross-entropy + 0.4 presence BCE',
        source_splits_sha256=file_hash(OUT/'segments.csv'),initial_checkpoint_sha256=file_hash(ROOT/'results/large_cnn/models/large_cnn.pt'),short_samples=4800,mixed_duration_training=True,ufrgs_entirely_held_out=True,
        augmentations=['training noise mixture, SNR 5..30dB','same-species synthetic mixture, never test samples','small time/frequency masks','spectral tilt, no pitch shift'],
        sampling_mass=dict(known=.75,noise=.15,unresolved_positive=.10),
        physical_board_tested=False,arduino_compiled=False))
    print('Model',parameters,'parameters;',device,';',len(train),'training windows',flush=True)
    for epoch in range(1,31):
        epoch_start=time.time();model.train();loss_sum=0;steps=0
        sampled=torch.multinomial(sampling,len(train),replacement=True)
        for start in range(0,len(sampled),96):
            size=4800 if steps%2==0 else SAMPLES
            ix=torch.multinomial(sampling_short if size==4800 else sampling,min(96,len(train)-start),replacement=True)
            wave=training_waves(ix,size);target=train_y[ix];binary=train_presence[ix]
            wave-=wave.mean(dim=1,keepdim=True);wave/=wave.square().mean(dim=1,keepdim=True).sqrt().clamp_min(1e-6)
            if len(noise_indices):
                other=training_waves(noise_indices[torch.randint(len(noise_indices),(len(wave),),device=device)],size)
                other=other-other.mean(dim=1,keepdim=True);other=other/other.square().mean(dim=1,keepdim=True).sqrt().clamp_min(1e-6)
                use=(torch.rand(len(wave),device=device)<.6)&binary.eq(1)
                snr=5+25*torch.rand(len(wave),device=device)
                wave+=other*(10**(-snr/20)*use).unsqueeze(1)
            # Synthetic overlapping same-species flight tones; not real swarm proof.
            for label in torch.unique(target[target>=0]).tolist():
                members=torch.nonzero(target.eq(label)&(torch.rand(len(target),device=device)<.2),as_tuple=True)[0]
                if len(members):
                    pool=class_indices[label]
                    if size==SAMPLES:pool=pool[torch.tensor(train_bucket[pool.cpu().numpy()]=='long',device=device)]
                    if not len(pool):continue
                    other=training_waves(pool[torch.randint(len(pool),(len(members),),device=device)],size)
                    other=other-other.mean(dim=1,keepdim=True);other/=other.square().mean(dim=1,keepdim=True).sqrt().clamp_min(1e-6)
                    wave[members]+=.3*other
            optimizer.zero_grad(set_to_none=True)
            logits,gate=model(wave,augment=True)
            k=target>=0
            species_loss=F.cross_entropy(logits[k],target[k],label_smoothing=.05) if k.any() else logits.sum()*0
            loss=species_loss+.4*F.binary_cross_entropy_with_logits(gate,binary)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            loss_sum+=float(loss.detach());steps+=1
        scheduler.step()
        vl,vg=bucket_predict(model,frame,val,waves,device)
        known_val=y[val]>=0
        ce=logsumexp(vl[known_val],axis=1)-vl[known_val,np.maximum(y[val][known_val],0)]
        vloss=float(np.average(ce,weights=val_weights[known_val]))+.4*float(np.average(np.logaddexp(0,vg)-presence[val]*vg,weights=val_weights))
        accuracy=float(np.average(vl[known_val].argmax(axis=1)==y[val][known_val],weights=val_weights[known_val]))
        entry=dict(epoch=epoch,training_loss=loss_sum/steps,validation_loss=vloss,validation_accuracy=accuracy,seconds=time.time()-epoch_start)
        history.append(entry);print(json.dumps(entry),flush=True)
        atomic_json(OUT/'history.json',history)
        if vloss<best_loss-1e-4:
            best_loss=vloss;best_epoch=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(best);model.eval()
    logits,gates=bucket_predict(model,frame,cal,waves,device)
    known_cal=y[cal]>=0
    cw=source_weights(frame.iloc[cal[known_cal]],y[cal][known_cal])
    def temperature_loss(log_t):
        z=logits[known_cal]/np.exp(log_t)
        return np.average(logsumexp(z,axis=1)-z[np.arange(len(z)),y[cal][known_cal]],weights=cw)
    temperature=float(np.exp(minimize_scalar(temperature_loss,bounds=(np.log(.25),np.log(4)),method='bounded').x))
    p=softmax(logits/temperature,axis=1)
    calibration=LogisticRegression(C=10,max_iter=2000).fit(gates[:,None],presence[cal].astype(int),sample_weight=source_weights(frame.iloc[cal],presence[cal]))
    slope,bias=float(calibration.coef_[0,0]),float(calibration.intercept_[0]);g=expit(slope*gates+bias)
    negatives=presence[cal]==0;bw=source_weights(frame.iloc[cal],presence[cal])
    pt=1.0001
    for threshold in np.r_[np.linspace(.01,.99,197),1.0001]:
        if np.average(g[negatives]>=threshold,weights=bw[negatives])<=.1:pt=float(threshold);break
    st=1.0001;search=[]
    for threshold in np.linspace(.05,.99,189):
        accepted=(p[known_cal].max(axis=1)>=threshold)&(g[known_cal]>=pt)
        coverage=float(cw[accepted].sum()/cw.sum())
        accuracy=float(np.average(p[known_cal].argmax(axis=1)[accepted]==y[cal][known_cal][accepted],weights=cw[accepted])) if accepted.any() else None
        count=int(frame.iloc[cal[known_cal][accepted]].group.nunique())
        search.append(dict(threshold=float(threshold),accuracy=accuracy,coverage=coverage,groups=count))
        if accuracy is not None and accuracy>=.8 and coverage>=.1 and count>=15:st=float(threshold);break
    saved=dict(version=VERSION,state_dict=best,classes=classes,temperature=temperature,
               presence_slope=slope,presence_bias=bias,presence_threshold=pt,species_threshold=st,
               parameters=parameters,best_epoch=best_epoch)
    checkpoint=OUT/'models/multisource_cnn.pt';torch.save(saved,checkpoint)
    frozen=datetime.now(timezone.utc).isoformat();sha=file_hash(checkpoint)
    atomic_json(OUT/'training_audit.json',dict(frozen_utc=frozen,checkpoint_sha256=sha,parameters=parameters,best_epoch=best_epoch,
        seconds=time.time()-started,calibration=dict(temperature=temperature,presence_slope=slope,presence_bias=bias,presence_threshold=pt,species_threshold=st,threshold_search=search),
        private_audio_used_for_training=False,previous_mosquitosong_audio_used_for_training=False,ufrgs_used_for_training=False,short_windows_trained=True,arduino_compiled=False,baseline_replaced=False))
    atomic_json(OUT/'reliability_decision.json',dict(status='PENDING_NEW_POST_TRAINING_AUDIO',model_sha256=sha,arduino_compile_allowed=False))
    tl,tg=bucket_predict(model,frame,test,waves,device);p=softmax(tl/temperature,axis=1);g=expit(slope*tg+bias)
    rows,per_class,output=weighted_metrics(frame.iloc[test].reset_index(drop=True),p,g,classes,pt,st,'multisource_cnn')
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(per_class).to_csv(OUT/'per_class.csv',index=False)
    output.to_csv(OUT/'test_predictions.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    pd.concat([pd.read_csv(ROOT/'results/large_cnn/metrics.csv'),pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/'comparison.csv',index=False)
    print('FROZEN',sha,flush=True);print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
