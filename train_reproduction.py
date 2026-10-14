"""Predefined validation-only model search, calibration, and immutable freeze.

Run training before evaluate_reproduction.py; never select by historical/private
or post-freeze evaluation results. All previous experimental artifacts preserved.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,time,subprocess
import numpy as np,pandas as pd,torch,joblib
from scipy.signal import resample_poly
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,roc_curve,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from aedes_binary import ROOT,targets,weights
from corrected_validation import assert_disjoint,sha
from reproduction_models import build_model,normalize,cross_domain_loss

OUT=ROOT/'results/reproduction_2026_10_01'
CACHE=ROOT/'data/cache/reproduction_2026_10_01'
SEED=20261007
CONFIGS=[dict(name='fernandes_clean',augmentation=False,alignment=False,epochs=25,lr=.0001),
         dict(name='fernandes_noise',augmentation=True,alignment=False,epochs=35,lr=.0001),
         dict(name='mosquitosong_binary',augmentation=True,alignment=False,epochs=40,lr=.0001),
         dict(name='residual_noise',augmentation=True,alignment=False,epochs=30,lr=.0002),
         dict(name='residual_domain_swarm',augmentation=True,alignment=True,epochs=35,lr=.0002)]


def now():return datetime.now(timezone.utc).isoformat()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))


def source_data():
    f=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);assert_disjoint(f)
    return f,{b:np.load(ROOT/f'data/cache/multisource/{b}.npy',mmap_mode='r') for b in ['long','short']}


def wave8_cache(waves):
    CACHE.mkdir(parents=True,exist_ok=True);result={}
    for b,x in waves.items():
        path=CACHE/(b+'_8k.npy')
        if not path.exists():
            out=np.lib.format.open_memmap(path,mode='w+',shape=(len(x),x.shape[1]//2),dtype='float32')
            for start in range(0,len(x),256):out[start:start+256]=resample_poly(np.asarray(x[start:start+256]),1,2,axis=1)
            out.flush()
        result[b]=np.load(path,mmap_mode='r')
    return result


def infer(model,name,frame,waves,device,batch=96):
    model.eval();out=np.empty(len(frame))
    with torch.no_grad():
        for bucket in ['long','short']:
            idx=np.flatnonzero(frame.bucket.eq(bucket));wi=frame.iloc[idx].wave_index.astype(int).to_numpy()
            for start in range(0,len(idx),batch):
                w=torch.tensor(np.array(waves[bucket][wi[start:start+batch]]),device=device)
                if name=='mosquitosong_binary' and w.shape[1]>2400:
                    pieces=torch.cat([w[:,offset:offset+2400] for offset in [0,2400,4800]])
                    p=expit(model(pieces)[0].cpu().numpy()).reshape(3,len(w)).mean(0)
                    z=logit(np.clip(p,1e-6,1-1e-6))
                else:z=model(w)[0].cpu().numpy()
                out[idx[start:start+batch]]=z
    return out


def choose_binary_cut(y,z,sample_weight,*,partition):
    if partition!='calibration':raise ValueError('Choose the binary cut only on calibration data.')
    fpr,tpr,cut=roc_curve(y,z,sample_weight=sample_weight)
    valid=np.isfinite(cut)&(tpr>=.6)&((1-fpr)>=.6)
    if not valid.any():valid=np.isfinite(cut)
    index=np.flatnonzero(valid)[np.argmax((tpr-fpr)[valid])]
    return float(cut[index])


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    if (OUT/'freeze.json').exists():raise FileExistsError('Already frozen: use a new experiment, never retune a tested one.')
    f,w=source_data();wv=wave8_cache(w);y=targets(f)
    frames={s:f[f.split.eq(s)&f.quality_ok&(y>=0)].copy() for s in ['train','validation','calibration']}
    train,val,cal=[frames[s] for s in ['train','validation','calibration']]
    old=json.loads((ROOT/'results/corrected_2026_10_01/original_weights_sha256.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in old.items());write('original_weights_sha256.json',old)
    repos={n:subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/f'tmp/reproductions_2026_10_01/{n}',text=True).strip()
           for n in ['fernandes','mosquitosongp']}
    protocol=dict(created_utc=now(),seed=SEED,target='Aedes aegypti versus other known species and noise',
        primary_metric='Maximize source/species/recording-balanced validation accuracy at logit 0 (probability 0.5 for trees) plus 0.001 weighted validation AUC.',
        target_balanced_accuracy=.60,selection_partition='validation',threshold_partition='calibration',configs=CONFIGS,
        stopping='Predefined 5 architectures, early stopping patience 7; no selection by test/private recordings.',
        source_splits_sha256=sha(ROOT/'results/multisource/segments.csv'),repositories=repos,
        architecture_adaptations=['PyTorch implementation, one BCE output, grouped source sampling.',
            'Fernandes 60x40 mel image: correct frequency-time axis and bilinear time resizing for real duration, no waveform extension.',
            'MosquitoSong+ Figure 2 convolution/pooling/dense sizes; 300ms crops, average three disjoint real crops on long inputs.',
            'Residual/domain-alignment network inspired by papers, not an exact DR-BioL replication.'],
        unknowns_not_negative=True,ufrgs_completely_excluded_from_training=True,private_audio_in_training=False,
        original_artifacts_preserved=True,arduino_compiled=False,swarm_augmentation='Same-labelled training sources only; never mixes held-out or private audio.')
    write('protocol.json',protocol);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu';cache={b:torch.tensor(np.array(x),device=device) for b,x in wv.items()}
    bucket=train.bucket.to_numpy();wi=train.wave_index.astype(int).to_numpy()
    ty=torch.tensor(targets(train),device=device,dtype=torch.float32)
    tw=weights(train);vw=weights(val);sample=torch.tensor(tw,device=device)
    longsample=sample*torch.tensor(bucket=='long',device=device)
    species=torch.tensor(pd.Categorical(train.label).codes.astype('int64'),device=device)
    domain=torch.tensor(pd.Categorical(train.domain).codes.astype('int64'),device=device)
    noise_indices=np.flatnonzero(train.label.eq('noise')&(train.bucket.eq('long')))
    pos=sample*ty;neg=sample*(1-ty)
    def batch(ix,size,random_crop=True):
        ii=ix.cpu().numpy();out=torch.empty(len(ii),size,device=device)
        for b,cachewave in cache.items():
            loc=np.flatnonzero(bucket[ii]==b)
            if len(loc):
                wave=cachewave[torch.tensor(wi[ii[loc]],device=device)]
                maxstart=wave.shape[1]-size
                offsets=torch.randint(maxstart+1,(len(loc),1),device=device) if random_crop and maxstart else torch.zeros(len(loc),1,device=device,dtype=torch.int64)
                out[torch.tensor(loc,device=device)]=wave.gather(1,offsets+torch.arange(size,device=device)[None,:])
        return out
    records=[]
    for ci,config in enumerate(CONFIGS):
        name=config['name'];path=OUT/f'models/{name}.pt'
        if path.exists():
            record=json.loads((OUT/f'{name}_training.json').read_text());records.append(record);print('Resuming after completed',name,flush=True);continue
        torch.manual_seed(SEED+ci);np.random.seed(SEED+ci);model=build_model(name).to(device)
        optimizer=(torch.optim.RMSprop(model.parameters(),lr=config['lr']) if name.startswith('fernandes') else
                   torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.005))
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['lr']/10)
        best=-np.inf;stale=0;history=[];started=time.time()
        print('TRAIN',name,len(train),device,sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,config['epochs']+1):
            model.train();losses=[]
            for step in range((len(train)+127)//128):
                size=2400 if name=='mosquitosong_binary' or step%2==0 else 7936
                ix=torch.multinomial(sample if size==2400 else longsample,128,replacement=True)
                x=normalize(batch(ix,size));target=ty[ix]
                if config['augmentation']:
                    ni=torch.tensor(np.random.choice(noise_indices,len(ix)),device=device)
                    n=normalize(batch(ni,size));snr=5+25*torch.rand(len(ix),device=device)
                    use=torch.rand(len(ix),device=device)<.9
                    x+=n*(10**(-snr/20)*use).unsqueeze(1)
                    if config['alignment']:
                        # Mix real training recordings from the same binary class.
                        jpos=torch.multinomial(pos,len(ix),replacement=True);jneg=torch.multinomial(neg,len(ix),replacement=True)
                        ji=torch.where(target.eq(1),jpos,jneg)
                        if size==7936:
                            lp=longsample*ty;ln=longsample*(1-ty)
                            ji=torch.where(target.eq(1),torch.multinomial(lp,len(ix),True),torch.multinomial(ln,len(ix),True))
                        mix=normalize(batch(ji,size));usemix=(torch.rand(len(ix),device=device)<.5).float()
                        x+=mix*(usemix*(.2+.8*torch.rand(len(ix),device=device))).unsqueeze(1)
                    x*=torch.exp(torch.empty(len(x),1,device=device).uniform_(np.log(.25),np.log(4)))
                optimizer.zero_grad(set_to_none=True);z,emb=model(x)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target)
                if config['alignment']:loss+=cross_domain_loss(emb,species[ix],domain[ix])
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss.')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step();zv=infer(model,name,val,wv,device);yv=targets(val)
            ba=float(np.average((zv>=0)==yv,weights=vw));auc=float(roc_auc_score(yv,zv,sample_weight=vw))
            vl=float(np.average(np.logaddexp(0,zv)-yv*zv,weights=vw))
            row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_weighted_ba=ba,validation_weighted_auc=auc,
                     validation_loss=vl,elapsed_s=time.time()-started)
            history.append(row);write(name+'_history.json',history);print(name,json.dumps(row),flush=True)
            # AUC tie breaker is fixed before training; no test data consulted.
            score=ba+.001*auc
            if score>best+1e-4:
                best=score;bestepoch=epoch;beststate=copy.deepcopy({k:v.detach().cpu().clone() for k,v in model.state_dict().items()});stale=0
            else:stale+=1
            if epoch>=10 and stale>=7:break
        model.load_state_dict(beststate);zc=infer(model,name,cal,wv,device);yc=targets(cal);cw=weights(cal)
        calibrator=LogisticRegression(C=10,max_iter=2000).fit(zc[:,None],yc,sample_weight=cw)
        slope=float(calibrator.coef_[0,0]);bias=float(calibrator.intercept_[0])
        if slope<=0:raise ValueError('Calibration inverted the model; inspect validation before claiming success.')
        pc=expit(slope*zc+bias);cut=choose_binary_cut(yc,pc,cw,partition='calibration')
        saved=dict(name=name,state_dict=beststate,slope=slope,bias=bias,cut=cut,best_epoch=bestepoch)
        torch.save(saved,path)
        chosen=history[bestepoch-1];record=dict(name=name,**chosen,parameters=sum(p.numel() for p in model.parameters()),
            calibration_cut=cut,calibration_weighted_ba=float(np.average((pc>=cut)==yc,weights=cw)),checkpoint_sha256=sha(path))
        write(name+'_training.json',record);records.append(record);del model,optimizer;torch.cuda.empty_cache()
    # An alternative ML method on the same source-balanced engineered features.
    x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],axis=1)
    ti=train.index.to_numpy();vi=val.index.to_numpy();ca=cal.index.to_numpy()
    for depth in [12,None]:
        name='extra_trees_'+('deep' if depth is None else 'regularized')
        m=ExtraTreesClassifier(n_estimators=500,max_depth=depth,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=SEED)
        m.fit(x[ti],targets(train),sample_weight=tw)
        p=m.predict_proba(x[vi])[:,1];pc=m.predict_proba(x[ca])[:,1]
        cut=choose_binary_cut(targets(cal),pc,weights(cal),partition='calibration')
        path=OUT/f'models/{name}.joblib';joblib.dump(dict(model=m,cut=cut,name=name),path)
        records.append(dict(name=name,validation_weighted_ba=float(np.average((p>=.5)==targets(val),weights=vw)),
             validation_weighted_auc=float(roc_auc_score(targets(val),p,sample_weight=vw)),checkpoint_sha256=sha(path),calibration_cut=cut))
    pd.DataFrame(records).to_csv(OUT/'validation_search.csv',index=False)
    chosen=max(records,key=lambda r:r['validation_weighted_ba']+.001*r['validation_weighted_auc'])
    write('selection.json',dict(selected_model=chosen['name'],criterion=protocol['primary_metric'],
        validation_weighted_ba=chosen['validation_weighted_ba'],validation_weighted_auc=chosen['validation_weighted_auc'],
        validation_target_met=chosen['validation_weighted_ba']>=.6,selected_before_evaluation=True))
    assert all(sha(ROOT/p)==h for p,h in old.items())
    write('freeze.json',dict(frozen_utc=now(),models_sha256={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'models').iterdir()},
        source_sha256={p:sha(ROOT/p) for p in ['reproduction_models.py','train_reproduction.py','results/multisource/segments.csv']},
        protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=sha(OUT/'selection.json'),arduino_compiled=False))
    print('FROZEN',json.dumps(chosen),flush=True)


if __name__=='__main__':main()
