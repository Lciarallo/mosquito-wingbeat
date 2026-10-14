"""Novos candidatos neurais, escolhidos pela validação por gravação.

Preserva a rodada inicial. Dados de teste já conhecidos: análise retrospectiva.
Não escolhe arquitetura/época pelos resultados internos de teste ou externos.
"""
from pathlib import Path
import argparse
import json
import time
import sys
import hashlib
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import soundfile as sf
from scipy.signal import resample_poly

from experimento_cnn import ROOT, frontend, evaluate, balanced_recording_weights, sha, save_json


class CNNFrequencia(nn.Module):
    """Mantém 16 posições de frequência, contra quatro na CNN inicial."""
    def __init__(self, classes=20):
        super().__init__()
        layers=[]
        for incoming,outgoing,pool in [(2,16,(2,2)),(16,24,(2,2)),(24,32,(1,2)),(32,48,(1,1))]:
            layers += [nn.Conv2d(incoming,outgoing,3,padding=1,bias=False),
                       nn.BatchNorm2d(outgoing),nn.GELU(),nn.AvgPool2d(pool)]
        self.encoder=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d((16,2)),nn.Flatten(),nn.Dropout(.4),
                                nn.Linear(1536,128),nn.GELU(),nn.Dropout(.4),nn.Linear(128,classes))
    def forward(self,x):
        return self.head(self.encoder(x))


class MLPProfunda(nn.Module):
    """Três camadas ocultas sobre média e desvio do mesmo frontend."""
    def __init__(self,classes=20):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(256,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.35),
                               nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.35),
                               nn.Linear(128,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.2),
                               nn.Linear(64,classes))
    def forward(self,x):
        return self.net(x)


def predict_logits(model,x,device):
    model.eval();values=[]
    with torch.inference_mode():
        for s in range(0,len(x),256):
            values.append(model(torch.tensor(np.array(x[s:s+256]),device=device)).cpu().numpy())
    return np.concatenate(values)


def recording_loss(frame,probs,classes):
    table=pd.DataFrame(probs,columns=classes)
    table['group']=frame.group.to_numpy()
    grouped=table.groupby('group')[classes].mean()
    truth=frame.groupby('group').label.first().reindex(grouped.index)
    target=truth.map({c:i for i,c in enumerate(classes)}).to_numpy()
    losses=-np.log(grouped.to_numpy()[np.arange(len(grouped)),target].clip(1e-12))
    # Cada espécie presente tem a mesma massa; cada gravação da espécie, também.
    counts=truth.value_counts()
    weights=1/truth.map(counts).to_numpy()
    return float(np.average(losses,weights=weights)), float((grouped.to_numpy().argmax(axis=1)==target).mean())


def fit_candidate(name,model,x,frame,y,tr,va,device,out,seed,epochs):
    torch.manual_seed(seed);np.random.seed(seed)
    model=model.to(device)
    tx=torch.tensor(x[tr],device=device);ty=torch.tensor(y[tr],device=device)
    weights=torch.tensor(balanced_recording_weights(frame.iloc[tr]),device=device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00003)
    best_loss=float('inf');best=None;stale=0;history=[];start=time.time()
    for epoch in range(1,epochs+1):
        model.train();order=torch.multinomial(weights,len(tr),replacement=True);losses=[]
        for s in range(0,len(order),128):
            idx=order[s:s+128];value=tx[idx]
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(value),ty[idx],label_smoothing=.03)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        vp=softmax(predict_logits(model,x[va],device),axis=1)
        vl,acc=recording_loss(frame.iloc[va],vp,sorted(frame.label.unique()))
        row=dict(model=name,epoch=epoch,train_loss=float(np.mean(losses)),
                 validation_recording_loss=vl,validation_recording_accuracy=acc)
        history.append(row);print(json.dumps(row),flush=True)
        if vl<best_loss-1e-4:
            best_loss=vl;stale=0;best_epoch=epoch
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(best)
    save_json(out/f'{name}_history.json',history)
    torch.save(dict(state_dict=best,classes=sorted(frame.label.unique()),model=name),out/f'{name}.pt')
    row=dict(model=name,best_epoch=best_epoch,epochs_run=len(history),
             validation_recording_loss=best_loss,seconds=time.time()-start,
             parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(out/f'{name}.pt'))
    return model,row


def external_features():
    records=pd.read_csv(ROOT/'results/species_nested_2026_10_09/external_recording_predictions.csv').drop_duplicates('path').sort_values('path')
    waves=[];metadata=[]
    for row in records.itertuples():
        audio,sr=sf.read(ROOT/row.path,dtype='float32',always_2d=True);audio=audio.mean(axis=1)
        if sr!=16000:
            div=np.gcd(sr,16000);audio=resample_poly(audio,16000//div,sr//div)
        for s in range(0,len(audio)-15872+1,15872):
            waves.append(audio[s:s+15872]);metadata.append(dict(path=row.path,group=row.group,label=row.label,start_s=s/16000))
    return np.array(waves),pd.DataFrame(metadata)


def main(out,epochs):
    if out.exists():raise FileExistsError(f'Resultados preservados: {out}')
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(20261012);np.random.seed(20261012)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_frame=pd.read_csv(ROOT/'results/expanded/segments.csv',low_memory=False)
    old=pd.read_csv(ROOT/'results/ifsc_cnn_2026_10_11/partitions.csv')
    frame=source_frame.loc[source_frame.domain.eq('dryad')&source_frame.label.ne('noise')&source_frame.quality_ok].copy()
    assert frame.path.tolist()==old.path.tolist()
    assert frame.split.tolist()==old.split.tolist()
    assert np.allclose(frame.start_s,old.start_s)
    assert frame.groupby('group').split.nunique().max()==1
    assert frame.groupby('path').split.nunique().max()==1
    assert frame.groupby('content_sha256').split.nunique().max()==1
    classes=sorted(frame.label.unique());y=frame.label.map({c:i for i,c in enumerate(classes)}).to_numpy()
    source=np.load(ROOT/'data/cache/large_cnn/waves.npy',mmap_mode='r')
    features=frontend(source[frame.index],device)
    summary=np.concatenate([features.mean(axis=3).reshape(len(features),-1),features.std(axis=3).reshape(len(features),-1)],axis=1)
    tr,va,te=[np.flatnonzero(frame.split.eq(s)) for s in ['train','validation','test']]
    scaler=StandardScaler().fit(summary[tr]);scaled=scaler.transform(summary).astype('float32')
    import joblib
    joblib.dump(scaler,out/'mlp_scaler.joblib')
    protocol=dict(seed=20261012,selection='menor entropia cruzada por gravação na validação, equilibrada por espécie',
                  selection_before_test_inference=True,epochs_max=epochs,patience=8,lr=.0006,weight_decay=.03,
                  label_smoothing=.03,optimizer='AdamW + cosine annealing',
                  candidates=['cnn_frequency','deep_mlp'],
                  changes=['CNN preserva 16 posições de frequência, contra 4 inicialmente',
                           'MLP: resumo espectral + três camadas ocultas, normalização ajustada no treino',
                           'dropout maior, AdamW mais regularizado, seleção por gravação'],
                  identical_historical_splits=True,calibration_not_used=True,
                  test_already_known_retrospective=True,external_already_known_retrospective=True,
                  private_audio_not_used=True,field_validated=False,firmware_changed=False,
                  source_code_sha256=sha(__file__),frontend_code_sha256=sha(ROOT/'large_audio_model.py'))
    save_json(out/'protocol.json',protocol)
    frame[['path','group','label','split','start_s','content_sha256']].to_csv(out/'partitions.csv',index=False)
    models={};rows=[]
    for name,klass,x in [('cnn_frequency',CNNFrequencia,features),('deep_mlp',MLPProfunda,scaled)]:
        # A inicialização também recebe semente definida antes de construir a rede.
        torch.manual_seed(20261012)
        model,row=fit_candidate(name,klass(len(classes)),x,frame,y,tr,va,device,out,20261012,epochs)
        models[name]=model;rows.append(row)
    selected=min(rows,key=lambda r:r['validation_recording_loss'])['model']
    save_json(out/'selection.json',dict(selected_model=selected,criterion=protocol['selection'],candidates=rows))
    save_json(out/'freeze.json',dict(selected_model=selected,model_sha256={r['model']:r['checkpoint_sha256'] for r in rows},
                                  scaler_sha256=sha(out/'mlp_scaler.joblib'),frozen_before_test_inference=True))
    # Só agora calcular as previsões do teste e do desafio externo.
    metrics=[]
    for name,x in [('cnn_frequency',features),('deep_mlp',scaled)]:
        p=softmax(predict_logits(models[name],x[te],device),axis=1)
        metrics.append(evaluate(frame.iloc[te],p,classes,name,out))
    save_json(out/'metrics.json',metrics);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    wave,ext_frame=external_features();ex=frontend(wave,device)
    es=np.concatenate([ex.mean(axis=3).reshape(len(ex),-1),ex.std(axis=3).reshape(len(ex),-1)],axis=1)
    external=[]
    for name,x in [('cnn_frequency',ex),('deep_mlp',scaler.transform(es).astype('float32'))]:
        p=softmax(predict_logits(models[name],x,device),axis=1)
        external.append(evaluate(ext_frame,p,classes,name+'_external',out))
    save_json(out/'external_metrics.json',external);pd.DataFrame(external).to_csv(out/'external_metrics.csv',index=False)
    print('SELECIONADO PELA VALIDAÇÃO:',selected,flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(pd.DataFrame(external).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/ifsc_melhorias_2026_10_11')
    p.add_argument('--epochs',type=int,default=40)
    args=p.parse_args();main(args.output,args.epochs)
