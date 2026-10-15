"""Controlled exploratory Aedes detection with Torres positives and real negatives.

Torres infection labels are discarded. Frontend uncertainty is documented, and
both original-corpus and Torres held-out tests remain separate.
"""
from pathlib import Path
import json,hashlib,time,copy
import joblib
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix,log_loss,balanced_accuracy_score
from threadpoolctl import threadpool_limits
import torch
from torres_lstm import MelSequence,digest
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/torres_aedes_transfer_2026_10_15'
SEED=20261016

def write(name,value):
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

def bank(mode):
 model=MelSequence(sample_rate=48000,n_fft=1024,hop=256,bands=32,fmin=0,fmax=24000)
 if mode=='htk':
  edges=700*(10**(np.linspace(0,2595*np.log10(1+24000/700),34)/2595)-1)
 else:
  # Slaney piecewise auditory scale and area-normalised triangular filters.
  hz=np.fft.rfftfreq(1024,1/48000);f_sp=200/3;logstep=np.log(6.4)/27
  maxmel=15+np.log(24000/1000)/logstep
  mel=np.linspace(0,maxmel,34);edges=np.where(mel<15,mel*f_sp,1000*np.exp(logstep*(mel-15)))
  weights=np.zeros((32,len(hz)),np.float32)
  for i in range(32):weights[i]=np.maximum(0,np.minimum((hz-edges[i])/(edges[i+1]-edges[i]),(edges[i+2]-hz)/(edges[i+2]-edges[i+1])))*2/(edges[i+2]-edges[i])
  model.bank.copy_(torch.tensor(weights))
 # Exclude bands touching >4 kHz: original cache has 8 kHz sampling.
 keep=np.flatnonzero(edges[2:]<=4000)
 return model,keep,edges

def describe(db,keep):
 # Same relative-dB scale for raw-audio frontend and author tensors; remove
 # two boundary frames in both. No time/band interpolation of author data.
 z=db[:,2:-2,keep];z=z-z.max((1,2),keepdims=True)
 z=np.clip(z,-80,0)
 return np.concatenate([z.mean(1),z.std(1),*np.quantile(z,[.1,.5,.9],axis=1)],axis=1).astype('float32')

def metrics(y,p):
 pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=[0,1,2]);tot=cm.sum(1)
 return dict(n=int(len(y)),accuracy=float(np.mean(y==pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),confusion_matrix=cm.tolist(),aedes_recall=float(cm[2,2]/tot[2]),noise_to_aedes_fpr=float(cm[0,2]/tot[0])if tot[0]else None,other_to_aedes_fpr=float(cm[1,2]/tot[1])if tot[1]else None,aedes_detected=int(cm[2,2]),aedes_total=int(tot[2]))

def main():
 OUT.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(SEED)
 protected=[ROOT/'results/ternary_lstm_2026_10_15/freeze.json',ROOT/'results/ternary_verified_2026_10_14_corrections/freeze.json',*list((ROOT/'output/ifsc_revisao_oral_2026_10_15').glob('*.pdf'))]
 hashes={str(p.relative_to(ROOT)):digest(p)for p in protected}
 frame=pd.read_csv(ROOT/'results/ternary_lstm_2026_10_15/segments.csv.gz',low_memory=False)
 assert not frame.domain.isin(['drive','drive_test','private_swarm','mosquitosong']).any()
 # Three most energetic windows per recording, before any model fitting.
 picked=frame.sort_values(['group','rms'],ascending=[True,False]).groupby('group',sort=False).head(3).sort_values('group').reset_index(drop=True)
 groups=picked.groupby('group',sort=True).agg(target=('true_label','max'),pcm_block=('pcm_block','first'),fold=('outer_fold','first'),domain=('domain','first')).reset_index()
 groups['target']=groups.group.map(frame.groupby('group').true_label.max()).to_numpy(int)
 groups['split']=np.where(groups.fold.eq(0),'test',np.where(groups.fold.eq(1),'validation','train'))
 for a in ['train','validation','test']:
  for b in ['train','validation','test']:
   if a!=b:assert not set(groups.loc[groups.split.eq(a),'pcm_block'])&set(groups.loc[groups.split.eq(b),'pcm_block'])
 author=json.loads((ROOT/'data/external/torres_2025/Dataset/Melspec.json').read_text());raw=np.asarray(author['mspec'],np.float32);assert raw.shape==(300,188,32)
 amap=pd.read_csv(ROOT/'results/torres_2025_2026_10_15/infection/partitions.csv');assert len(amap)==300
 assert [hashlib.sha256(x.tobytes()).hexdigest()for x in raw]==amap.content_sha256.tolist()
 assert amap.groupby('group').split.nunique().max()==1
 amap=amap.assign(target=2,domain='torres')
 partitions=pd.concat([groups.assign(origin='primary'),amap[['group','target','pcm_block','split','domain']].assign(origin='torres')],ignore_index=True)
 partitions.to_csv(OUT/'partitions.csv',index=False);picked[['group','path','start_s','wave_index','rms']].to_csv(OUT/'selected_windows.csv',index=False)
 (OUT/'analise-lacunas.txt').write_text('Tarefa: 3 classes, Aedes/outro mosquito/ruído. Os 300 tensores Torres são todos positivos, independentemente de infecção. Há parâmetros publicados (48 kHz, FFT1024, hop256, Hann, 32 mel), mas faltam receita completa/fmin/fmax/escala/norma. Testar duas representações aproximadas HTK e Slaney (0–24k), restringindo bandas abaixo de4k para compatibilidade física com cache8k. Reextrair áudio real em48k e resumir distribuições temporais, sem interpolar tensores do artigo. Seleção só por validação. Comparar novo baseline e novo modelo acrescido no mesmo teste. Torres sem identificação biológica: hashes exatos separados não comprovam animais independentes. Resultados exploratórios, sem alteração da apresentação ou modelos existentes.\n')
 settings=dict(seed=SEED,labels=['no_mosquito','other_mosquito','aedes_aegypti'],target_remapping='All Torres rows -> aedes_aegypti; infection labels never enter model',primary_test_fold=0,primary_validation_fold=1,primary_train_folds=[2,3,4],torres_split='Reuse frozen 192/48/60 exact-tensor split, unknown individual identities',frontend_candidates=['htk','slaney'],model='HistGradientBoostingClassifier on temporal mel statistics',hyperparameters=dict(max_iter=180,learning_rate=.06,max_leaf_nodes=[15,31],l2_regularization=1,min_samples_leaf=12,early_stopping=False),selection='minimum class-balanced log loss on validation; tie smaller max_leaf_nodes',rule='argmax of three probabilities',positive_weight='Augmented: 50% positive mass primary, 50% Torres. Each class total mass1; negatives unchanged.',retrospective_known_sources=True,official_frontend_fully_reproduced=False,no_presentation_edits=True)
 write('protocol.json',settings)
 waves=np.load(ROOT/'data/cache/reliability_2026_10_01/long_8k.npy',mmap_mode='r')
 candidates=[];feature_sets={}
 for mode in ['htk','slaney']:
  front,keep,edges=bank(mode);features=[]
  for start in range(0,len(picked),128):
   batch=np.asarray(waves[picked.wave_index.iloc[start:start+128].to_numpy(int)],dtype=np.float32)
   up=resample_poly(batch,6,1,axis=1)
   with torch.inference_mode():db=front(torch.from_numpy(up.copy())).numpy()
   features.append(describe(db,keep))
  window_x=np.concatenate(features);f=pd.DataFrame(window_x);f['group']=picked.group.to_numpy();x=f.groupby('group',sort=True).mean().loc[groups.group].to_numpy(np.float32)
  t=describe(raw,keep);feature_sets[mode]=(x,t,keep,edges)
  np.savez(OUT/f'features_{mode}.npz',primary=x,torres=t,kept_bands=keep,edges_hz=edges)
  print('FEATURES',mode,len(picked),'windows',x.shape,t.shape,flush=True)
  for augmented in [False,True]:
   fit=groups.split.eq('train').to_numpy();val=groups.split.eq('validation').to_numpy();at=amap.split.eq('train').to_numpy();av=amap.split.eq('validation').to_numpy()
   xt=x[fit];yt=groups.target.to_numpy()[fit];w=np.zeros(len(yt),np.float64)
   for c in range(3):w[yt==c]=1/(yt==c).sum()
   if augmented:
    w[yt==2]*=.5;xt=np.concatenate([xt,t[at]]);yt=np.r_[yt,np.full(at.sum(),2)];w=np.r_[w,np.full(at.sum(),.5/at.sum())]
   w*=len(w)/w.sum()
   # Same original-corpus validation; augmented selection also checks Torres validation.
   xv=x[val];yv=groups.target.to_numpy()[val];vw=np.zeros(len(yv))
   for c in range(3):vw[yv==c]=1/(yv==c).sum()
   if augmented:
    vw[yv==2]*=.5;xv=np.concatenate([xv,t[av]]);yv=np.r_[yv,np.full(av.sum(),2)];vw=np.r_[vw,np.full(av.sum(),.5/av.sum())]
   for leaves in [15,31]:
    model=HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=leaves,l2_regularization=1,min_samples_leaf=12,early_stopping=False,random_state=SEED)
    with threadpool_limits(limits=4):model.fit(xt,yt,sample_weight=w);p=model.predict_proba(xv)
    loss=float(log_loss(yv,p,sample_weight=vw,labels=[0,1,2]));item=dict(augmented=augmented,frontend=mode,leaves=leaves,validation_loss=loss)
    joblib.dump(model,OUT/f'candidate_{mode}_{int(augmented)}_{leaves}.joblib');candidates.append(item);print('FIT',item,flush=True)
 # Freeze choices BEFORE test inference; both models use validation only.
 selected={name:min([c for c in candidates if c['augmented']==aug],key=lambda c:(c['validation_loss'],c['leaves']))for name,aug in [('baseline',False),('with_torres',True)]}
 write('selection_frozen.json',dict(selected=selected,candidates=candidates,partitions_sha256=digest(OUT/'partitions.csv'),script_sha256=digest(Path(__file__)),test_predictions_not_yet_computed=True))
 results={};predictions=[]
 for name,sel in selected.items():
  model=joblib.load(OUT/f"candidate_{sel['frontend']}_{int(sel['augmented'])}_{sel['leaves']}.joblib");x,t,keep,edges=feature_sets[sel['frontend']]
  results[name]={}
  for source,matrix,table in [('primary',x,groups),('torres',t,amap)]:
   mask=table.split.eq('test').to_numpy();y=table.target.to_numpy()[mask]
   with threadpool_limits(limits=4):probs=model.predict_proba(matrix[mask])
   results[name][source]=metrics(y,probs)
   for gid,target,p in zip(table.group[mask],y,probs):predictions.append(dict(model=name,source=source,group=gid,target=int(target),prediction=int(p.argmax()),p0=float(p[0]),p1=float(p[1]),p2=float(p[2])))
  joblib.dump(dict(model=model,frontend=sel['frontend'],kept_bands=keep,edges_hz=edges,classes=settings['labels'],training_selection=sel),OUT/f'{name}_model.joblib')
 pd.DataFrame(predictions).to_csv(OUT/'test_predictions.csv',index=False)
 assert hashes=={str(p.relative_to(ROOT)):digest(p)for p in protected}
 write('metrics.json',results);write('verification.json',dict(complete=True,protected_hashes=hashes,existing_models_and_pdf_unchanged=True,primary_pcm_disjoint=True,torres_exact_hash_disjoint=True,unknown_individual_overlap=True,presentation_edits=False,raw_author_300_signals_unavailable=True,source_frontend_approximation=True))
 print('COMPLETE',json.dumps(results),flush=True)
if __name__=='__main__':main()
