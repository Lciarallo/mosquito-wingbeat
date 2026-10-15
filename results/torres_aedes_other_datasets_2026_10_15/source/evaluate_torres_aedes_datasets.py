"""Frozen new detector across datasets; preserve test/training and duration scope."""
from pathlib import Path
import json,gzip
import numpy as np,pandas as pd,torch,joblib
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from train_torres_aedes_transfer import bank,describe
from ternary_protocol import SEGMENTS,read_audio,pcm_signatures,digest,excluded_drive_audio_mask
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'results/torres_aedes_transfer_2026_10_15';OUT=ROOT/'results/torres_aedes_other_datasets_2026_10_15'

def write(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def measure(df):
 known=df[df.target>=0];pred=known.prediction.to_numpy();y=known.target.to_numpy();ae=y==2;neg=y!=2
 exact=known[known.target.isin([0,1,2])]
 return dict(known=len(known),unknown_labels=int((df.target<0).sum()),aedes_total=int(ae.sum()),aedes_detected=int(((pred==2)&ae).sum()),aedes_recall=float((pred[ae]==2).mean())if ae.any()else None,negative_total=int(neg.sum()),negative_to_aedes=int(((pred==2)&neg).sum()),negative_to_aedes_fpr=float((pred[neg]==2).mean())if neg.any()else None,binary_accuracy=float(((pred==2)==ae).mean())if len(y)else None,ternary_accuracy=float((exact.prediction==exact.target).mean())if len(exact)and not (known.target==3).any()else None,ternary_denominator=len(exact)if not (known.target==3).any()else 0)

def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(4)
 protected=[RUN/'baseline_model.joblib',RUN/'with_torres_model.joblib',RUN/'selection_frozen.json',*list((ROOT/'output/ifsc_revisao_oral_2026_10_15').glob('*.pdf'))];before={str(p.relative_to(ROOT)):digest(p)for p in protected}
 meta=pd.read_csv(RUN/'partitions.csv');primary=pd.read_csv(RUN/'test_predictions.csv');primary=primary[primary.source.eq('primary')].merge(meta.loc[meta.origin.eq('primary'),['group','domain','split']],on='group',validate='many_to_one');assert primary.split.eq('test').all()
 primary['dataset']=primary.domain;primary['evaluation_scope']='held_out_primary_fold_0';primary['level']='group'
 selected=pd.read_csv(RUN/'selected_windows.csv');fitgroups=set(meta.loc[meta.origin.eq('primary')&meta.split.eq('train'),'group']);assert not set(primary.group)&fitgroups
 table=pd.read_csv(SEGMENTS,low_memory=False);external=table[table.domain.isin(['mosquitosong','ufrgs','figshare'])&~excluded_drive_audio_mask(table)&table.quality_ok.isin([True,'True','true',1,'1'])].copy()
 long=np.load(ROOT/'data/cache/reliability_2026_10_01/long_8k.npy',mmap_mode='r');short=np.load(ROOT/'data/cache/reliability_2026_10_01/short_8k.npy',mmap_mode='r')
 # Match trained per-group top3 RMS pooling. Both genuine durations are retained;
 # .3-second inference has the same statistics width, no padding/interpolation.
 external['actual_rms']=0.
 for bucket,waves in [('long',long),('short',short)]:
  ii=np.flatnonzero(external.bucket.eq(bucket));index=external.iloc[ii].wave_index.to_numpy(int)
  external.iloc[ii,external.columns.get_loc('actual_rms')]=np.sqrt(np.mean(np.asarray(waves[index],dtype=np.float32)**2,axis=1))
 eligible=external[external.domain.eq('figshare')|external.bucket.eq('long')]
 picked=eligible.sort_values(['group','actual_rms'],ascending=[True,False],kind='stable').groupby('group',sort=False).head(3).copy();picked.to_csv(OUT/'selected_external_windows.csv',index=False)
 # Figshare individual clip predictions are diagnostic only, grouped by broad
 # source labels. Keep them separate from the three-group primary report.
 fig=external[external.domain.eq('figshare')].drop_duplicates(['path','start_s']).copy();work=pd.concat([picked,fig],ignore_index=True)
 groups=[];clips=[]
 for variant in ['baseline','with_torres']:
  bundle=joblib.load(RUN/f'{variant}_model.joblib');front,keep,edges=bank(bundle['frontend']);assert np.array_equal(keep,bundle['kept_bands']);fx=np.zeros((len(work),len(keep)*5),np.float32)
  for bucket,waves in [('long',long),('short',short)]:
   ids=np.flatnonzero(work.bucket.eq(bucket))
   for start in range(0,len(ids),128):
    ix=ids[start:start+128];batch=np.asarray(waves[work.iloc[ix].wave_index.to_numpy(int)],dtype=np.float32)
    with torch.inference_mode():db=front(torch.from_numpy(resample_poly(batch,6,1,axis=1).copy())).numpy()
    fx[ix]=describe(db,keep)
  pooled=pd.DataFrame(fx[:len(picked)]);pooled['group']=picked.group.to_numpy();pooled=pooled.groupby('group',sort=True).mean()
  with threadpool_limits(limits=4):probs=bundle['model'].predict_proba(pooled.to_numpy());fp=bundle['model'].predict_proba(fx[len(picked):])
  for g,p in zip(pooled.index,probs):
   s=external[external.group.eq(g)];labels=set(s.label);label=next(iter(labels));assert len(labels)==1
   target=2 if label=='Aedes aegypti'else 3 if label=='not_aedes'else -1 if label=='unknown_mosquito'else 1
   groups.append(dict(model=variant,dataset=s.domain.iloc[0],group=g,label=label,target=target,prediction=int(p.argmax()),p0=float(p[0]),p1=float(p[1]),p2=float(p[2]),duration_scope='0.3s_only'if set(s.bucket)=={'short'}else'0.992s_selected',evaluation_scope='external_known_source',level='group'))
  for (_,r),p in zip(fig.iterrows(),fp):clips.append(dict(model=variant,dataset='figshare',path=r.path,group=r.group,target=2 if r.label=='Aedes aegypti'else -1,prediction=int(p.argmax()),p2=float(p[2]),level='clip'))
 rows=pd.concat([primary[['model','dataset','group','target','prediction','p0','p1','p2','evaluation_scope','level']],pd.DataFrame(groups)],ignore_index=True);rows.to_csv(OUT/'group_predictions.csv',index=False);pd.DataFrame(clips).to_csv(OUT/'figshare_clip_predictions.csv',index=False)
 result={}
 for (variant,dataset),f in rows.groupby(['model','dataset']):result.setdefault(variant,{})[dataset]=measure(f)
 clipmetrics={name:measure(f)for name,f in pd.DataFrame(clips).groupby('model')}
 # Exact/aligned PCM overlap audit on all selected external recordings plus all
 # Figshare clips; each file is read only and never enters fitting/calibration.
 paths=sorted(set(work.path));training=pd.read_csv(ROOT/'results/ternary_verified_2026_10_14_corrections/source_pcm_audit.csv');knownbytes=set(training.file_sha256)
 with gzip.open(ROOT/'results/ternary_verified_2026_10_14_corrections/pcm_signatures.json.gz','rt')as f:knownpcm={s for arr in json.load(f).values()for s in arr}
 overlap=[]
 for name in paths:
  path=ROOT/name;overlap.append(dict(path=name,exact_byte_match=digest(path)in knownbytes,aligned_pcm_matches=len(pcm_signatures(read_audio(path,8000))&knownpcm)))
 write('overlap_audit.json',dict(files_checked=len(paths),overlaps=[r for r in overlap if r['exact_byte_match']or r['aligned_pcm_matches']],scope='Against full original-corpus PCM audit, stricter than only current train; Torres raw signals not available for all300 examples.'))
 assert not any(r['exact_byte_match']or r['aligned_pcm_matches']for r in overlap)
 assert before=={str(p.relative_to(ROOT)):digest(p)for p in protected}
 write('metrics.json',dict(group_level=result,figshare_clip_level=clipmetrics,primary_test_only=True,no_fit_no_threshold_adjustment=True,short_duration_shift_in_figshare=True,ufrgs_not_aedes_binary_only=True,unknown_figshare_not_used_as_negative=True))
 write('verification.json',dict(complete=True,protected_sha256=before,selected_models_and_pdf_unchanged=True,primary_train_groups_excluded=True,external_sources_absent_from_new_training=True,external_exact_pcm_overlap_detected=False,script_sha256=digest(Path(__file__)),no_presentation_changes=True))
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);print('FIGSHARE CLIPS',clipmetrics,flush=True)
if __name__=='__main__':main()
