"""Build duration buckets and source-family splits; retain old partitions intact."""
from pathlib import Path
import re,json,hashlib,collections
import numpy as np,pandas as pd,soundfile as sf
from scipy.signal import resample_poly
from train_large_cnn import wave_cache
from improve_device import file_hash,atomic_json
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/multisource';CACHE=ROOT/'data/cache/multisource';NEW=ROOT/'data/external/additional'
LONG=15872;SHORT=4800

def resample(path):
 w,sr=sf.read(path,dtype='float32',always_2d=True);w=w.mean(axis=1);g=np.gcd(sr,16000)
 return resample_poly(w,16000//g,sr//g) if sr!=16000 else w

def main():
 OUT.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 base=pd.read_csv(ROOT/'results/expanded/segments.csv',low_memory=False)
 waves=wave_cache(base);rows=[];long=[];short=[];excluded=[];seen={};old_hash=set(base.content_sha256.dropna());issues=[]
 def add(path,label,domain,group,split):
  digest=file_hash(path)
  if digest in old_hash:issues.append(dict(path=str(path),reason='duplicate_old_file'));return
  if digest in seen:issues.append(dict(path=str(path),reason='identical_new_file',first=seen[digest]));return
  seen[digest]=str(path.relative_to(ROOT));w=resample(path)
  if len(w)<SHORT:excluded.append(dict(path=str(path.relative_to(ROOT)),reason='less_than_0.3s',samples=len(w)));return
  size=LONG if len(w)>=LONG else SHORT
  for start in range(0,len(w)-size+1,size):
   chunk=w[start:start+size].copy();peak=np.max(np.abs(chunk));quality=peak>=32/32768 and np.mean(np.abs(chunk)>=32700/32768)<=.01
   dest=long if size==LONG else short;ix=len(dest);dest.append(chunk)
   rows.append(dict(path=str(path.relative_to(ROOT)),label=label,domain=domain,group=group,split=split,start_s=start/16000,step_s=size/16000,quality_ok=quality,bucket='long' if size==LONG else 'short',wave_index=ix,id=None,content_sha256=digest))
 # Canonical highest sample rate per numbered cut: variants are not independent.
 fig={}
 for p in sorted((NEW/'figshare').rglob('*.wav')):
  species=p.parts[-3];group=species+'/'+p.stem
  if group not in fig or sf.info(p).samplerate>sf.info(fig[group]).samplerate:fig[group]=p
 for group,p in fig.items():
  label='Aedes aegypti' if 'Aedes' in group else 'unknown_mosquito'
  # Original long-recording identity is unavailable: use the whole corpus only
  # for adaptation, never report a random clip split as independent validation.
  add(p,label,'figshare','figshare:'+p.parts[-3],'train')
 mapping={'ae.aegypti':'Aedes aegypti','ae.albopictus':'Aedes albopictus','an.dirus':'Anopheles dirus','cx.quin':'Culex quinquefasciatus'}
 families=collections.defaultdict(list)
 for p in sorted((NEW/'mosquitosong').glob('*.wav')):
  name=p.name.replace(' ','');family=re.split('_cut',name,flags=re.I)[0].replace('_LowMic','')
  families[family].append(p)
 # Both channels, every cut and age of a specimen/sex stay together in the
 # validation split for the other three species. Aedes age-5 historical
 # challenges are held out; both age-21 recordings are adaptation only.
 for family,paths in families.items():
  species=family.split('_')[0].lower();label=mapping[species];sex=family.split('_')[1]
  if species=='ae.aegypti':
   split='challenge' if '_5D_' in family else 'train';group='mosquitosong:'+family
  else:
   split='validation' if 'F' in sex else 'train';group='mosquitosong:'+species+'_'+sex
  for p in paths:add(p,label,'mosquitosong',group,split)
 for p in sorted((NEW/'ufrgs/dataset').rglob('*.wav')):
  label='not_aedes' if p.parent.name=='No_Aedes_aegypti' else 'Aedes aegypti'
  add(p,label,'ufrgs','ufrgs:'+p.parent.name,'test')
 # Existing windows retain exactly their split and waveform cache indices.
 old=base.copy();old['bucket']='long';old['wave_index']=np.arange(len(old))
 # Long chunks appended after existing cache; shift only additional indices.
 for row in rows:
  if row['bucket']=='long':row['wave_index']+=len(base)
 combined=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
 assert combined.groupby('group').split.nunique().max()==1
 longpath=CACHE/'long.npy'; dest=np.lib.format.open_memmap(longpath,mode='w+',dtype='float32',shape=(len(base)+len(long),LONG));dest[:len(base)]=waves
 if long:dest[len(base):]=np.stack(long)
 dest.flush();np.save(CACHE/'short.npy',np.stack(short))
 combined.to_csv(OUT/'segments.csv',index=False)
 pd.DataFrame(excluded).to_csv(OUT/'excluded_short_files.csv',index=False)
 pd.DataFrame(issues).to_csv(OUT/'duplicates.csv',index=False)
 pd.DataFrame(rows).groupby(['domain','label','split','bucket']).agg(windows=('path','size'),files=('path','nunique'),groups=('group','nunique')).to_csv(OUT/'additional_counts.csv')
 atomic_json(OUT/'data_audit.json',dict(additional_unique_files=len(seen),additional_long_windows=len(long),additional_short_windows=len(short),canonical_figshare_files=len(fig),excluded_too_short_files=len(excluded),duplicate_files=len(issues),old_partitions_unchanged=True,source_family_splits_disjoint=True,ufrgs_entirely_held_out=True,figshare_all_adaptation=True,figshare_individual_ids_unavailable=True,mosquitosong_aedes_5d_historical_challenges_excluded_from_train=True,mosquitosong_aedes_cross_age_individual_identity_unverified=True,private_audio_not_in_training=True,synthetic_not_used_ancestry_unavailable=True,biodcase_not_in_training_provenance_pending=True,segments_sha256=file_hash(OUT/'segments.csv'),no_waveform_padding_repetition_or_concatenation=True))
 print('Prepared',len(combined),'windows;',len(long),'additional long;',len(short),'short;',len(excluded),'excluded',flush=True)
 print(pd.DataFrame(rows).groupby(['domain','split']).size().to_string(),flush=True)
if __name__=='__main__':main()
