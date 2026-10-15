"""Decode official D1-D4 test audio for an auxiliary, provenance-limited benchmark."""
from pathlib import Path
import zipfile,re,json,io,hashlib
import numpy as np,pandas as pd,soundfile as sf
from scipy.signal import resample_poly
from improve_device import atomic_json,file_hash
ROOT=Path(__file__).resolve().parent;D=ROOT/'data/external/additional';O=ROOT/'results/multisource';C=ROOT/'data/cache/biodcase_benchmark'
MAP={1:'Aedes aegypti',2:'Aedes albopictus',3:'Culex quinquefasciatus',4:'Anopheles gambiae',5:'Anopheles arabiensis',6:'Anopheles dirus',7:'Culex pipiens',8:'Anopheles minimus',9:'Anopheles stephensi'}

def main():
 C.mkdir(parents=True,exist_ok=True);O.mkdir(exist_ok=True)
 ids=(D/'biodcase/Development_data/metadata/Test_ids.txt').read_text().splitlines()
 ids=[n for n in ids if not '_D_5_' in n];rows=[];long=[];short=[];removed=[]
 frame=pd.read_csv(O/'segments.csv',low_memory=False);train=frame[frame.split.eq('train')]
 hashes={file_hash(ROOT/p) for p in train.path.unique()};seen=set()
 with zipfile.ZipFile(D/'Development_data.zip') as z:
  for count,name in enumerate(ids,1):
   raw=z.read('Development_data/raw_audio/'+name+'.wav');digest=hashlib.sha256(raw).hexdigest()
   if digest in hashes or digest in seen:removed.append(dict(id=name,reason='byte_identical_training_or_benchmark_duplicate'));continue
   seen.add(digest);w,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True);w=w.mean(axis=1);g=np.gcd(sr,16000)
   if sr!=16000:w=resample_poly(w,16000//g,sr//g)
   size=15872 if len(w)>=15872 else 4800
   if len(w)<size:removed.append(dict(id=name,reason='less_than_0.3s'));continue
   sid,domain=map(int,re.match(r'S_(\d+)_D_(\d+)_',name).groups())
   for start in range(0,len(w)-size+1,size):
    chunk=w[start:start+size];bucket='long' if size==15872 else 'short';dest=long if bucket=='long' else short
    rows.append(dict(path=name,label=MAP[sid],domain=f'biodcase_D{domain}',group=name,start_s=start/16000,step_s=size/16000,quality_ok=np.max(np.abs(chunk))>=32/32768 and np.mean(np.abs(chunk)>=32700/32768)<=.01,id=name,bucket=bucket,wave_index=len(dest)))
    dest.append(chunk.copy())
   if count%500==0:print('Decoded',count,'/',len(ids),flush=True)
 pd.DataFrame(rows).to_csv(O/'biodcase_segments.csv',index=False)
 pd.DataFrame(removed).to_csv(O/'biodcase_removed.csv',index=False)
 for bucket,clips in [('long',long),('short',short)]:np.save(C/f'{bucket}.npy',np.stack(clips))
 atomic_json(O/'biodcase_audit.json',dict(official_non_D5_test_files=len(ids),retained_files=len({r['path'] for r in rows}),long_windows=len(long),short_windows=len(short),removed_files=len(removed),exact_byte_duplicate_check=True,ancestry_or_individual_identity_not_fully_available=True,benchmark_not_claimed_independent_from_previous_corpora=True,D5_excluded_unverified_sensor_origin=True,not_used_for_training_calibration_or_selection=True,zip_md5_verified=True,no_audio_padding_repetition_or_concatenation=True))
 print('Auxiliary benchmark ready',len(rows),'windows',flush=True)
if __name__=='__main__':main()
