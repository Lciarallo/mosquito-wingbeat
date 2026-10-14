"""Duration-respecting extended corpus; no private or held-out Aedes adaptation."""
import json
import numpy as np,pandas as pd
from scipy.signal import resample_poly
from train_reproduction import ROOT,source_data,wave8_cache,sha,now
from corrected_validation import assert_disjoint,audio_windows
from aedes_classical import features
from bioacoustic_invariance import extract_batch_bioacoustic_features
from fetch_reliability_data import OUT

CACHE=ROOT/'data/cache/reliability_2026_10_01'


def main():
    if (OUT/'freeze.json').exists():raise FileExistsError('Frozen data cannot be rebuilt.')
    CACHE.mkdir(parents=True,exist_ok=True);f,w=source_data();base=wave8_cache(w)
    plan=json.loads((OUT/'additional_selection.json').read_text());assert plan.get('completed_utc')
    files=list(plan['files'])
    for p in [ROOT/'results/reproduction_2026_10_01/fresh_selection.json',ROOT/'results/reproduction_2026_10_01/hard_negative/final_selection.json']:
        old=json.loads(p.read_text())
        for item in old['files']:
            if not item['reference_aedes']:
                files.append(dict(**item,label=item['reference_species'],domain='humbug',split='train'))
    seen=set(f.content_sha256.dropna());add={b:[] for b in base};rows=[];feat=[];duplicates=[]
    used_groups=set(f.group);path_groups={}
    for item in files:
        p=ROOT/item['path'];assert sha(p)==item['sha256']
        if item['sha256'] in seen:
            duplicates.append(dict(path=item['path'],reason='Exact previous bytes, excluded before training'));continue
        seen.add(item['sha256']);wave,q,size=audio_windows(p);bucket='long' if size==15872 else 'short'
        group=item['group']
        if item['source'].startswith('HumBug') and not group.startswith('humbug:'):group='humbug:'+group
        if group in used_groups:raise ValueError('New original group already in base partition: '+group)
        used_groups.add(group)
        xx=np.concatenate([features(wave),extract_batch_bioacoustic_features(wave)],1);feat.append(xx)
        for i,(a,quality) in enumerate(zip(wave,q)):
            ix=len(base[bucket])+len(add[bucket]);add[bucket].append(resample_poly(a,1,2).astype('float32'))
            rows.append(dict(path=item['path'],label=item['label'],group=group,domain=item['domain'],split=item['split'],
                quality_ok=bool(quality),bucket=bucket,wave_index=ix,start_s=i*size/16000,content_sha256=item['sha256'],step_s=size/16000))
        if len(feat)%25==0:print('Prepared files',len(feat),flush=True)
    combined=pd.concat([f,pd.DataFrame(rows)],ignore_index=True);assert_disjoint(combined)
    for b,x in base.items():
        dest=np.lib.format.open_memmap(CACHE/(b+'_8k.npy'),mode='w+',dtype='float32',shape=(len(x)+len(add[b]),x.shape[1]))
        dest[:len(x)]=x
        if add[b]:dest[len(x):]=np.stack(add[b])
        dest.flush()
    x=np.concatenate([np.load(ROOT/'data/cache/aedes_classical_features.npy'),np.load(ROOT/'data/cache/bioacoustic_invariance_features.npy')],1)
    np.save(CACHE/'features.npy',np.concatenate([x,np.concatenate(feat)],0).astype('float32'))
    combined.to_csv(OUT/'segments.csv',index=False);pd.DataFrame(duplicates).to_csv(OUT/'excluded_exact_duplicates.csv',index=False)
    combined.groupby(['domain','label','split']).agg(windows=('path','size'),files=('path','nunique'),groups=('group','nunique')).to_csv(OUT/'counts.csv')
    audit=dict(completed_utc=now(),original_windows=len(f),added_windows=len(rows),added_files=len(feat),
        source_sha256=sha(OUT/'additional_selection.json'),segments_sha256=sha(OUT/'segments.csv'),
        group_and_byte_partition_overlap=0,feedback_training_only=True,private_in_training=False,
        all_cdmsc_aedes_excluded_from_training=True,arduino_compiled=False)
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2));print('EXTENDED CORPUS',json.dumps(audit),flush=True)


if __name__=='__main__':main()
