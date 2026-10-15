"""Select two unused positive clips by metadata after freeze; audit bytes/aligned PCM."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import struct
import zlib
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from fetch_humbug import byte_range
from nested_species_ensemble_v2 import source_pcm_keys
from reproduce_notebook_transfer import ROOT, OUT, digest, now, verify_freeze, audio_bag, combinations, predict_bundle, save

POST = ROOT/'results/notebook_transfer_postfreeze_2026_10_14_2044'


def index_file(path):
    try:
        return digest(path), source_pcm_keys(path), None
    except Exception as error:
        return digest(path), set(), dict(path=str(Path(path).relative_to(ROOT)), error=type(error).__name__)


def main():
    freeze=verify_freeze()
    if POST.exists():raise FileExistsError('Never replace a previous acquisition.')
    POST.mkdir();dest=POST/'audio';dest.mkdir()
    central_path=ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_full.json'
    central=json.loads(central_path.read_text())
    if central.get('partial_central_directory'):raise ValueError('Incomplete ZIP directory.')
    known=sorted({p.resolve() for folder in ['data/raw','data/external','data/noise','data/external_datasets','tmp','external','results'] for p in (ROOT/folder).rglob('*.wav')})
    names={p.name for p in known}
    candidates=sorted([e for e in central['entries'] if Path(e['name']).name.startswith('S_1_D_3_') and e['name'].endswith('.wav') and Path(e['name']).name not in names],key=lambda e:int(Path(e['name']).stem.split('_')[-1]))[:30]
    plan=dict(created_utc=now(),searched_utc=now(),official_source='https://zenodo.org/records/19095788',official_source_read_after_freeze=True,
              frozen_utc=freeze['created_utc'],freeze_sha256=digest(OUT/'freeze.json'),central_sha256=digest(central_path),
              rule='First two unused numeric D3 Ae. aegypti clips of >=1 s; no replacement after novelty exclusion.',
              familiar_corpus=True,new_campaign=False,biological_independence_proven=False,candidates=candidates,selected=[],accepted=[],excluded=[])
    save(POST/'selection.json',plan)
    known_hashes=set();known_pcm=set();errors=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(h,k,error) in enumerate(pool.map(index_file,known,chunksize=8,buffersize=16)):
            known_hashes.add(h);known_pcm.update(k)
            if error:errors.append(error)
            if i%1000==0:print('NOVELTY',i,'/',len(known),flush=True)
    save(POST/'novelty_index.json',dict(files=len(known),byte_hashes=len(known_hashes),pcm_keys=len(known_pcm),decode_errors=errors,
         limitations='Aligned whole/1/.992/.3 s 8 kHz raw and gain-normalised nonconstant PCM; arbitrary shifts and biological/session ancestry unknown.'))
    for e in candidates:
        raw=byte_range(central['file']['links']['self'],e['offset'],e['offset']+30+len(e['name'])+e['compressed']+1024)
        header=struct.unpack_from('<4s5H3I2H',raw)
        if header[0]!=b'PK\x03\x04':raise ValueError('Invalid ZIP header.')
        start=30+header[-2]+header[-1];data=raw[start:start+e['compressed']]
        if e['method']==8:data=zlib.decompress(data,-15)
        elif e['method']!=0:raise ValueError('Unsupported compression.')
        if len(data)!=e['uncompressed'] or zlib.crc32(data)!=e['crc']:raise ValueError('CRC mismatch.')
        path=dest/Path(e['name']).name;path.write_bytes(data)
        if sf.info(path).duration<1:continue
        item=dict(path=str(path.relative_to(ROOT)),group='BioDCASE_D3|'+path.stem,reference_species='Aedes aegypti',seconds=sf.info(path).duration,sha256=digest(path),zip_crc32=e['crc'])
        plan['selected'].append(item)
        if len(plan['selected'])==2:break
    for item in plan['selected']:
        keys=source_pcm_keys(ROOT/item['path']);collision=len(keys&known_pcm);duplicate=item['sha256'] in known_hashes
        item.update(pcm_matches=collision,byte_duplicate=duplicate)
        if collision or duplicate:plan['excluded'].append(item)
        else:
            plan['accepted'].append(item);known_pcm.update(keys);known_hashes.add(item['sha256'])
    save(POST/'selection_complete.json',dict(**plan,completed_utc=now()))
    rows=[]
    for item in plan['accepted']:
        bags,arrays=audio_bag(ROOT/item['path'])
        for task,variant,family in combinations():
            bundle=joblib.load(OUT/'models'/f'final_{task}_{variant}_{family}.joblib');s=bundle['settings'];p=predict_bundle(bundle,np.array([bags[variant]]))[0];classes=np.array(s['classes'])
            rows.append(dict(scope='postfreeze_transfer2',task=task,variant=variant,family=family,group=item['group'],pcm_block=item['group'],source_label='Aedes aegypti',true_label='Aedes aegypti',predicted_label=classes[p.argmax()],confidence=float(p.max()),accepted=bool(p.max()>=s['rejection_threshold']),aegypti_score=float(p[np.searchsorted(classes,'Aedes aegypti')])))
    pd.DataFrame(rows).to_csv(POST/'predictions.csv',index=False);verify_freeze()
    print('ACQUIRED',len(plan['accepted']),'/',len(plan['selected']),'WITHOUT RETUNING',flush=True)


if __name__=='__main__':main()
