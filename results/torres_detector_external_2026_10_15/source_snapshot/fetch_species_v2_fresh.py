"""Fixed post-freeze metadata selection; acquisition is never prediction-dependent."""
import json
import struct
import zlib
from pathlib import Path
import pandas as pd
import soundfile as sf

from fetch_humbug import byte_range
from evaluate_nested_species import pcm_keys
from improve_device import ROOT, atomic_json, file_hash
from nested_species_ensemble import now


def acquire():
    from nested_species_ensemble_v2 import OUT, verify_freeze, source_pcm_keys
    freeze=verify_freeze(OUT)
    evidence=OUT/'postfreeze_web_search.json'
    if not evidence.exists():raise ValueError('Perform/document post-freeze official-source web search first.')
    web=json.loads(evidence.read_text())
    if web['searched_utc']<=freeze['created_utc']:raise ValueError('Web search predates freeze.')
    planfile=OUT/'fresh_selection_v2.json'
    if planfile.exists():raise FileExistsError('Fresh selection already exists; no silent replacement.')
    centralpath=ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_full.json'
    central=json.loads(centralpath.read_text())
    dest=ROOT/'data/external/species_nested_fresh_2026_10_13';dest.mkdir(parents=True,exist_ok=False)
    old=sorted({p.resolve() for root in ['data/raw','data/external','data/noise','data/external_datasets','tmp','external']
                for p in (ROOT/root).rglob('*.wav') if dest not in p.parents})
    old.append(ROOT/'drive_audio.wav');names={p.name for p in old}
    old=sorted(set(old)|{ROOT/p for p in pd.read_csv(OUT/'segments_v2.csv').path.unique()})
    plan=dict(selected_utc=now(),frozen_utc=freeze['created_utc'],source='https://zenodo.org/records/19095788',
              rule='First 2 unused numeric D3 IDs each for exact species Aedes aegypti and albopictus, >=1 real second; scan up to 30; reject bytes or aligned PCM collisions before inference; no replacement after collision rejection.',
              known_files=len(old),central_directory_sha256=file_hash(centralpath),
              familiar_corpus_not_new_campaign=True,biological_session_identity_unavailable=True,
              candidates={},accepted_files=[],exclusions=[])
    for sid in [1,2]:
        pool=sorted([e for e in central['entries'] if Path(e['name']).name.startswith(f'S_{sid}_D_3_')
                     and e['name'].endswith('.wav') and Path(e['name']).name not in names],
                    key=lambda e:int(Path(e['name']).stem.split('_')[-1]))
        plan['candidates'][str(sid)]=pool[:30]
    atomic_json(planfile,plan)  # Commit candidates before any downloads/inference.
    bytehash=set();knownpcm=set()
    for i,path in enumerate(old):
        bytehash.add(file_hash(path))
        knownpcm.update(source_pcm_keys(path))
        if i%1500==0:print('KNOWN AUDIO PCM',i,'/',len(old),flush=True)
    plan['novelty_index']=dict(byte_hashes=len(bytehash),pcm_signatures=len(knownpcm),
        scope='Whole/aligned 1/.992/.3-second 8 kHz PCM raw and gain-normalised; not arbitrary-offset or biological ancestry proof.')
    chosen=[]
    for sid,species in [(1,'Aedes aegypti'),(2,'Aedes albopictus')]:
        n=0
        for e in plan['candidates'][str(sid)]:
            raw=byte_range(central['file']['links']['self'],e['offset'],
                           e['offset']+30+len(e['name'])+e['compressed']+1024)
            header=struct.unpack_from('<4s5H3I2H',raw)
            if header[0]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header.')
            offset=30+header[-2]+header[-1];data=raw[offset:offset+e['compressed']]
            if e['method']==8:data=zlib.decompress(data,-15)
            elif e['method']!=0:raise ValueError('Unsupported ZIP compression.')
            if len(data)!=e['uncompressed'] or zlib.crc32(data)!=e['crc']:raise ValueError('CRC mismatch.')
            path=dest/Path(e['name']).name;path.write_bytes(data)
            item=dict(path=str(path.relative_to(ROOT)),group='BioDCASE_D3|'+path.stem,
                      reference_species=species,seconds=sf.info(path).duration,sha256=file_hash(path),entry=e['name'])
            if item['seconds']<1:
                plan['exclusions'].append(dict(**item,reason='duration<1 s'));continue
            chosen.append(item);n+=1
            if n==2:break
    newpcm=set();newbytes=set();audit=[]
    for item in chosen:
        keys=pcm_keys(ROOT/item['path']);matches=len(keys&knownpcm);within=len(keys&newpcm)
        duplicate=item['sha256'] in bytehash|newbytes
        excluded=bool(matches or within or duplicate)
        audit.append(dict(**item,known_pcm_matches=matches,within_pcm_matches=within,byte_duplicate=duplicate,excluded=excluded))
        if excluded:plan['exclusions'].append(dict(**item,reason='PCM/byte collision'))
        else:
            plan['accepted_files'].append(item);newpcm.update(keys);newbytes.add(item['sha256'])
    plan['completed_utc']=now();atomic_json(planfile,plan)
    pd.DataFrame(audit).to_csv(OUT/'fresh_novelty_audit.csv',index=False)
    verify_freeze(OUT)
    print('POST-FREEZE NOVEL FILES',len(plan['accepted_files']),flush=True)


if __name__=='__main__':acquire()
