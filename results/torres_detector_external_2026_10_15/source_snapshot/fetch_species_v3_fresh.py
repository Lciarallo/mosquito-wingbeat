"""Post-freeze acquisition of genuinely unused clips, separate from reused fresh v2."""
import json
from pathlib import Path
import struct
import subprocess
import zlib
import pandas as pd
import soundfile as sf
from fetch_humbug import byte_range
from evaluate_nested_species import pcm_keys
from improve_device import ROOT,file_hash,atomic_json
from nested_species_ensemble import now


def acquire():
    from nested_species_ensemble_v3 import OUT,V2,verify_freeze
    from nested_species_ensemble_v2 import source_pcm_keys
    freeze=verify_freeze()
    web=json.loads((OUT/'postfreeze_web_search.json').read_text())
    if web['searched_utc']<=freeze['created_utc']:raise ValueError('Official web search must follow freeze.')
    marker=OUT/'fresh_selection_v3.json'
    if marker.exists():raise FileExistsError('Do not silently replace a post-freeze selection.')
    centralpath=ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_full.json'
    protocol=json.loads((OUT/'protocol.json').read_text())
    if file_hash(centralpath)!=protocol['input_sha256'][str(centralpath.relative_to(ROOT))]:
        raise ValueError('Predeclared archive index changed.')
    central=json.loads(centralpath.read_text())
    if central.get('partial_central_directory'):raise ValueError('Archive index is incomplete.')
    dest=OUT/'postfreeze_audio';dest.mkdir(exist_ok=False)
    known=sorted({p.resolve() for root in ['data/raw','data/external','data/noise','data/external_datasets','tmp','external']
                  for p in (ROOT/root).rglob('*.wav')}|{ROOT/p for p in pd.read_csv(V2/'segments_v2.csv').path.unique()}|{ROOT/'drive_audio.wav'})
    names={p.name for p in known}
    pool=sorted([e for e in central['entries'] if Path(e['name']).name.startswith('S_1_D_3_')
                 and e['name'].endswith('.wav') and Path(e['name']).name not in names],
                key=lambda e:int(Path(e['name']).stem.split('_')[-1]))[:30]
    plan=dict(selected_utc=now(),frozen_utc=freeze['created_utc'],source='https://zenodo.org/records/19095788',
              source_code_sha256=file_hash(Path(__file__)),central_directory_sha256=file_hash(centralpath),
              rule='First 2 unused numeric D3 Aedes aegypti clips >=1 real second; scan <=30; no replacement after byte/PCM collision exclusion.',
              candidates=pool,accepted_files=[],exclusions=[],known_files=len(known),
              familiar_corpus_not_new_campaign=True,biological_session_identity_unavailable=True)
    atomic_json(marker,plan)
    bytehash=set();knownpcm=set();errors=[]
    for i,path in enumerate(known):
        bytehash.add(file_hash(path))
        try:knownpcm.update(source_pcm_keys(path))
        except (sf.LibsndfileError,ValueError,subprocess.CalledProcessError) as e:
            errors.append(dict(path=str(path.relative_to(ROOT)),error=str(e)))
        if i%1500==0:print('V3 KNOWN PCM',i,'/',len(known),flush=True)
    plan['novelty_index']=dict(byte_hashes=len(bytehash),pcm_signatures=len(knownpcm),decoding_errors=errors,
        limitation='Whole/aligned 1/.992/.3-second 8 kHz raw and gain-normalised nonconstant PCM; not arbitrary offsets or biological ancestry proof.')
    chosen=[]
    for e in pool:
        raw=byte_range(central['file']['links']['self'],e['offset'],e['offset']+30+len(e['name'])+e['compressed']+1024)
        header=struct.unpack_from('<4s5H3I2H',raw)
        if header[0]!=b'PK\x03\x04':raise ValueError('Invalid ZIP header.')
        start=30+header[-2]+header[-1];data=raw[start:start+e['compressed']]
        if e['method']==8:data=zlib.decompress(data,-15)
        elif e['method']!=0:raise ValueError('Unsupported ZIP method.')
        if len(data)!=e['uncompressed'] or zlib.crc32(data)!=e['crc']:raise ValueError('ZIP CRC mismatch.')
        path=dest/Path(e['name']).name;path.write_bytes(data)
        item=dict(path=str(path.relative_to(ROOT)),group='BioDCASE_D3|'+path.stem,
                  reference_species='Aedes aegypti',seconds=sf.info(path).duration,sha256=file_hash(path),entry=e['name'])
        if item['seconds']<1:
            plan['exclusions'].append(dict(**item,reason='duration<1 second'));continue
        chosen.append(item)
        if len(chosen)==2:break
    seenpcm=set();seenbytes=set();audit=[]
    for item in chosen:
        keys=pcm_keys(ROOT/item['path']);collision=len(keys&knownpcm);within=len(keys&seenpcm)
        duplicate=item['sha256'] in bytehash|seenbytes
        excluded=bool(collision or within or duplicate)
        audit.append(dict(**item,known_pcm_matches=collision,within_pcm_matches=within,byte_duplicate=duplicate,excluded=excluded))
        if excluded:plan['exclusions'].append(dict(**item,reason='Known/within byte or PCM collision'))
        else:
            plan['accepted_files'].append(item);seenpcm.update(keys);seenbytes.add(item['sha256'])
    plan['completed_utc']=now();atomic_json(marker,plan)
    pd.DataFrame(audit).to_csv(OUT/'fresh_novelty_audit.csv',index=False)
    verify_freeze();print('V3 POSTFREEZE UNUSED',len(plan['accepted_files']),flush=True)


if __name__=='__main__':acquire()
