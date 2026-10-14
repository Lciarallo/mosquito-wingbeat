"""Post-freeze source selection; do not discard examples using predictions.

Figure 7 contains natural recordings, whereas Figure S7/S8 contain speaker
stimuli and are excluded. Include every >=.3s WAV from Figure 7, all genotypes.
These are positive recording-presence references, not time-resolved flight labels.
"""
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,struct,zlib,zipfile
import requests,pandas as pd,soundfile as sf
import fetch_humbug as fetch
from train_reproduction import ROOT,OUT,now,sha
from evaluate_reproduction import verify_freeze

D=ROOT/'data/external/reproduction_fresh_2026_10_01'


def main():
    freeze=verify_freeze();D.mkdir(parents=True,exist_ok=True)
    planpath=OUT/'fresh_selection.json'
    if planpath.exists():
        plan=json.loads(planpath.read_text())
        assert plan['model_hashes']==freeze['models_sha256']
        if plan.get('download_complete_utc'):raise FileExistsError('Already completed: evaluate the same fixed selection.')
    else:
        m=pd.read_csv(ROOT/'data/external/humbug/neurips_2021_zenodo_0_0_1.csv')
        m['group']=m[['country','record_datetime','name']].fillna('').agg('|'.join,axis=1)
        used=set(pd.read_csv(ROOT/'data/external/humbug/selected_metadata.csv').group)
        seen={int(p.stem) for root in [ROOT/'data/external',ROOT/'tmp'] for p in root.rglob('*.wav') if p.stem.isdigit()}
        used.update(m.loc[m.id.isin(seen),'group']);year=pd.to_datetime(m.record_datetime,dayfirst=True,format='mixed',errors='coerce').dt.year
        allowed={'an arabiensis':'Anopheles arabiensis','an maculatus':'Anopheles maculatus',
                 'an leesoni':'Anopheles leesoni','an pharoensis':'Anopheles pharoensis',
                 'an rivulorum':'Anopheles rivulorum','an ziemanni':'Anopheles ziemanni'}
        c=m[year.ge(2018)&m.length.ge(.992)&~m.group.isin(used)].sort_values('id').drop_duplicates('group')
        chosen=c[c.sound_type.eq('mosquito')&c.species.isin(allowed)].groupby('species',sort=True).head(3).copy()
        chosen['reference_species']=chosen.species.map(allowed)
        noise=c[c.sound_type.eq('background')].head(6).copy();noise['reference_species']='noise'
        chosen=pd.concat([chosen,noise]);assert not chosen.group.isin(used).any()
        plan=dict(selection_utc=now(),frozen_utc=freeze['frozen_utc'],model_hashes=freeze['models_sha256'],files=[],
             rule='All WAV files >=0.3s in PNAS Figure_7.zip, all four pair genotypes; first three unused groups per six explicitly species-labelled HumBug negative species, plus first six unused background groups, ordered by id; year>=2018, length>=0.992s. No prediction-based exclusion.',
             pnas='https://datadryad.org/dataset/doi:10.5061/dryad.qz612jmrb',
             archive_expected_sha256='9f98b6abd780ef4e1a37b4af011b8b7beeef65b18d0c4aad143ea85c61a0e1de',
             humbug='https://zenodo.org/records/4904800',planned_humbug=chosen.fillna('').to_dict('records'),
             pnas_reference='Known species of recorded mosquito pairs; no per-window flight labels. Mutants included, genotype reported.',
             source_confounding='Fresh positives and negatives come from different campaigns; report separately and never claim universal balanced accuracy.')
        planpath.write_text(json.dumps(plan,indent=2));assert plan['selection_utc']>freeze['frozen_utc']
    # Fingerprint every previous WAV and all supervised source files.
    known=set(pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False).content_sha256.dropna())
    for root in [ROOT/'tmp',ROOT/'data/external']:
        for p in root.rglob('*.wav'):
            if D not in p.parents:known.add(sha(p))
    files=[];archive=D/'Figure_7.zip'
    try:
        if not archive.exists():
            url='https://datadryad.org/downloads/file_stream/3544389'
            with requests.get(url,stream=True,timeout=(20,60)) as resp:
                resp.raise_for_status()
                with archive.with_suffix('.part').open('wb') as stream:
                    for block in resp.iter_content(4*1024*1024):stream.write(block)
            archive.with_suffix('.part').replace(archive)
        assert sha(archive)==plan['archive_expected_sha256']
        with zipfile.ZipFile(archive) as z:
            names=sorted(n for n in z.namelist() if n.lower().endswith('.wav') and not n.startswith('__MACOSX/'))
            plan['pnas_all_wav_entries']=names;planpath.write_text(json.dumps(plan,indent=2))
            print('PNAS WAV inventory',len(names),names[:12],flush=True)
            for name in names:
                # Preserve original hierarchy; reject path traversal.
                path=D/'pnas'/name
                if not path.resolve().is_relative_to((D/'pnas').resolve()):raise ValueError('Unsafe archive path.')
                path.parent.mkdir(parents=True,exist_ok=True);raw=z.read(name);h=hashlib.sha256(raw).hexdigest()
                if h in known:raise ValueError('Novel source is an exact old recording: '+name)
                path.write_bytes(raw);info=sf.info(path)
                if info.duration<.3:
                    plan.setdefault('excluded_by_duration',[]).append(name);continue
                files.append(dict(path=str(path.relative_to(ROOT)),sha256=h,reference_aedes=True,
                     reference_species='Aedes aegypti',group='PNAS2024|'+name,original_entry=name,
                     seconds=info.duration,source='PNAS2024_pairs',reference_scope='recording-presence, no event labels'))
        plan['pnas_download_status']='complete, published ZIP SHA256 verified'
    except requests.HTTPError as error:
        plan['pnas_download_status']='HTTP download unavailable: '+str(error)
        print(plan['pnas_download_status'],flush=True)
    record=json.loads((ROOT/'data/external/humbug/record.json').read_text());index={}
    for a in record['files']:
        if a['key'].endswith('.zip'):
            for e in fetch.entries(a):index[e['name']]=(a,e)
    def download(row):
        a,e=index[f"{row['id']}.wav"];path=D/f"{row['id']}.wav"
        if not path.exists():
            raw_header=fetch.byte_range(a['links']['self'],e['offset'],e['offset']+30+len(e['name'])+e['compressed']+1024)
            head=struct.unpack_from('<4s5H3I2H',raw_header);assert head[0]==b'PK\x03\x04'
            offset=30+head[-2]+head[-1];raw=zlib.decompress(raw_header[offset:offset+e['compressed']],-15)
            assert len(raw)==e['uncompressed'] and zlib.crc32(raw)==e['crc'];path.write_bytes(raw)
        h=sha(path)
        if h in known:raise ValueError('Fresh negative duplicates earlier WAV.')
        print('NEW negative',row['id'],row['reference_species'],flush=True)
        return dict(path=str(path.relative_to(ROOT)),sha256=h,reference_aedes=False,
             reference_species=row['reference_species'],group=row['group'],source='HumBug_fresh_noise' if row['sound_type']=='background' else 'HumBug_fresh_other_species',
             seconds=sf.info(path).duration,id=int(row['id']),mic_type=row['mic_type'])
    with ThreadPoolExecutor(max_workers=2) as pool:files.extend(pool.map(download,plan['planned_humbug']))
    plan['files']=files;plan['download_complete_utc']=now();planpath.write_text(json.dumps(plan,indent=2))
    print('NEW CORPUS',len(files),sum(x['seconds'] for x in files),'seconds',flush=True);verify_freeze()


if __name__=='__main__':main()
