"""Download model-independent fresh controls AFTER the corrected model freeze."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,struct,zlib
import pandas as pd
import requests
import soundfile as sf
import fetch_humbug as fetch
from corrected_validation import ROOT,OUT,sha

D=ROOT/'data/external/corrected_fresh_2026_10_01'


def main():
    freeze=json.loads((OUT/'freeze.json').read_text());stamp=datetime.now(timezone.utc).isoformat()
    assert stamp>freeze['frozen_utc']
    D.mkdir(parents=True,exist_ok=True)
    if (OUT/'fresh_selection.json').exists():raise FileExistsError('Selection already fixed; do not replace difficult examples.')
    m=pd.read_csv(ROOT/'data/external/humbug/neurips_2021_zenodo_0_0_1.csv')
    m['group']=m[['country','record_datetime','name']].fillna('').agg('|'.join,axis=1)
    year=pd.to_datetime(m.record_datetime,dayfirst=True,format='mixed',errors='coerce').dt.year
    used=set(pd.read_csv(ROOT/'data/external/humbug/selected_metadata.csv').group)
    seen=set()
    for root in [ROOT/'tmp',ROOT/'data/external']:
        seen.update(int(p.stem) for p in root.rglob('*.wav') if p.stem.isdigit())
    used.update(m.loc[m.id.isin(seen),'group'])
    candidates=m[year.ge(2018)&m.sound_type.eq('mosquito')&m.length.ge(.992)&
                 m.species.isin(fetch.ALIASES)&~m.group.isin(used)].sort_values(['species','id']).drop_duplicates('group')
    positive=candidates[candidates.species.eq('ae aegypti')].head(3)
    negative=candidates[~candidates.species.eq('ae aegypti')].head(3)
    chosen=pd.concat([positive,negative]);assert not chosen.group.isin(used).any()
    folder='12enzpCQlpVEAevcasrk0zs5Lu0_qjSIQ'
    listing=requests.get('https://drive.google.com/embeddedfolderview?id='+folder+'#list',timeout=40)
    listing.raise_for_status()
    noises=sorted(re.findall(r'id="entry-([^"]+)".*?<div class="flip-entry-title">([^<]+)</div>',listing.text,re.S),key=lambda item:item[1])
    assert len(noises)==12 and all(name.endswith('.wav') for _,name in noises)
    plan=dict(selection_utc=stamp,frozen_utc=freeze['frozen_utc'],model_hashes=freeze['models_sha256'],
              rule='All 12 previously unused MosquitoSong environmental/fan controls; first three unused HumBug negative recording groups ordered by species/id, year>=2018. Up to three unused positive groups by same rule; report lack of candidates without relabelling or reusing a recording.',
              positive_groups_available=int(len(positive)),negative_groups=int(len(negative)),
              positive_limitation='No unused HumBug Aedes recording groups satisfying the duration rule remain; this fresh acoustic set cannot estimate Aedes sensitivity.' if not len(positive) else None,
              sources=['https://github.com/akaraspt/mosquitosongp','https://zenodo.org/records/4904800'],
              planned_noise_files=[dict(id=fid,name=name) for fid,name in noises],planned_humbug_ids=chosen.id.astype(int).tolist(),files=[])
    (OUT/'fresh_selection.json').write_text(json.dumps(plan,indent=2))
    known=set(pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False).content_sha256.dropna())
    for root in [ROOT/'data/external',ROOT/'tmp']:
        for p in root.rglob('*.wav'):
            if D not in p.parents:known.add(sha(p))
    def noise(item):
        fid,name=item;resp=requests.get('https://drive.google.com/uc?export=download&id='+fid,timeout=(20,90));resp.raise_for_status()
        raw=resp.content;h=hashlib.sha256(raw).hexdigest()
        if h in known:raise ValueError('Fresh file duplicates previously downloaded audio: '+name)
        p=D/name;p.write_bytes(raw);info=sf.info(p)
        return dict(path=str(p.relative_to(ROOT)),sha256=h,reference_aedes=False,reference_species='environmental_noise',
                    group='mosquitosong_noise|'+re.sub(r'^30cm_|_normalized_16bit\.wav$','',name),source_id=fid,
                    seconds=info.duration,source='MosquitoSong environmental/fan recording')
    with ThreadPoolExecutor(max_workers=2) as pool:plan['files']=list(pool.map(noise,noises))
    record=json.loads((ROOT/'data/external/humbug/record.json').read_text());index={}
    for archive in record['files']:
        if archive['key'].endswith('.zip'):
            for entry in fetch.entries(archive):index[entry['name']]=(archive,entry)
    def mosquito(row):
        archive,e=index[f'{row.id}.wav'];start=e['offset'];end=start+30+len(e['name'])+e['compressed']+1024
        raw_header=fetch.byte_range(archive['links']['self'],start,end);head=struct.unpack_from('<4s5H3I2H',raw_header)
        assert head[0]==b'PK\x03\x04';offset=30+head[-2]+head[-1]
        raw=zlib.decompress(raw_header[offset:offset+e['compressed']],-15)
        assert len(raw)==e['uncompressed'] and zlib.crc32(raw)==e['crc']
        h=hashlib.sha256(raw).hexdigest()
        if h in known:raise ValueError('Fresh mosquito is a previously downloaded copy; no substitution by prediction.')
        p=D/f'{row.id}.wav';p.write_bytes(raw);info=sf.info(p)
        return dict(path=str(p.relative_to(ROOT)),sha256=h,reference_aedes=row.species=='ae aegypti',
                    reference_species=fetch.ALIASES[row.species],group=row.group,id=int(row.id),seconds=info.duration,
                    source='HumBug',mic_type=row.mic_type,country=row.country)
    with ThreadPoolExecutor(max_workers=2) as pool:plan['files'].extend(pool.map(mosquito,chosen.itertuples(index=False)))
    plan['download_complete_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'fresh_selection.json').write_text(json.dumps(plan,indent=2))
    print('FRESH DOWNLOAD COMPLETE',len(plan['files']),'files',sum(x['seconds'] for x in plan['files']),'seconds',flush=True)


if __name__=='__main__':main()
