"""Select/download unused positive and negative HumBug recordings AFTER binary freeze."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,struct,zlib
import pandas as pd
import fetch_humbug as fetch
from aedes_binary import ROOT

O=ROOT/'results/aedes_classical/fresh_external';D=ROOT/'data/external/humbug';T=ROOT/'tmp/aedes_classical_fresh'
def main():
    audit=json.loads((ROOT/'results/aedes_classical/training_audit.json').read_text());now=datetime.now(timezone.utc).isoformat();assert now>audit['frozen_utc']
    m=pd.read_csv(D/'neurips_2021_zenodo_0_0_1.csv');m['group']=m[['country','record_datetime','name']].fillna('').agg('|'.join,axis=1)
    excluded=set(pd.read_csv(D/'selected_metadata.csv').group)
    old=json.loads((ROOT/'results/multisource/fresh_external/selection.json').read_text());excluded.update(x['group'] for x in old['files'])
    # Exclude metadata identities of any earlier downloaded HumBug excerpts.
    seen=set()
    for parent in [ROOT/'tmp',D/'audio']:
        for p in parent.rglob('*.wav'):
            if p.stem.isdigit():seen.add(int(p.stem))
    excluded.update(m.loc[m.id.isin(seen),'group'])
    candidates=m[m.sound_type.eq('mosquito')&m.length.ge(.3)&~m.group.isin(excluded)&m.species.isin(fetch.ALIASES)]
    positive=candidates[candidates.species.eq('ae aegypti')].sort_values('id').drop_duplicates('group').head(2)
    negatives=candidates[~candidates.species.eq('ae aegypti')].sort_values(['species','id']).drop_duplicates('group').groupby('species',sort=True).head(1).head(3)
    chosen=pd.concat([positive,negatives]);assert len(positive)==2 and len(negatives)==3 and chosen.group.nunique()==5
    O.mkdir(parents=True,exist_ok=True);T.mkdir(exist_ok=True)
    protocol=dict(selection_utc=now,frozen_utc=audit['frozen_utc'],models_sha256={x['model']:x['sha256'] for x in audit['models']},source='https://zenodo.org/records/4904800',
        rule='First two positive groups ordered by ID, one negative per exact species ordered by species/ID, length >=0.3; exclude all groups of selected training/evaluation metadata and previously downloaded numeric HumBug WAV IDs. Only two unused positive source groups >=0.3s remained after excluding previous downloads; both selected before inference. Familiar campaign, not new domain.',files=[])
    (O/'selection.json').write_text(json.dumps(protocol,indent=2))
    record=json.loads((D/'record.json').read_text());index={}
    for item in record['files']:
        if item['key'].endswith('.zip'):
            for e in fetch.entries(item):index[e['name']]=(item,e)
    hashes=set(pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False).content_sha256.dropna())
    # Auxiliary benchmark originals may overlap HumBug; reject byte matches there too.
    for p in (ROOT/'data/external').rglob('*.wav'):
        hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    def one(row):
        item,e=index[f'{row.id}.wav'];start=e['offset'];end=start+30+len(e['name'])+e['compressed']+1024
        content=fetch.byte_range(item['links']['self'],start,end);header=struct.unpack_from('<4s5H3I2H',content);assert header[0]==b'PK\x03\x04'
        off=30+header[-2]+header[-1];raw=zlib.decompress(content[off:off+e['compressed']],-15)
        assert len(raw)==e['uncompressed'] and zlib.crc32(raw)==e['crc']
        sha=hashlib.sha256(raw).hexdigest();assert sha not in hashes,'Previously downloaded byte-identical audio; stop selection, do not choose by predictions.'
        p=T/f'{row.id}.wav';p.write_bytes(raw)
        return dict(id=int(row.id),group=row.group,reference_species=fetch.ALIASES[row.species],reference_aedes=bool(row.species=='ae aegypti'),path=str(p.relative_to(ROOT)),sha256=sha,mic_type=row.mic_type,country=row.country)
    with ThreadPoolExecutor(max_workers=2) as pool:protocol['files']=list(pool.map(one,chosen.itertuples(index=False)))
    (O/'selection.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
