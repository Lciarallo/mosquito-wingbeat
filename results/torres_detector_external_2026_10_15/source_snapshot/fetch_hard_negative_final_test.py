"""Fixed post-round-three sample: new challenge WAVs and unused HumBug groups.

Challenge D3 files lack original individual/session mapping; their novelty is
file/PCM based, not proof of a new collection campaign or independent individuals.
Never use D5 optical signals to validate an acoustic microphone classifier.
"""
import json,struct,zlib,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests,pandas as pd,soundfile as sf
import fetch_humbug as fetch
from train_reproduction import ROOT,OUT,sha,now

D=ROOT/'data/external/reproduction_final_2026_10_01'
HARD=OUT/'hard_negative'


def central_prefix():
    marker=ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_prefix.json'
    if marker.exists():return json.loads(marker.read_text())
    f=json.loads((ROOT/'tmp/reproductions_2026_10_01/biodcase2026_record.json').read_text())['files'][0]
    tail=fetch.byte_range(f['links']['self'],f['size']-65536,f['size']-1)
    ix=tail.rfind(b'PK\x06\x06')
    if ix>=0:
        a=struct.unpack_from('<4sQ2H2I4Q',tail,ix);length,offset=a[8],a[9]
    else:
        a=struct.unpack_from('<4s4H2IH',tail,tail.rfind(b'PK\x05\x06'));length,offset=a[5],a[6]
    # Read only a bounded prefix, containing the class-1/D3 files. No 3.8GB download.
    block=fetch.byte_range(f['links']['self'],offset,offset+min(length,8*1024*1024)-1)
    pos=0;entries=[]
    while pos+46<len(block):
        a=struct.unpack_from('<4s6H3I5H2I',block,pos)
        if a[0]!=b'PK\x01\x02':break
        n,extra,comment=a[10:13]
        if pos+46+n+extra+comment>len(block):break
        name=block[pos+46:pos+46+n].decode()
        entries.append(dict(name=name,compressed=a[8],uncompressed=a[9],offset=a[16],method=a[4],crc=a[7]))
        pos+=46+n+extra+comment
    marker.write_text(json.dumps(dict(file=f,entries=entries,partial_central_directory=True)))
    print('Bounded ZIP64 metadata',len(entries),'entries',flush=True)
    return json.loads(marker.read_text())


def main():
    fr=json.loads((HARD/'freeze.json').read_text());D.mkdir(exist_ok=True,parents=True)
    path=HARD/'final_selection.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan.get('completed_utc'):raise FileExistsError('Fixed test already downloaded.')
    else:
        cd=central_prefix();entries=sorted([e for e in cd['entries'] if Path(e['name']).name.startswith('S_1_D_3_') and e['name'].endswith('.wav')],
                key=lambda e:int(Path(e['name']).stem.split('_')[-1]))
        assert len(entries)>=12
        m=pd.read_csv(ROOT/'data/external/humbug/neurips_2021_zenodo_0_0_1.csv');m['group']=m[['country','record_datetime','name']].fillna('').agg('|'.join,axis=1)
        used=set(pd.read_csv(ROOT/'data/external/humbug/selected_metadata.csv').group)
        seen={int(p.stem) for d in [ROOT/'data/external',ROOT/'tmp'] for p in d.rglob('*.wav') if p.stem.isdigit()}
        used.update(m.loc[m.id.isin(seen),'group']);year=pd.to_datetime(m.record_datetime,dayfirst=True,format='mixed',errors='coerce').dt.year
        allowed={'an arabiensis':'Anopheles arabiensis','an maculatus':'Anopheles maculatus','an leesoni':'Anopheles leesoni',
            'an pharoensis':'Anopheles pharoensis','an rivulorum':'Anopheles rivulorum','an ziemanni':'Anopheles ziemanni'}
        c=m[year.ge(2018)&m.length.ge(.992)&~m.group.isin(used)].sort_values('id').drop_duplicates('group')
        n=c[c.sound_type.eq('mosquito')&c.species.isin(allowed)].groupby('species',sort=True).head(3).copy();n['reference_species']=n.species.map(allowed)
        noise=c[c.sound_type.eq('background')].head(6).copy();noise['reference_species']='noise';n=pd.concat([n,noise])
        plan=dict(selection_utc=now(),frozen_utc=fr['frozen_utc'],model_hashes=fr['model_hashes'],
            rule='First twelve S_1_D_3 WAVs in numeric clip order; first three unused HumBug groups per six exact negative species, plus first six unused backgrounds, year>=2018, duration>=.992s; no score-based exclusions or substitutions.',
            planned_cdmsc=entries[:12],cdmsc_archive=cd['file'],planned_humbug=n.fillna('').to_dict('records'),files=[],
            source='https://zenodo.org/records/19095788',no_optical_D5=True,
            challenge_novelty_limitation='Original source-session and individual mapping is unavailable; report new files, not proven new individuals/campaign.',
            source_label_confound='Positives CD-MSC D3, negatives HumBug; report each source separately.')
        assert plan['selection_utc']>fr['frozen_utc'];path.write_text(json.dumps(plan,indent=2))
    known=set(pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False).content_sha256.dropna())
    for d in [ROOT/'data/external',ROOT/'tmp']:
        for p in d.rglob('*.wav'):
            if D not in p.parents:known.add(sha(p))
    def one(a,e,target):
        if not target.exists():
            raw=fetch.byte_range(a['links']['self'],e['offset'],e['offset']+30+len(e['name'])+e['compressed']+1024)
            head=struct.unpack_from('<4s5H3I2H',raw);assert head[0]==b'PK\x03\x04';start=30+head[-2]+head[-1]
            content=raw[start:start+e['compressed']];content=zlib.decompress(content,-15) if e['method']==8 else content
            assert len(content)==e['uncompressed'] and zlib.crc32(content)==e['crc'];target.write_bytes(content)
        assert sha(target) not in known,'Previously downloaded exact audio; never silently substitute another clip.'
        return sha(target),sf.info(target)
    def cd_download(e):
        target=D/Path(e['name']).name;h,info=one(plan['cdmsc_archive'],e,target)
        print('New positive WAV',target.name,info.duration,flush=True)
        return dict(path=str(target.relative_to(ROOT)),sha256=h,group='CDMSC2026_D3|'+target.stem,source='CDMSC2026_D3_new_files',
            reference_species='Aedes aegypti',reference_aedes=True,seconds=info.duration,original_entry=e['name'],crc32=e['crc'])
    with ThreadPoolExecutor(max_workers=2) as p:files=list(p.map(cd_download,plan['planned_cdmsc']))
    record=json.loads((ROOT/'data/external/humbug/record.json').read_text());index={}
    for a in record['files']:
        if a['key'].endswith('.zip'):
            for e in fetch.entries(a):index[e['name']]=(a,e)
    def hum_download(row):
        a,e=index[f"{row['id']}.wav"];target=D/f"{row['id']}.wav";h,info=one(a,e,target)
        print('New negative WAV',target.name,row['reference_species'],flush=True)
        return dict(path=str(target.relative_to(ROOT)),sha256=h,group=row['group'],source='HumBug_final_noise' if row['sound_type']=='background' else 'HumBug_final_other_species',
            reference_species=row['reference_species'],reference_aedes=False,seconds=info.duration,id=int(row['id']))
    with ThreadPoolExecutor(max_workers=2) as p:files.extend(p.map(hum_download,plan['planned_humbug']))
    plan['files']=files;plan['completed_utc']=now();path.write_text(json.dumps(plan,indent=2))
    print('FINAL DOWNLOAD COMPLETE',len(files),'files',sum(x['seconds'] for x in files),'seconds',flush=True)


if __name__=='__main__':main()
