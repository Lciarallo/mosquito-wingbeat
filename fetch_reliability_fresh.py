"""Post-freeze acquisition, selected without predictions or private audio.

CD-MSC files are new files, not demonstrably new individuals or campaigns.
ESC-50 fold 5 has original sources distinct from the training/validation folds.
"""
import json,struct,zlib,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests,pandas as pd,soundfile as sf
from train_reproduction import ROOT,now,sha
from fetch_reliability_data import OUT,canonical_non_aedes
import fetch_humbug as fetch

D=ROOT/'data/external/reliability_fresh_2026_10_01'


def main():
    freeze=json.loads((OUT/'freeze.json').read_text())
    for path,h in freeze['models_sha256'].items():assert sha(ROOT/path)==h
    D.mkdir(exist_ok=True,parents=True);pp=OUT/'fresh_selection.json'
    if pp.exists():
        plan=json.loads(pp.read_text())
        if plan.get('completed_utc'):raise FileExistsError('Completed test cannot be reselected.')
    else:
        index=ROOT/'tmp/reproductions_2026_10_01/cdmsc_central_full.json'
        if not index.exists():raise FileNotFoundError('Complete archive metadata required; do not select on a partial list.')
        cd=json.loads(index.read_text());assert not cd['partial_central_directory']
        seen_names={p.name for root in [ROOT/'data/external',ROOT/'tmp'] for p in root.rglob('*.wav') if D not in p.parents}
        positive=sorted([e for e in cd['entries'] if Path(e['name']).name.startswith('S_1_D_3_') and e['name'].endswith('.wav') and Path(e['name']).name not in seen_names],key=lambda e:int(Path(e['name']).stem.split('_')[-1]))[:24]
        for e in positive:e.update(reference_species='Aedes aegypti',reference_aedes=True)
        # Official class IDs, verified in BioDCASE 2026 Table I. Include negatives
        # in the SAME acoustic domain to reduce label/source confounding.
        cdnegative=[]
        for species_id,taxon in [(2,'Aedes albopictus'),(7,'Culex pipiens'),(8,'Anopheles minimus'),(9,'Anopheles stephensi')]:
            entries=sorted([e for e in cd['entries'] if Path(e['name']).name.startswith(f'S_{species_id}_D_3_') and e['name'].endswith('.wav') and Path(e['name']).name not in seen_names],key=lambda e:int(Path(e['name']).stem.split('_')[-1]))[:6]
            for e in entries:e.update(reference_species=taxon,reference_aedes=False)
            cdnegative.extend(entries)
        m=pd.read_csv(ROOT/'data/external/humbug/neurips_2021_zenodo_0_0_1.csv');m['group']=m[['country','record_datetime','name']].fillna('').agg('|'.join,axis=1)
        used=set(pd.read_csv(ROOT/'data/external/humbug/selected_metadata.csv').group)
        ids={int(Path(n).stem) for n in seen_names if Path(n).stem.isdigit()};used.update(m.loc[m.id.isin(ids),'group'])
        year=pd.to_datetime(m.record_datetime,dayfirst=True,format='mixed',errors='coerce').dt.year
        m['reference_species']=m.species.map(canonical_non_aedes);m.loc[m.sound_type.eq('background'),'reference_species']='noise'
        candidate=m[year.ge(2018)&m.length.ge(.992)&m.reference_species.notna()&~m.group.isin(used)&m.sound_type.isin(['mosquito','background'])]
        candidate=candidate.sort_values('id').drop_duplicates('group')
        chosen=candidate.groupby('reference_species',sort=True).head(3)
        extra=json.loads((OUT/'additional_selection.json').read_text());esc=pd.read_csv(ROOT/'tmp/reproductions_2026_10_01/esc50_metadata.csv')
        esc=esc[esc.fold.eq(5)&esc.category.ne('insects')].sort_values('filename').groupby('category',sort=True).apply(lambda x:x.drop_duplicates('src_file').head(2),include_groups=False).reset_index()
        plan=dict(selection_utc=now(),frozen_utc=freeze['frozen_utc'],models_sha256=freeze['models_sha256'],fresh_protocol_sha256=sha(OUT/'fresh_protocol.json'),
            rule='First 24 unused acoustic class-1/domain-3 WAVs in full archive numeric order, plus first6/domain3 each of classes2,7,8,9 as SAME-domain negatives; first 3 unused HumBug recording groups per known non-Aedes taxon and noise; ESC50 fold5 first2 distinct original sources/category excluding flying insects.',
            cdmsc_archive=cd['file'],planned_cdmsc=positive+cdnegative,planned_humbug=chosen.fillna('').to_dict('records'),planned_esc50=esc.to_dict('records'),
            esc50_commit=extra['esc50_commit'],files=[],
            limitations=['CDMSC original individual/session mapping unavailable; file/PCM novelty only.','HumBug negatives have no fresh positives; report SAME-domain CDMSC subset separately.','Whole-file mosquito labels do not annotate wingbeat presence at each moment.'],
            target_sensor_independent_negative_recordings=False,no_optical_D5=True)
        assert positive and plan['selection_utc']>freeze['frozen_utc'];pp.write_text(json.dumps(plan,indent=2))
    known={sha(p) for root in [ROOT/'data/external',ROOT/'tmp'] for p in root.rglob('*.wav') if D not in p.parents}
    def extract(a,e,target):
        if not target.exists():
            raw=fetch.byte_range(a['links']['self'],e['offset'],e['offset']+30+len(e['name'])+e['compressed']+1024)
            h=struct.unpack_from('<4s5H3I2H',raw);assert h[0]==b'PK\x03\x04';start=30+h[-2]+h[-1]
            raw=raw[start:start+e['compressed']];raw=zlib.decompress(raw,-15) if e['method']==8 else raw
            assert len(raw)==e['uncompressed'] and zlib.crc32(raw)==e['crc'];target.write_bytes(raw)
        assert sha(target) not in known,'Duplicate audio; no silent replacement allowed.'
        return sha(target),sf.info(target).duration
    def positive(e):
        target=D/Path(e['name']).name;h,sec=extract(plan['cdmsc_archive'],e,target)
        return dict(path=str(target.relative_to(ROOT)),sha256=h,seconds=sec,group='CDMSC2026_D3|'+target.stem,source='CDMSC2026_D3',reference_aedes=e['reference_aedes'],reference_species=e['reference_species'],original_entry=e['name'],crc32=e['crc'])
    record=json.loads((ROOT/'data/external/humbug/record.json').read_text());index={}
    for a in record['files']:
        if a['key'].endswith('.zip'):
            for e in fetch.entries(a):index[e['name']]=(a,e)
    def hum(row):
        a,e=index[f"{row['id']}.wav"];target=D/f"{row['id']}.wav";h,sec=extract(a,e,target)
        return dict(path=str(target.relative_to(ROOT)),sha256=h,seconds=sec,group='humbug:'+row['group'],source='HumBug_noise' if row['sound_type']=='background' else 'HumBug_other_mosquito',reference_aedes=False,reference_species=row['reference_species'],id=int(row['id']))
    def esc(row):
        target=D/row['filename'];url=f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{plan['esc50_commit']}/audio/{row['filename']}"
        if not target.exists():
            for attempt in range(3):
                try:
                    r=requests.get(url,timeout=(15,40));r.raise_for_status();target.write_bytes(r.content);sf.info(target);break
                except (requests.RequestException,RuntimeError):
                    if attempt==2:raise
                    time.sleep(2)
        h=sha(target);assert h not in known
        return dict(path=str(target.relative_to(ROOT)),sha256=h,seconds=sf.info(target).duration,group='ESC50|'+str(row['src_file']),source='ESC50_noise',reference_aedes=False,reference_species='noise',category=row['category'],fold=5)
    files=[]
    for fn,items in [(positive,plan['planned_cdmsc']),(hum,plan['planned_humbug']),(esc,plan['planned_esc50'])]:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for item in pool.map(fn,items):
                files.append(item)
                if len(files)%20==0:print('Fresh verified',len(files),flush=True)
    plan['files']=files;plan['completed_utc']=now();pp.write_text(json.dumps(plan,indent=2))
    print('FRESH COMPLETE',len(files),'files',flush=True)


if __name__=='__main__':main()
