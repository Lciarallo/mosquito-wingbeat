"""Expanded non-Aedes and noise acquisition; no model-based sample selection."""
import json,struct,zlib,hashlib,time
from concurrent.futures import ThreadPoolExecutor
import requests,pandas as pd,soundfile as sf,numpy as np
from train_reproduction import ROOT,now,sha
import fetch_humbug as fetch

OUT=ROOT/'results/reliability_2026_10_01';D=ROOT/'data/external/reliability_2026_10_01'


def canonical_non_aedes(species):
    # Genus-known Anopheles/Culex/etc. are valid NON-AEDES binary examples even
    # when exact species is unresolved. Unknown mosquito remains excluded.
    if not isinstance(species,str):return None
    genus={'an':'Anopheles','culex':'Culex','ma':'Mansonia','coquillettidia':'Coquillettidia'}
    tokens=species.lower().split()
    if len(tokens)<2 or tokens[0] not in genus:return None
    return genus[tokens[0]]+' '+' '.join(tokens[1:])


def main():
    OUT.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
    if (OUT/'freeze.json').exists():raise FileExistsError('Do not change frozen training data.')
    pp=OUT/'additional_selection.json'
    if pp.exists():
        plan=json.loads(pp.read_text())
        if plan.get('completed_utc'):raise FileExistsError('Data already acquired.')
    else:
        m=pd.read_csv(ROOT/'data/external/humbug/neurips_2021_zenodo_0_0_1.csv');m['group']=m[['country','record_datetime','name']].fillna('').agg('|'.join,axis=1)
        used=set(pd.read_csv(ROOT/'data/external/humbug/selected_metadata.csv').group)
        seen={int(p.stem) for root in [ROOT/'data/external',ROOT/'tmp'] for p in root.rglob('*.wav') if p.stem.isdigit()}
        used.update(m.loc[m.id.isin(seen),'group']);year=pd.to_datetime(m.record_datetime,dayfirst=True,format='mixed',errors='coerce').dt.year
        m['label']=m.species.map(canonical_non_aedes);m.loc[m.sound_type.eq('background'),'label']='noise'
        c=m[year.ge(2018)&m.length.ge(.3)&m.label.notna()&m.sound_type.isin(['mosquito','background'])&~m.group.isin(used)]
        c=c.sort_values(['length','id'],ascending=[False,True]).drop_duplicates('group')
        rng=np.random.default_rng(20261011);chosen=[]
        for label,part in c.groupby('label',sort=True):
            cap=20 if label=='noise' else 36
            part=part.sort_values('id').iloc[rng.permutation(len(part))[:cap]].copy();n=len(part)
            nt=min(24,max(1,int(n*.67))) if label!='noise' else min(12,n)
            nv=max(1,(n-nt)//2) if n>nt else 0
            part['split']=['train']*nt+['validation']*nv+['calibration']*(n-nt-nv)
            chosen.extend(part.fillna('').to_dict('records'))
        esc=pd.read_csv(ROOT/'tmp/reproductions_2026_10_01/esc50_metadata.csv')
        # All fragments of each original Freesound source stay in one partition.
        esc=esc[esc.category.ne('insects')].sort_values('filename');escchosen=[]
        for split,folds,cap in [('train',[1,2,3],4),('validation',[4],2)]:
            for category,part in esc[esc.fold.isin(folds)].groupby('category',sort=True):
                part=part.drop_duplicates('src_file').head(cap).copy();part['split']=split
                escchosen.extend(part.to_dict('records'))
        plan=dict(created_utc=now(),seed=20261011,rule='Random up to 36 unused HumBug original groups per known non-Aedes taxon, 24/6/6 train/validation/calibration; max20 unused backgrounds 12/4/4. ESC50 non-flying-insect categories: four source groups/category folds1-3 train, two/category fold4 validation; fold5 untouched.',
            taxonomy='Genus-known non-Aedes complexes are negative only; never report them as precisely identified species. Unresolved mosquito excluded.',
            planned_humbug=chosen,planned_esc50=escchosen,
            esc50_commit=(ROOT/'tmp/reproductions_2026_10_01/esc50_commit.txt').read_text().strip(),files=[],
            previous_feedback='All 30 previously evaluated HumBug negatives may become training. All previous CDMSC Aedes and private audio stay excluded from training.',
            prior_test_results_now_exploratory=True)
        pp.write_text(json.dumps(plan,indent=2))
    record=json.loads((ROOT/'data/external/humbug/record.json').read_text());index={}
    for a in record['files']:
        if a['key'].endswith('.zip'):
            for e in fetch.entries(a):index[e['name']]=(a,e)
    def hum(row):
        a,e=index[f"{row['id']}.wav"];path=D/f"{row['id']}.wav"
        if not path.exists():
            b=fetch.byte_range(a['links']['self'],e['offset'],e['offset']+30+len(e['name'])+e['compressed']+1024)
            h=struct.unpack_from('<4s5H3I2H',b);offset=30+h[-2]+h[-1];raw=zlib.decompress(b[offset:offset+e['compressed']],-15)
            assert len(raw)==e['uncompressed'] and zlib.crc32(raw)==e['crc'];path.write_bytes(raw)
        return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),label=row['label'],group='humbug:'+row['group'],
                    domain='humbug',split=row['split'],seconds=sf.info(path).duration,id=int(row['id']),source='HumBugDB',mic_type=row['mic_type'])
    def esc(row):
        path=D/row['filename'];url=f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{plan['esc50_commit']}/audio/{row['filename']}"
        if not path.exists():
            for attempt in range(3):
                try:
                    r=requests.get(url,timeout=(15,40));r.raise_for_status();path.write_bytes(r.content);sf.info(path);break
                except (requests.RequestException,RuntimeError):
                    if attempt==2:raise
                    time.sleep(2)
        return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),label='noise',group='ESC50|'+str(row['src_file']),
                    domain='esc50',split=row['split'],seconds=sf.info(path).duration,source='ESC50',category=row['category'],fold=row['fold'])
    files=[]
    for fn,rows in [(hum,plan['planned_humbug']),(esc,plan['planned_esc50'])]:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for item in pool.map(fn,rows):
                files.append(item)
                if len(files)%25==0:print('Additional verified',len(files),'of',len(plan['planned_humbug'])+len(plan['planned_esc50']),flush=True)
    f=pd.DataFrame(files)
    if f.groupby('group').split.nunique().max()!=1:raise ValueError('Group split overlap.')
    if f.groupby('sha256').split.nunique().max()!=1:raise ValueError('Duplicate bytes across partitions.')
    plan['files']=files;plan['completed_utc']=now();pp.write_text(json.dumps(plan,indent=2))
    print('ADDITIONAL DATA COMPLETE',len(files),f.groupby(['domain','split']).seconds.agg(['size','sum']).to_string(),flush=True)


if __name__=='__main__':main()
