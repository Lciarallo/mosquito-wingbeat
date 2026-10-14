"""After freeze, select five previously unused Aedes source recordings by metadata."""
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd,json,hashlib,struct,zlib
from concurrent.futures import ThreadPoolExecutor
import fetch_humbug as fetch
ROOT=Path(__file__).resolve().parent;O=ROOT/'results/multisource/fresh_external';D=ROOT/'data/external/humbug';T=ROOT/'tmp/multisource_fresh'
def main():
 O.mkdir(parents=True,exist_ok=True);T.mkdir(exist_ok=True)
 a=json.loads((ROOT/'results/multisource/training_audit.json').read_text());now=datetime.now(timezone.utc).isoformat();assert now>a['frozen_utc']
 m=pd.read_csv(D/'neurips_2021_zenodo_0_0_1.csv');m['group']=m[['country','record_datetime','name']].fillna('').agg('|'.join,axis=1)
 selected=pd.read_csv(D/'selected_metadata.csv');year=pd.to_datetime(m.record_datetime,dayfirst=True,format='mixed',errors='coerce').dt.year
 candidates=m[m.species.eq('ae aegypti') & m.sound_type.eq('mosquito') & m.length.ge(2.976) & year.ge(2018) & ~m.group.isin(selected.group)]
 chosen=candidates.sort_values('id').drop_duplicates('group').head(5);assert len(chosen)==5
 record=json.loads((D/'record.json').read_text());index={}
 for item in record['files']:
  if item['key'].endswith('.zip'):
   for entry in fetch.entries(item):index[entry['name']]=(item,entry)
 protocol=dict(selection_utc=now,frozen_utc=a['frozen_utc'],model_sha256=a['checkpoint_sha256'],source='https://zenodo.org/records/4904800',rule='First five distinct Aedes aegypti source groups ordered by excerpt ID, >=2.976s, recorded >=2018, excluding all previously selected 307 excerpts and their original source groups. Chosen by metadata before inference.',files=[])
 (O/'selection.json').write_text(json.dumps(protocol,indent=2))
 def one(row):
  item,e=index[f'{row.id}.wav'];start=e['offset'];end=start+30+len(e['name'])+e['compressed']+1024
  content=fetch.byte_range(item['links']['self'],start,end);h=struct.unpack_from('<4s5H3I2H',content);assert h[0]==b'PK\x03\x04';offset=30+h[-2]+h[-1]
  raw=zlib.decompress(content[offset:offset+e['compressed']],-15);assert len(raw)==e['uncompressed'] and zlib.crc32(raw)==e['crc']
  digest=hashlib.sha256(raw).hexdigest();old=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);assert digest not in set(old.content_sha256.dropna())
  p=T/f'{row.id}.wav';p.write_bytes(raw)
  return dict(id=int(row.id),group=row.group,reference_species='Aedes aegypti',length_s=row.length,gender=None if pd.isna(row.gender) else row.gender,method=None if pd.isna(row.method) else row.method,mic_type=None if pd.isna(row.mic_type) else row.mic_type,path=str(p.relative_to(ROOT)),sha256=digest,archive=item['key'],crc32=e['crc'])
 with ThreadPoolExecutor(max_workers=2) as pool:protocol['files']=list(pool.map(one,chosen.itertuples(index=False)))
 (O/'selection.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
