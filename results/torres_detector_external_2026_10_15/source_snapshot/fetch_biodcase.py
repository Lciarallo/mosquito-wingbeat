from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,hashlib,time,zipfile,json
root=Path(__file__).resolve().parent/'data/external/additional';parts=root/'biodcase_parts';parts.mkdir(exist_ok=True)
url='https://zenodo.org/records/20478577/files/Development_data.zip?download=1';size=3799065219;chunk=64*1024*1024
jobs=[(i,start,min(size-1,start+chunk-1)) for i,start in enumerate(range(0,size,chunk))]
def one(job):
 i,a,b=job;p=parts/f'{i:03}.bin'
 if p.exists() and p.stat().st_size==b-a+1:return i
 for attempt in range(4):
  try:
   with requests.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=(20,90),stream=True) as r:
    assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
    with p.with_suffix('.part').open('wb') as f:
     for data in r.iter_content(1024*1024):f.write(data)
   assert p.with_suffix('.part').stat().st_size==b-a+1;p.with_suffix('.part').replace(p);return i
  except Exception:
   if attempt==3:raise
   time.sleep(attempt+1)
with ThreadPoolExecutor(max_workers=20) as pool:
 for i,j in enumerate(as_completed([pool.submit(one,x) for x in jobs]),1):j.result();print('BioDCASE chunks',i,'/',len(jobs),flush=True)
p=root/'Development_data.zip'
with p.open('wb') as f:
 for i,_,_ in jobs:
  q=parts/f'{i:03}.bin'
  with q.open('rb') as src:
   while data:=src.read(4*1024*1024):f.write(data)
h=hashlib.md5()
with p.open('rb') as f:
 while data:=f.read(4*1024*1024):h.update(data)
assert h.hexdigest()=='dbf35b9066e82cdee5461fca138f4986'
with zipfile.ZipFile(p) as z:
 names=z.namelist();(root/'biodcase_files.json').write_text(json.dumps(names));print('entries',len(names),names[:15],flush=True)
 # Only metadata initially; acquisition provenance must be established before use.
 for name in names:
  if not name.endswith('/') and (name.endswith(('.txt','.json','.csv','.md'))):
   dest=root/'biodcase'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
(root/'biodcase_download_audit.json').write_text(json.dumps(dict(url=url,bytes=size,md5=h.hexdigest(),md5_matches=True)))
print('BioDCASE complete and verified',flush=True)
