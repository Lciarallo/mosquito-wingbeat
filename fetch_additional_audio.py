"""Fetch public source audio for computer-only experiments, without executing remote code."""
from pathlib import Path
import requests,json,re,time,hashlib,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
import libarchive,soundfile as sf
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/external/additional';OUT.mkdir(parents=True,exist_ok=True)
def fetch(url,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():return
 for attempt in range(4):
  try:
   r=requests.get(url,timeout=(20,90));r.raise_for_status();path.with_suffix(path.suffix+'.part').write_bytes(r.content);path.with_suffix(path.suffix+'.part').replace(path);return
  except Exception:
   if attempt==3:raise
   time.sleep(1+attempt)
def github(repo,name):
 r=requests.get(f'https://api.github.com/repos/{repo}/git/trees/main?recursive=1',timeout=30);r.raise_for_status();x=r.json();(OUT/f'{name}_tree.json').write_text(json.dumps(x))
 files=[i for i in x['tree'] if i['type']=='blob' and (i['path'].endswith('.wav') or ('Swarm_label' in i['path']))]
 with ThreadPoolExecutor(max_workers=6) as pool:
  jobs=[pool.submit(fetch,f"https://raw.githubusercontent.com/{repo}/main/{i['path']}",OUT/name/i['path']) for i in files]
  for job in as_completed(jobs):job.result()
 print(name,'complete',len(files),flush=True)
def figshare():
 p=OUT/'figshare_samples.rar';source=ROOT/'tmp/dataset_scout/figshare_samples.rar'
 if source.exists():shutil.copyfile(source,p)
 else:fetch('https://ndownloader.figshare.com/files/21827307',p)
 with libarchive.file_reader(str(p)) as archive:
  for entry in archive:
   if not entry.pathname.lower().endswith('.wav'):continue
   dest=OUT/'figshare'/entry.pathname;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.write_bytes(b''.join(entry.get_blocks()))
 print('figshare complete',len(list((OUT/'figshare').rglob('*.wav'))),flush=True)
def mosquitosong():
 listing=ROOT/'tmp/large_fresh/embedded_listing.html'
 if not listing.exists():fetch('https://drive.google.com/embeddedfolderview?id=1AksZW-3KCSPQqsPJ8LwyS1i_qjVDvtI6#list',listing)
 items=re.findall(r'id="entry-([^"]+)".*?<div class="flip-entry-title">([^<]+)</div>',listing.read_text(),re.S)
 records=[];failed=[]
 def one(item):
  fid,name=item;p=OUT/'mosquitosong'/name;fetch('https://drive.google.com/uc?export=download&id='+fid,p);info=sf.info(p)
  return dict(id=fid,path=str(p.relative_to(ROOT)),name=name,seconds=info.duration,sr=info.samplerate,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 with ThreadPoolExecutor(max_workers=8) as pool:
  jobs={pool.submit(one,item):item for item in items if item[1].endswith('.wav')}
  for j in as_completed(jobs):
   try:records.append(j.result())
   except Exception as e:failed.append(dict(item=jobs[j],error=str(e)))
   if (len(records)+len(failed))%100==0:print('mosquitosong',len(records),'ok',len(failed),'failed',flush=True)
 (OUT/'mosquitosong_manifest.json').write_text(json.dumps(dict(source='https://github.com/akaraspt/mosquitosongp',files=records,failed=failed),indent=2));print('mosquitosong complete',len(records),flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(figshare),pool.submit(github,'ComputerNetworks-UFRGS/Aedes_Detector_Smartphone_App_Paper','ufrgs'),pool.submit(github,'duydinhthai27/synthetic-swarm-mosquito','synthetic'),pool.submit(mosquitosong)]
  for job in as_completed(jobs):job.result()
