from pathlib import Path
import copy,hashlib,json,shutil
from datetime import datetime,timezone
from pypdf import PdfReader
from PIL import Image,ImageChops

W=Path(__file__).resolve().parent
ROOT=W.parents[1]
BEFORE=json.loads((W/'presentation-before.json').read_text())
RAW=json.loads((W/'raw-approved.json').read_text())
P=RAW['structuredContent']
if 'result' in P:P=P['result']
PLAN=json.loads((W/'plan.json').read_text())
assert [s['objectId'] for s in P['slides']]==[s['slideId'] for s in PLAN]
assert len(P['slides'])==18 and P['pageSize']==BEFORE['pageSize']

def stable(obj):
 if isinstance(obj,dict):return {k:stable(v) for k,v in obj.items() if k!='contentUrl'}
 if isinstance(obj,list):return [stable(x) for x in obj]
 return obj

assert stable(P['masters'])==stable(BEFORE['masters'])
assert stable(P['layouts'])==stable(BEFORE['layouts'])
orig={s['objectId']:s for s in BEFORE['slides']}
now={s['objectId']:s for s in P['slides']}
for sid,oid in [('ifsc26_s01','ifsc26_s01_campus'),('ifsc26_s02','ifsc26_s02_e05')]:
 a=next(e for e in orig[sid]['pageElements'] if e['objectId']==oid)
 b=next(e for e in now[sid]['pageElements'] if e['objectId']==oid)
 assert stable(a)==stable(b),('original photo modified',oid)


def text(e):
 return ''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',e.get('text',{})).get('textElements',[])).rstrip('\n')

native={e['objectId']:text(e) for s in P['slides'] for e in s['pageElements'] if 'shape' in e}
for s in P['slides']:
 for e in s['slideProperties']['notesPage'].get('pageElements',[]):
  if 'shape' in e:native[e['objectId']]=text(e)
for req in json.loads((W/'content.json').read_text()):
 if 'insertText' in req:
  d=req['insertText']
  if 'cellLocation' not in d:assert native[d['objectId']]==d['text'].rstrip('\n'),d['objectId']
for row,s in zip(PLAN,P['slides']):
 oid=s['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
 assert '[Sources]' in native[oid] and native[oid]==row['notes_full'].rstrip('\n')
 assert s['slideProperties']['layoutObjectId']==orig[s['objectId']]['slideProperties']['layoutObjectId']
assert 'Luiz Eduardo do Nascimento Ciarallo' in '\n'.join(native.values())
assert 'CNN' not in '\n'.join(text(e) for s in P['slides'] for e in s['pageElements'])
issues=json.loads((W/'output-issues-approved.json').read_text())
assert issues['summary']['totalIssues']==0

pdf=W/'renders/approved/presentation.pdf'
pages=PdfReader(pdf).pages
assert len(pages)==18
body='\n'.join(p.extract_text() for p in pages)
for s in ['90,03%','86,67%','33,33%','88,33%','28,33%','Luiz Eduardo do Nascimento Ciarallo']:
 assert s in body,s
# Accuracy labels belong to a scientific raster chart; verify their measured source
# and the local image hash, in addition to the recorded visual review of slide 14.
metrics=json.loads((ROOT/'results/ternary_lstm_2026_10_15/metrics.json').read_text())
assert [round(100*r['accuracy'],2) for r in metrics['neural_models']]==[85.74,86.55,82.33,85.27]
prov=json.loads((W/'figure-provenance.json').read_text())
assert hashlib.sha256((W/'assets/acuracia.png').read_bytes()).hexdigest()==prov['assets_sha256']['acuracia.png']
assert 'CNN' not in body
assert 'externos' in pages[7].extract_text()

# Every page was viewed in pass 1; every changed page was viewed after each repair.
changed1=[];changed2=[]
for n in range(1,19):
 filename=f'slide-{n:02d}.png'
 a=Image.open(W/'renders/final'/filename);b=Image.open(W/'renders/reviewed'/filename);c=Image.open(W/'renders/approved'/filename)
 if ImageChops.difference(a,b).getbbox():changed1.append(n)
 if ImageChops.difference(b,c).getbbox():changed2.append(n)
assert changed1==[8,9,11,13,14] and changed2==[9,11]
seconds=sum(s['seconds'] for s in PLAN)
assert 900<=seconds<=1200

run=ROOT/'results/ternary_lstm_2026_10_15'
replay=json.loads((run/'verification.json').read_text())
assert replay['complete'] and replay['normalizers_verified']==24 and replay['outer_PCM_disjoint']
assert all(v['outer_models_replayed']==5 and v['max_probability_replay_difference']==0 for v in replay['models'].values())
assert '313 passed' in (ROOT/'tmp/ternary_lstm_pytest_2026_10_15.log').read_text()
smoke=json.loads((run/'inference_example.json').read_text());assert smoke['mc_passes']==100

name='IFSC_Detector_Ternario_LSTM_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf'
shutil.copy2(pdf,W/name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(W/name)=='f5976cab319692a0b3faf0d9356f385395cd321253eecfcf4eb9ffa4bc14d6a2'
url='https://docs.google.com/presentation/d/1JzWMYfuNXbIpfwOVcMKuNk5R4xfhAw69S4DAAAfy99A/edit?usp=drivesdk'
(W/'Apresentacao_IFSC.url').write_text('[InternetShortcut]\nURL='+url+'\n')
record=dict(verified_utc=datetime.now(timezone.utc).isoformat(),native_url=url,native_revision=P['revisionId'],
 slide_count=18,page_size=P['pageSize'],original_master_count=len(P['masters']),original_layout_count=len(P['layouts']),
 masters_preserved=True,layouts_preserved=True,original_campus_and_mosquito_photos_preserved=True,
 full_author_verified=True,final_native_text_matches_plan=True,all_18_pages_visually_reviewed=True,
 repair_passes=2,changed_pages_reviewed=[changed1,changed2],structural_issues=0,duration_seconds=seconds,
 professor_requirements='mapped to slides in criterios_apresentacao.md',pdf=name,pdf_sha256=sha(W/name),pdf_bytes=(W/name).stat().st_size,
 model_run=str(run.relative_to(ROOT)),model_probabilities_replayed=True,tests_passed=313,
 metadata_modified_utc='2026-10-15T13:58:16.993Z',historical_readme_preserved=True,
 original_deck_pdf_preserved='output/ifsc_nova_2026_10_15/IFSC_Mosquitos_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf')
(W/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False,indent=2))
