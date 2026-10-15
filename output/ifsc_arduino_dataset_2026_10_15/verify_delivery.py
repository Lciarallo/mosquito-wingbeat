from pathlib import Path
import json,hashlib,shutil,unicodedata
from datetime import datetime,timezone
from pypdf import PdfReader
from PIL import Image,ImageChops

W=Path(__file__).resolve().parent
A=json.loads((W/'raw-before.json').read_text())['structuredContent']
B=json.loads((W/'raw-approved.json').read_text())['structuredContent']
plan=json.loads((W/'plan.json').read_text())
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='contentUrl'}
    if isinstance(v,list):return [clean(x) for x in v]
    return v
def text(e):
    return ''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[])).rstrip('\n')
assert len(B['slides'])==19
assert [s['objectId'] for s in B['slides']]==[s['slideId'] for s in plan]
assert clean(A['masters'])==clean(B['masters']) and clean(A['layouts'])==clean(B['layouts'])
assert A['pageSize']==B['pageSize']
old={s['objectId']:s for s in A['slides']}
for s in B['slides']:
    if s['objectId'] not in old:continue
    for e,a in zip(s['pageElements'],old[s['objectId']]['pageElements']):
        if e['objectId'] in ['ifsc26_s19_e01','ifsc26_s20_e01','ifsc26_s19_e08']:continue
        assert clean(e)==clean(a),e['objectId']
for s,row in zip(B['slides'],plan):
    n=s['slideProperties']['notesPage'];oid=n['notesProperties']['speakerNotesObjectId']
    note=next(e for e in n['pageElements'] if e['objectId']==oid)
    assert text(note)==row['notes_full'].rstrip('\n'),s['objectId']
    assert '[Sources]' in text(note)
assert json.loads((W/'output-issues-approved.json').read_text())['summary']['totalIssues']==0
pdf=W/'renders/approved/presentation.pdf';pages=PdfReader(pdf).pages
assert len(pages)==19
body=[unicodedata.normalize('NFKC',p.extract_text()) for p in pages]
assert 'Coleta futura: novo dataset com Arduino' in body[16]
assert '90,03%' in body[13]
assert 'Luiz Eduardo do Nascimento Ciarallo' in body[0]
assert 'Discussão final' in body[-1]
changed=[]
for n in range(1,20):
    f=f'slide-{n:02d}.png'
    if ImageChops.difference(Image.open(W/'renders/final'/f),Image.open(W/'renders/approved'/f)).getbbox():changed.append(n)
assert changed==[17,18]
assert sum(s['seconds'] for s in plan)==1110
filename='IFSC_Mosquitos_Arduino_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf'
shutil.copy2(pdf,W/filename)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(W/filename)=='cf3a5cb32cc83f0e96e820a3bbd3e162b80ebffbbdb1e6f1155709b85e28b591'
url='https://docs.google.com/presentation/d/1JzWMYfuNXbIpfwOVcMKuNk5R4xfhAw69S4DAAAfy99A/edit?usp=drivesdk'
record=dict(verified_utc=datetime.now(timezone.utc).isoformat(),native_url=url,presentation_id=B['presentationId'],
    revision=B['revisionId'],slides=19,new_slide_number=17,new_slide_title='Coleta futura: novo dataset com Arduino',
    duration_seconds=1110,all_slides_visually_reviewed=True,consolidated_repair_passes=1,repaired_pages_reviewed=changed,
    master_count=1,layout_count=11,official_template_preserved=True,original_results_preserved=True,
    future_collection_explicit=True,independent_ground_truth_in_notes=True,individual_session_split_in_notes=True,
    official_arduino_photo=True,arduino_image_sha256=sha(W/'assets/arduino_nano_sense_rev2.jpg'),author_verified=True,
    pdf=filename,pdf_sha256=sha(W/filename),pdf_bytes=(W/filename).stat().st_size,structural_issues=0,
    prior_pdf_preserved='output/ifsc_ternario_lstm_2026_10_15/IFSC_Detector_Ternario_LSTM_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf')
(W/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
(W/'Apresentacao_IFSC.url').write_text('[InternetShortcut]\nURL='+url+'\n')
(W/'criterios_apresentacao.txt').write_text('Verificação da edição com plano Arduino\n\nContextualização: slides 1-2.\nTrabalhos relacionados: 3, 5, 10 e 18.\nRede neural profunda implementada, processamento, otimização e regularização: 6-13.\nResultados: 13-16.\nNovo dataset com Arduino, como proposta futura: 17.\nReferências e créditos: 18.\nDiscussão final e limitações: 19.\nRoteiro: 18min30s, dentro dos 15-20 minutos exigidos.\nAutor: Luiz Eduardo do Nascimento Ciarallo.\nFoto e modelo oficial IFSC preservados.\nTodos os 19 slides renderizados e inspecionados visualmente.\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
