from pathlib import Path
import json,copy,hashlib,shutil,datetime,zipfile
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
a=json.loads((W/'raw-before.json').read_text())['structuredContent'];b=json.loads((W/'raw-output.json').read_text())['structuredContent']
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k!='contentUrl'}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
assert a['masters']==b['masters'] and a['layouts']==b['layouts']
assert [s['objectId'] for s in a['slides']]==[s['objectId'] for s in b['slides']]
for i,(s,t) in enumerate(zip(a['slides'],b['slides'])):
 if i not in [17,18]:assert clean(s)==clean(t),i
assert len(b['slides'])==20
plan=json.loads((W/'plan.json').read_text());assert sum(x['seconds'] for x in plan)==1175
for row,sl in zip(plan,b['slides']):
 oid=sl['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
 elem=next(e for e in sl['slideProperties']['notesPage']['pageElements'] if e['objectId']==oid)
 notes=''.join(x.get('textRun',{}).get('content','') for x in elem['shape']['text']['textElements']).strip()
 assert notes==row['notes_full'].strip(),row['number']
photos=json.loads((W/'photo-provenance.json').read_text())
for row in photos:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
old=ROOT/'output/ifsc_introducao_problema_2026_10_15/renders/approved';unchanged=[]
for i in range(1,21):
 p=f'slide-{i:02d}.png'
 if (W/'renders/final'/p).read_bytes()==(old/p).read_bytes():unchanged.append(i)
assert unchanged==list(range(1,18))+[20]
name='IFSC_Mosquitos_Prototipo_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf';pdf=W/name;shutil.copyfile(W/'renders/final/presentation.pdf',pdf)
receipt=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=b['revisionId'],slides=20,modified_slides=[18,19],unchanged_pngs=unchanged,official_master_and_layouts_preserved=True,all_other_slides_identical_after_stripping_ephemeral_urls=True,photos_original_and_aspect_ratio_preserved=True,visual_review='Slides 18 and 19 reviewed at full resolution; remaining 18 PNGs byte-identical to the previously inspected deck.',structural_issues=0,repair_passes=0,notes_match_plan=True,duration_s=1175,pdf=name,pdf_bytes=pdf.stat().st_size,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),native_url='https://docs.google.com/presentation/d/1JzWMYfuNXbIpfwOVcMKuNk5R4xfhAw69S4DAAAfy99A/edit?usp=drivesdk')
(W/'verification.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False))
(W/'visual-review.txt').write_text('Slide 18: two full 4:3 photos, correct caption, title/banner single-line, body readable, no overlaps or clipping. Slide 19: updated photo credits fit in native row. Other 18 pages byte-identical to approved prior render. Official master/layouts and all scientific results unchanged.\n')
u=Path('/tmp/mosquito_prototipo_fotos_upload_2026_10_15');u.mkdir(exist_ok=True)
# Update the existing standalone Drive PDF so its existing file link stays valid.
shutil.copyfile(pdf,u/'IFSC_Mosquitos_Introducao_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf')
original_readme=ROOT/'output/entrega_drive_2026_10_15/LEIA-ME.txt'
readme=original_readme.read_text()+'\nATUALIZAÇÃO: FOTOS DO PROTÓTIPO — 15/10/2026\nO PDF avulso e os slides editáveis incluem agora as duas fotos reais da montagem no slide 18 e créditos atualizados no slide 19. O ZIP principal é o snapshot anterior a essa alteração. O pacote Atualizacao_fotos_Arduino_2026-10-15.zip contém os arquivos da apresentação atualizada e as fotos originais; extraia sua estrutura sobre a cópia do projeto. A apresentação continua com 20 slides e roteiro de 19min35s.\n'
(u/'LEIA-ME.txt').write_text(readme)
package=u/'Atualizacao_fotos_Arduino_2026-10-15.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for p in sorted(W.rglob('*')):
  if p.is_file() and 'renders' not in p.relative_to(W).parts:z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(package) as z:assert z.testzip() is None
print(json.dumps(receipt,ensure_ascii=False));print('UPLOAD',u,'ZIP_BYTES',package.stat().st_size)
