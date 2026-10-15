from pathlib import Path
import json,copy,hashlib,shutil
from PIL import Image
W=Path(__file__).resolve().parent;ROOT=W.parents[1];P=json.loads((W/'raw-before.json').read_text())['structuredContent'];SID='ifsc26_arduino_dataset'
s=next(s for s in P['slides'] if s['objectId']==SID);requests=[]
values={SID+'_e00':'Arduino: montagem para a coleta de dados',SID+'_e02':'Montagem do protótipo\nCaptura por microfone\nUSB → arquivos WAV\nColeta planejada\nAedes / outros / ruído\nPor indivíduo e sessão',SID+'_e03':'Montagem e placa · fotos fornecidas pelo autor',SID+'_e04':'Próxima etapa: coletar voos e criar a base própria.','ifsc26_s19_e08':'Fotos: IFSC, CDC, SES-RS, SESA-ES e acervo do projeto'}
for oid,value in values.items():
 e=next(e for sl in P['slides'] for e in sl['pageElements'] if e['objectId']==oid)
 styles=[t['textRun'].get('style',{}) for t in e['shape']['text']['textElements'] if 'textRun' in t];assert all(st==styles[0] for st in styles)
 st=copy.deepcopy(styles[0]);st.pop('link',None)
 if 'weightedFontFamily' in st:st.pop('fontFamily',None)
 st={k:v for k,v in st.items() if v!={}}
 requests.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':value}},{'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':st,'fields':','.join(st)}}])
assets=W/'assets';assets.mkdir(exist_ok=True);photos=[]
ATT=Path('/tmp/codex-remote-attachments/01a13c81-8919-7761-a5f7-04084ee7ad16/0cf7ee23-8f5f-471f-849a-57040ca7219f')
for source,name in [('1-IMG-20261015-WA0101.jpg','prototipo_montado.jpg'),('2-IMG-20261015-WA0102.jpg','placa_na_base.jpg')]:
 p=assets/name;shutil.copyfile(ATT/source,p)
 with Image.open(p) as im:width,height=im.size
 photos.append(dict(path=str(p),original=str(ATT/source),width=width,height=height,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),credit='Fotografia fornecida pelo autor; autoria fotográfica não especificada',unaltered=True))
# Preserve the existing native image object and use uniform scaling.
e=next(e for e in s['pageElements'] if 'image' in e);width=145;height=145*photos[0]['height']/photos[0]['width'];base=e['size']['width']['magnitude']/12700
requests.append({'updatePageElementTransform':{'objectId':e['objectId'],'applyMode':'ABSOLUTE','transform':{'scaleX':width/base,'scaleY':width/base,'translateX':280,'translateY':180-height/2,'unit':'PT'}}})
images=[dict(path=photos[0]['path'],requests=[{'replaceImage':{'imageObjectId':e['objectId'],'url':photos[0]['path'],'imageReplaceMethod':'CENTER_INSIDE'}}]),dict(path=photos[1]['path'],requests=[{'createImage':{'objectId':SID+'_foto_placa_real','url':photos[1]['path'],'elementProperties':{'pageObjectId':SID,'size':{'width':{'magnitude':145,'unit':'PT'},'height':{'magnitude':145*photos[1]['height']/photos[1]['width'],'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':433,'translateY':180-145*photos[1]['height']/photos[1]['width']/2,'unit':'PT'}}}}])]
plan=json.loads((ROOT/'output/ifsc_introducao_problema_2026_10_15/plan.json').read_text())
notes='''Tempo sugerido: 65 segundos.

Estas duas fotografias, fornecidas por Luiz Eduardo do Nascimento Ciarallo, mostram a montagem física do protótipo: à esquerda, o recipiente transparente com a placa na base e ligação USB; à direita, o detalhe da placa e da conexão. A montagem apresentada será usada na preparação da coleta acústica.

A próxima etapa é implementar e verificar a captura do microfone, transferir áudio PCM por USB e salvar WAVs no computador. A proposta inicial é mono, 16 kHz e 16 bits, com verificação de saturação, taxa efetiva e perdas de amostras. As fotos documentam a montagem; o novo dataset ainda será coletado.

Vamos registrar voos de Aedes aegypti, outros mosquitos e controles de ambiente sem mosquito, com espécie confirmada e metadados de indivíduo, sessão, distância, temperatura e umidade. O áudio de enxame do Drive está fora do escopo desta coleta e da avaliação atual do detector.

Treino, validação e teste serão separados por indivíduo e sessão. Parte das novas gravações avaliará o detector congelado; apenas a parcela de treino poderá ser usada para reajustá-lo. Os 90,03% anteriores não incluem essa coleta futura.

[Sources]
Fotografias fornecidas pelo usuário nesta conversa: IMG-20261015-WA0101.jpg e IMG-20261015-WA0102.jpg; originais preservados sem edição, com hashes em photo-provenance.json.
https://docs.arduino.cc/tutorials/nano-33-ble-sense-rev2/microphone-sensor/
Arquivo local: firmware/MosquitoPresence/MosquitoPresence.ino
Arquivo local: reports/exclusao_audio_drive_2026_10_15.md
Montagem física documentada pelas fotografias; protocolo de aquisição e dataset apresentados como próxima etapa.
[/Sources]'''
for row in plan:
 if row['slideId']==SID:row['title']=values[SID+'_e00'];row['notes_full']=notes
 if row['slideId']=='ifsc26_s19':row['notes_full']=row['notes_full'].replace('e a foto oficial da placa Arduino','e as duas fotografias da montagem fornecidas pelo autor').replace('[/Sources]','Fotos do protótipo: IMG-20261015-WA0101.jpg e IMG-20261015-WA0102.jpg, fornecidas pelo autor nesta conversa.\n[/Sources]')
 if row['slideId'] in [SID,'ifsc26_s19']:
  sl=next(sl for sl in P['slides'] if sl['objectId']==row['slideId']);oid=sl['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId'];requests.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':row['notes_full']}}])
for name,value in [('content',requests),('images',images),('plan',plan),('photo-provenance',photos),('frame-map',dict(slideId=SID,exemplar=SID,construction='Edit native inherited elements in place',elementMap={el['objectId']:('replace with first user photograph; uniform scale' if 'image' in el else 'replace text and preserve native style' if el['objectId'] in values else 'keep unchanged') for el in s['pageElements']},addedElement=SID+'_foto_placa_real',photoPlacement='Two full, unaltered 4:3 images side by side at 145 x 108.75 pt',sourceSlideCount=20,finalSlideCount=20))]:
 (W/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
(W/'Roteiro_apresentacao_IFSC.txt').write_text('\n\n'.join(f"{r['number']:02d}. {r.get('title') or 'Capa'} ({r['seconds']} s)\n{r['notes_full']}" for r in plan)+'\n\nTempo total sugerido: 19min35s.\n')
(W/'analise-lacunas.txt').write_text('O slide 18 usa foto genérica da placa. Substituir por duas fotos originais da montagem fornecidas pelo usuário; preservar o template, os resultados e 20 slides. Distinguir montagem física documentada da coleta de WAVs proposta. Créditos devem dizer fornecidas pelo autor, sem presumir autoria fotográfica. Não atribuir espécie aos recipientes nem afirmar coleta ou desempenho embarcado comprovado.\n')
(W/'fallback-reasons.txt').write_text('Native Google Slides edited in place; no PPTX roundtrip. Reuse local Node and Poppler because the managed dependency loader is unavailable. The authoring marker uses supported pptx format. Export uses the established bounded legacy include_base64 compatibility adapter because the streamed connector reference was not materialized to workspace_path.\n')
print(json.dumps(requests,ensure_ascii=False,separators=(',',':')))
