from pathlib import Path
import copy,json,shutil,hashlib

W=Path(__file__).resolve().parent
ROOT=W.parents[1]
P=json.loads((W/'raw-before.json').read_text())['structuredContent']
source=next(s for s in P['slides'] if s['objectId']=='ifsc26_s02')
newid='ifsc26_arduino_dataset'
mapping={source['objectId']:newid,**{e['objectId']:f'{newid}_e{i:02d}' for i,e in enumerate(source['pageElements'])}}
copy_request=[{'duplicateObject':{'objectId':source['objectId'],'objectIds':mapping}}]
requests=[]
copy_text={
 'ifsc26_s02_e00':'Coleta futura: novo dataset com Arduino',
 'ifsc26_s02_e01':'17',
 'ifsc26_s02_e02':'Capturar o som\nMicrofone → USB → WAV\nAnotar a coleta\nAedes / outros / ruído\nSeparar os dados\nPor indivíduo e sessão',
 'ifsc26_s02_e03':'Nano 33 BLE Sense Rev2 · foto oficial Arduino',
 'ifsc26_s02_e04':'Base própria para treinar e validar o detector.',
}
for oid,value in copy_text.items():
 e=next(e for e in source['pageElements'] if e['objectId']==oid)
 styles=[t['textRun'].get('style',{}) for t in e['shape']['text']['textElements'] if 'textRun' in t]
 assert all(s==styles[0] for s in styles)
 st=copy.deepcopy(styles[0]);st.pop('link',None)
 if 'weightedFontFamily' in st:st.pop('fontFamily',None)
 st={k:v for k,v in st.items() if v!={}}
 requests.extend([{'deleteText':{'objectId':mapping[oid],'textRange':{'type':'ALL'}}},
  {'insertText':{'objectId':mapping[oid],'insertionIndex':0,'text':value}},
  {'updateTextStyle':{'objectId':mapping[oid],'textRange':{'type':'ALL'},'style':st,'fields':','.join(st)}}])
for sid,page in [('ifsc26_s19','18'),('ifsc26_s20','19')]:
 requests.extend([{'deleteText':{'objectId':sid+'_e01','textRange':{'type':'ALL'}}},
  {'insertText':{'objectId':sid+'_e01','insertionIndex':0,'text':page}}])
requests.extend([{'deleteText':{'objectId':'ifsc26_s19_e08','textRange':{'type':'ALL'}}},
 {'insertText':{'objectId':'ifsc26_s19_e08','insertionIndex':0,'text':'Fotos: IFSC, CDC e Arduino · modelo: IFSC'}}])

asset=W/'assets/arduino_nano_sense_rev2.jpg'
asset.parent.mkdir(exist_ok=True)
shutil.copy2(ROOT/'output/ifsc_2026_10_13_visual/assets/arduino_nano_sense_rev2.jpg',asset)
image=[{'replaceImage':{'imageObjectId':mapping['ifsc26_s02_e05'],'url':str(asset),'imageReplaceMethod':'CENTER_INSIDE'}}]
notes='''Tempo sugerido: 65 segundos.

O próximo passo será criar um dataset próprio usando o Arduino Nano 33 BLE Sense ou Sense Rev2 como dispositivo de aquisição. A foto é oficial da Arduino e ilustra a placa planejada para a coleta.

Será preparado um firmware de aquisição pelo microfone PDM integrado, com transferência binária por USB para um computador e gravação de WAVs PCM mono. O ponto de partida proposto é 16 kHz e 16 bits; essa configuração será medida na placa, verificando taxa efetiva, saturação e perda de amostras. Os áudios brutos serão preservados, permitindo gerar a versão de 8 kHz compatível com a comparação atual, sem descartar o original de maior frequência de amostragem.

Serão gravadas as três classes do detector: Aedes aegypti, outros mosquitos e períodos sem mosquito, incluindo sons de ambiente. A classe de cada coleta virá da identificação confirmada do material biológico e do protocolo de observação, não da previsão do próprio modelo. A captura deverá incluir trechos contínuos e controles negativos, além de sobrevoos, para medir também falsos alarmes.

Cada arquivo terá identificadores de indivíduo, sessão e placa, espécie, data, distância ao microfone e condições de temperatura, umidade e ruído. As gravações e os intervalos anotados serão ligados por um manifesto, com versões e hashes dos arquivos.

Os conjuntos de treino, validação e teste serão separados por indivíduo e sessão; recortes da mesma origem não poderão atravessar conjuntos. Uma parcela será reservada antes dos ajustes para avaliar o detector congelado no novo domínio de captura. Em uma etapa posterior, somente a parcela de treino poderá alimentar um novo ajuste, com validação separada e teste preservado. Serão novamente medidos acurácia geral, sensibilidade de Aedes e falsos positivos nas duas classes negativas; falsos alarmes por hora exigirão anotação temporal adequada.

Este slide apresenta o plano de coleta futura. O novo dataset não compõe os resultados de 90,03% apresentados anteriormente. A etapa inicial usa a placa para capturar dados; a classificação dos novos áudios poderá ser executada no computador, seguida de avaliação de porte embarcado em outra etapa.

[Sources]
https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2/
https://docs.arduino.cc/tutorials/nano-33-ble-sense-rev2/microphone-sensor/
https://store.arduino.cc/products/nano-33-ble-sense-rev2
Arquivo local: firmware/README.md
Arquivo local: firmware/MosquitoPresence/MosquitoPresence.ino
Arquivo local: output/ifsc_2026_10_13_visual/assets/arduino_nano_sense_rev2.jpg
Proposta futura de coleta elaborada para esta apresentação; não é um resultado medido.
[/Sources]'''
oldplan=json.loads((ROOT/'output/ifsc_ternario_lstm_2026_10_15/plan.json').read_text())
# The short oral script is maintained as editable text; detailed acquisition notes
# are preserved separately and should not all be read during the 65-second slot.
if (W/'arduino-notes.txt').exists():notes=(W/'arduino-notes.txt').read_text()
newrow=dict(slideId=newid,number=17,title=copy_text['ifsc26_s02_e00'],seconds=65,notes_full=notes)
plan=oldplan[:16]+[newrow]+[dict(s,number=i) for s,i in zip(oldplan[16:],[18,19])]
plan[-2]['notes_full']=plan[-2]['notes_full'].replace('e a fotografia do mosquito do CDC',
 'a fotografia do mosquito do CDC e a foto oficial da placa Arduino').replace('[/Sources]',
 'https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2/\nhttps://store.arduino.cc/products/nano-33-ble-sense-rev2\n[/Sources]')
assert len(plan)==19 and sum(s['seconds'] for s in plan)==1110
sources={e['objectId']:{'target':mapping[e['objectId']],'action':'replace native text; preserve all source run styles' if 'shape' in e else 'replace CDC photo with official Arduino photo; CENTER_INSIDE'} for e in source['pageElements']}
for name,value in [('duplicate',copy_request),('content',requests),('image',image),('plan',plan),('frame-map',dict(
 narrative='Explain future Arduino audio acquisition and dataset organization',exemplar=source['objectId'],
 method='duplicate existing native photo-led slide',allSourceElementsMapped=True,elementMap=sources,
 finalOrder=[s['slideId'] for s in plan],officialMasterAndLayouts='preserve unchanged'))]:
 (W/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
(W/'arduino-notes.txt').write_text(notes)
(W/'Roteiro_apresentacao_IFSC.txt').write_text('\n\n'.join(f"{s['number']:02d}. {s.get('title') or 'Capa'} ({s['seconds']} s)\n{s['notes_full']}" for s in plan)+'\n\nTempo total sugerido: 18min30s.\n')
(W/'analise-lacunas.txt').write_text('O deck atual tem 18 slides e encerra com discussão geral, sem explicar a nova coleta de áudio por Arduino. Será duplicado o exemplar nativo ifsc26_s02, substituindo todos os textos e a foto por um plano futuro de dataset e uma foto oficial da placa. O novo slide ficará após a matriz de confusão e antes das referências; a discussão continuará no final. Os resultados anteriores, fotos do IFSC/mosquito, mestre, 11 layouts e demais elementos serão preservados. Plano de captura e organização de dados não será apresentado como implementação física já realizada.\n')
(W/'fallback-reasons.txt').write_text('Existing native Google Slides deck edited via its connector; no local PPTX roundtrip. The workspace dependency loader is unavailable in this session, so the native reader/parser/checker/export helper and available Poppler are reused. The marker supports pptx rather than native Slides, and records an edit with that supported format. The previous export established bounded legacy base64 compatibility because streaming references have no workspace_path; use the existing export helper with that compatibility adapter if necessary.\n')
(W/'source-notes.txt').write_text('Official Arduino Nano 33 BLE Sense Rev2 documentation confirms the integrated digital PDM microphone. Official product photograph reused from the prior sourced project asset. Acquisition and dataset choices are proposed future work, not product specifications or measured outcomes.\nImage SHA256: '+hashlib.sha256(asset.read_bytes()).hexdigest()+'\n')
print('Prepared one native slide, 19 total; 18min30s')
