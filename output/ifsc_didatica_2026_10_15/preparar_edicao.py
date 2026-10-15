from pathlib import Path
import json,copy,hashlib
from PIL import Image
W=Path(__file__).resolve().parent;P=json.loads((W/'raw-before.json').read_text())['structuredContent'];plan=json.loads((W/'plan.json').read_text());req=[];changes={};images=[]
all_elements={e['objectId']:e for s in P['slides'] for e in s['pageElements']}
def replace(oid,text):
 e=all_elements[oid];styles=[x['textRun'].get('style',{}) for x in e['shape']['text']['textElements'] if 'textRun' in x]
 assert all(x==styles[0] for x in styles),oid
 st=copy.deepcopy(styles[0]);st.pop('link',None)
 if 'weightedFontFamily' in st:st.pop('fontFamily',None)
 st={k:v for k,v in st.items() if v!={}}
 req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':text}},{'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':st,'fields':','.join(st)}}]);changes[oid]='replace native copy; preserve uniform source text and paragraph styles'
def subtitle(sid,text):
 oid=sid+'_didatica_subtitulo';req.extend([{'createShape':{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':{'pageObjectId':sid,'size':{'width':{'magnitude':556,'unit':'PT'},'height':{'magnitude':24,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':28,'translateY':75,'unit':'PT'}}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':text}},{'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'weightedFontFamily':{'fontFamily':'Oswald','weight':300},'fontSize':{'magnitude':18,'unit':'PT'},'foregroundColor':{'opaqueColor':{'rgbColor':{'red':.21,'green':.29,'blue':.23}}}},'fields':'weightedFontFamily,fontSize,foregroundColor'}}]);changes[oid]='new concise explanatory subtitle in native template typography'
for row in plan:
 sid=row['slideId'];sl=next(s for s in P['slides'] if s['objectId']==sid)
 title=sid+'_e00';replace(title,'Mosquitos\npelo som' if sid=='ifsc26_s01' else row['title'])
 if row['number']>1:
  counters=[e['objectId'] for e in sl['pageElements'] if ''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[])).strip().isdigit() and len(''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[])).strip())==2]
  assert len(counters)==1,(sid,counters);replace(counters[0],f"{row['number']:02d}")
 oid=sl['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId'];req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':row['notes_full']}}])
texts={
'ifsc26_s01_e02':'Detecção de Aedes aegypti\npor análise acústica',
'ifsc26_introducao_problema_e02':'Hoje: vistoria e coleta.\nIdentificação com lupa.\nProposta: analisar o voo.\nClassificar o áudio.\nSinalizar presença de Aedes.\nApoiar o monitoramento.',
'ifsc26_introducao_problema_e04':'Triagem acústica para apoiar a vigilância.',
'ifsc26_s02_e02':'Entrada: áudio de voo.\nSaída por gravação:\nAedes aegypti\nOutro mosquito\nRuído / sem mosquito\nAprendizado supervisionado.',
'ifsc26_s02_e04':'Rótulos conhecidos ensinam as três classes.',
'ifsc26_s03_e03':'Captura com celulares', 'ifsc26_s03_e05':'Memória temporal','ifsc26_s03_e07':'Robustez ao ruído','ifsc26_s03_e11':'Técnicas adaptadas à tarefa ternária; resultados medidos neste projeto.',
'ifsc26_s08_e01':'5 dobras por grupo: 4 ajustam; 1 testa · cada grupo testado uma vez.\nDryad/HumBug: mosquitos e fundos · ESC-50: sons ambientais.',
'ifsc26_s04_e01':'FFT decompõe o sinal temporal em componentes de frequência.',
'ifsc26_s05_e01':'Fundamental: ritmo básico · harmônicos: múltiplos dessa frequência.',
'ifsc26_s06_e01':'Hann 128 ms · avanço 16 ms · FFT 1.024 · áudio a 8 kHz.',
'ifsc26_s07_e01':'64 bandas de 80–4.000 Hz · 55 quadros por trecho · energia em dB.',
'ifsc26_s11_e01':'LR: regressão logística · HistGB: árvores em boosting · gate: limiar.',
 'tern_classic_result':'Dois modelos, uma decisão por gravação',
'ifsc26_s09_e01':'Duas camadas · 128 unidades por direção · dropout 0,20 · três saídas.',
 'tern_lstm_label':'LSTM · memória do início até o fim', 'tern_bilstm_label':'Bi-LSTM · contexto nos dois sentidos',
'ifsc26_s12_e01':'Clipping limita gradientes altos e ajuda a estabilizar o treino.',
 'tern_adam_node0':'Entropia cruzada\nmede o erro','tern_adam_node1':'Gradiente\nindica o ajuste','tern_adam_node2':'Adam\nadapta o passo','tern_adam_node3':'Pesos\nsão atualizados','tern_adam_info':'Taxa 0,0006 · lote de 256 janelas\nClipping = 3 · até 30 épocas',
'ifsc26_regularizacao_e03':'Dropout reduz dependências entre unidades',
'ifsc26_regularizacao_e04':'Zera 20% das ativações entre camadas.',
'ifsc26_regularizacao_e06':'Amostragem equilibra as três classes',
'ifsc26_regularizacao_e07':'Classes equilibradas; peso por gravação com γ = 0,50.',
'ifsc26_regularizacao_e09':'Validação escolhe quando parar',
'ifsc26_regularizacao_e10':'Guarda a menor perda; paciência de 5 épocas.',
'ifsc26_regularizacao_e11':'MC dropout: 100 passagens medem discordância na inferência.',
'ifsc26_s14_e01':'LSTM: melhor época 4 de 9 observadas · Bi-LSTM: 2 de 7 · contexto final.',
'ifsc26_s15_e01':'Mesmos 1.494 grupos · direta: maior probabilidade · calibrada: dois limiares.',
'ifsc26_s10_e01':'Sensibilidade: TP/60 · FPR ruído: FP/426 · FPR outro: FP/1.008.',
'ifsc26_s17_e01':'Aedes: 47/60 detectados · precisão: 47/70 alertas = 67,14%.',
'ifsc26_arduino_dataset_e02':'Microfone capta o voo.\nUSB leva o áudio ao PC.\nWAV preserva o sinal.\nIdentificar espécie e sessão.\nIncluir outros e ruído.\nSeparar indivíduos no teste.',
'ifsc26_arduino_dataset_e04':'Próximo passo: coletar, rotular e validar.',
'ifsc26_s19_e02':'Mukundarajan et al. (2017) · captura com celulares',
'ifsc26_s19_e03':'Torres et al. (2025) · memória temporal e MC dropout',
'ifsc26_s19_e04':'Supratak et al. (2024) · CNN e robustez ao ruído',
'ifsc26_s19_e05':'Sinka et al. (2021) · frequências e HumBug',
'ifsc26_s19_e06':'Código e resultados · github.com/Lciarallo/mosquito-wingbeat',
'ifsc26_s19_e07':'Conceitos, contas e fontes completas no guia e nas notas.',
'ifsc26_s19_e08':'Marca: IFSC · fotos: CDC, SES-RS, SESA-ES e acervo do projeto',
'ifsc26_s20_e03':'O que foi demonstrado',
'ifsc26_s20_e04':'Cascata: 90,03%; redes diretas detectam mais Aedes.',
'ifsc26_s20_e06':'Onde a conclusão se aplica',
'ifsc26_s20_e07':'Corpus conhecido; sessões e dispositivos podem se repetir.',
'ifsc26_s20_e09':'Próxima validação',
'ifsc26_s20_e10':'Coleta própria, sessões novas e detector congelado.',
'ifsc26_s20_e11':'MosquitoSong externo: 0/4 grupos Aedes · redes sem porte embarcado validado.'}
for oid,t in texts.items():replace(oid,t)
subs={'ifsc26_s08':'Dryad, HumBug e ESC-50 · 42.029 janelas de 0,992 s',
'ifsc26_s04':'Um mesmo trecho real de voo: amplitude no tempo e espectro.',
'ifsc26_s05':'Médias publicadas a 25 °C para fêmeas e machos · Sinka et al. (2021)',
'ifsc26_s06':'A STFT repete a FFT em janelas para preservar a posição no tempo.',
'ifsc26_s07':'Mel resume a energia em bandas: uma sequência compacta para a rede.',
'ifsc26_s11':'A referência combina atributos acústicos antes de aplicar dois limiares.',
'ifsc26_s09':'A memória temporal relaciona quadros mel de um mesmo trecho.',
'ifsc26_s12':'O erro gera gradientes; o otimizador atualiza os parâmetros.',
'ifsc26_s14':'Época: um ciclo de amostragem e atualização dos pesos.',
'ifsc26_s15':'Acurácia = classificações corretas ÷ todos os grupos avaliados.',
'ifsc26_s17':'Linhas: classe real · colunas: previsão · diagonal: acertos.'}
for sid,text in subs.items():subtitle(sid,text)
for sid in ['ifsc26_s08','ifsc26_s04','ifsc26_s05','ifsc26_s06','ifsc26_s07','ifsc26_s14','ifsc26_s15','ifsc26_s17']:
 e=all_elements[sid+'_e02'];base=e['size']['width']['magnitude']/12700;req.append({'updatePageElementTransform':{'objectId':e['objectId'],'applyMode':'ABSOLUTE','transform':{'scaleX':540/base,'scaleY':540/base,'translateX':39,'translateY':104,'unit':'PT'}}});changes[e['objectId']]='keep measured scientific figure; uniform resize to accommodate explanatory subtitle'
# Replace the campus photo with an authentic official vertical logo at its own aspect ratio.
for oid in ['ifsc26_s01_campus','ifsc26_s01_campus_credit']:req.append({'deleteObject':{'objectId':oid}});changes[oid]='delete campus photograph/obsolete caption as explicitly requested'
logo=str(W/'assets/logo_ifsc_florianopolis_oficial.png');images.append(dict(path=logo,requests=[{'createImage':{'objectId':'ifsc26_logo_oficial_florianopolis','url':logo,'elementProperties':{'pageObjectId':'ifsc26_s01','size':{'width':{'magnitude':150,'unit':'PT'},'height':{'magnitude':150*965/662,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':418,'translateY':62,'unit':'PT'}}}}]))
for sid,name in [('ifsc26_s08','dados_e_classes.png'),('ifsc26_s15','acuracia_cinco_regras.png')]:
 path=str(W/'assets'/name);images.append(dict(path=path,requests=[{'replaceImage':{'imageObjectId':sid+'_e02','url':path,'imageReplaceMethod':'CENTER_INSIDE'}}]));changes[sid+'_e02']='replace with scientific figure recalculated from frozen metrics'
frame=[]
for row in plan:
 sl=next(sl for sl in P['slides'] if sl['objectId']==row['slideId']);frame.append(dict(number=row['number'],slideId=row['slideId'],narrative=row['title'],exemplar=sl['objectId'],construction='edit inspected native slide in place; official master and layout preserved',elementMap={e['objectId']:changes.get(e['objectId'],'keep source content/media and geometry') for e in sl['pageElements']},addedElements=[oid for oid in changes if oid.startswith(sl['objectId']) and oid not in all_elements]))
(W/'content.json').write_text(json.dumps(req,ensure_ascii=False,indent=2));(W/'images.json').write_text(json.dumps(images,ensure_ascii=False,indent=2));(W/'frame-map.json').write_text(json.dumps(frame,ensure_ascii=False,indent=2))
(W/'reorder.json').write_text(json.dumps([{'updateSlidesPosition':{'slideObjectIds':['ifsc26_s08'],'insertionIndex':4}},{'updateSlidesPosition':{'slideObjectIds':['ifsc26_s11'],'insertionIndex':9}}]))
(W/'source-notes.txt').write_text('Official unmodified logo: https://ifsc.edu.br/documents/d/comunicacao/florianopolis linked by https://ifsc.edu.br/web/campus-florianopolis/identidade-visual. Original PNG 662x965 RGBA; sha256 '+hashlib.sha256(Path(logo).read_bytes()).hexdigest()+'\nAll existing photographs retained except campus photo explicitly replaced. Existing generated scientific figures are measured data, not illustrations. New charts are derived from actual metrics. Diagram elements remain native.\n')
(W/'fallback-reasons.txt').write_text('Existing native Google Slides edited in place. Managed dependency loader unavailable; native connector, provided Node helpers and installed Poppler used. No PPTX roundtrip. Marker run in presentations skill directory. Bounded legacy PDF compatibility is reused because streamed file references lack workspace_path; inline PDF stays below32MB and is not printed.\n')
print('Requests',len(req),'images',len(images),'slides',len(plan))
