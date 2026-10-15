"""Map every existing IFSC slide to the measured ternary study; native API requests only."""
from pathlib import Path
import copy
import json

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
P = json.loads((W / 'presentation-before.json').read_text())
S = {s['objectId']:s for s in P['slides']}
OLD = {s['slideId']:s for s in json.loads((ROOT / 'tmp/ifsc_enxuta_2026_10_15/plan.json').read_text())}
R = ROOT / 'results/ternary_lstm_2026_10_15'
MET = json.loads((R / 'metrics.json').read_text())
requests, images, element_map, plan = [], [], {}, []
G = {'red':39/255, 'green':131/255, 'blue':51/255}
INK = {'red':.07, 'green':.12, 'blue':.08}
PALE = {'red':.93, 'green':.96, 'blue':.93}
WHITE = {'red':1, 'green':1, 'blue':1}
REPO = 'https://github.com/Lciarallo/mosquito-wingbeat'
TORRES = 'https://latamt.ieeer9.org/index.php/transactions/article/view/9221'


def replace(sid, oid, text):
    e = next(e for e in S[sid]['pageElements'] if e['objectId'] == oid)
    styles = [t['textRun'].get('style', {}) for t in e['shape']['text']['textElements'] if 'textRun' in t]
    assert all(st == styles[0] for st in styles), ('mixed styles need explicit ranges', oid)
    style = copy.deepcopy(styles[0]); style.pop('link', None)
    if 'weightedFontFamily' in style: style.pop('fontFamily', None)
    style = {k:v for k,v in style.items() if v != {}}
    requests.extend([{'deleteText':{'objectId':oid, 'textRange':{'type':'ALL'}}},
        {'insertText':{'objectId':oid,'insertionIndex':0,'text':text}},
        {'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':style,'fields':','.join(style)}}])
    element_map[oid] = {'action':'replace native text', 'style':'source uniform run preserved', 'text':text}


def props(sid, x, y, width, height):
    return {'pageObjectId':sid, 'size':{'width':{'magnitude':width,'unit':'PT'},'height':{'magnitude':height,'unit':'PT'}},
            'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}


def box(sid, oid, text, x, y, width, height, font=20, fill=None, color=INK, align='CENTER'):
    requests.append({'createShape':{'objectId':oid,'shapeType':'ROUND_RECTANGLE' if fill else 'TEXT_BOX',
                                   'elementProperties':props(sid,x,y,width,height)}})
    requests.append({'updateShapeProperties':{'objectId':oid,'shapeProperties':{
        'shapeBackgroundFill':{'solidFill':{'color':{'rgbColor':fill},'alpha':1}} if fill else {'propertyState':'NOT_RENDERED'},
        'outline':{'propertyState':'NOT_RENDERED'}, 'contentAlignment':'MIDDLE'},
        'fields':'shapeBackgroundFill,outline,contentAlignment'}})
    requests.extend([{'insertText':{'objectId':oid,'text':text,'insertionIndex':0}},
        {'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{
            'fontFamily':'Oswald','fontSize':{'magnitude':font,'unit':'PT'},'foregroundColor':{'opaqueColor':{'rgbColor':color}}},
            'fields':'fontFamily,fontSize,foregroundColor'}},
        {'updateParagraphStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'alignment':align,'spaceAbove':{'magnitude':0,'unit':'PT'},
            'spaceBelow':{'magnitude':0,'unit':'PT'}},'fields':'alignment,spaceAbove,spaceBelow'}}])


def arrow(sid, oid, x, y, length, both=False):
    requests.extend([{'createLine':{'objectId':oid,'category':'STRAIGHT','elementProperties':props(sid,x,y,length,0)}},
       {'updateLineProperties':{'objectId':oid,'lineProperties':{
           'lineFill':{'solidFill':{'color':{'rgbColor':G},'alpha':1}},'weight':{'magnitude':1.7,'unit':'PT'},
           'endArrow':'FILL_ARROW','startArrow':'FILL_ARROW' if both else 'NONE'},
           'fields':'lineFill,weight,endArrow,startArrow'}}])


def remove_image(sid, reason):
    oid = sid+'_e02'
    requests.append({'deleteObject':{'objectId':oid}})
    element_map[oid] = {'action':'replace with native editable structure','reason':reason}


def figure(sid, name):
    path = str(W / 'assets' / name)
    images.append({'path':path,'requests':[{'replaceImage':{'imageObjectId':sid+'_e02','url':path,'imageReplaceMethod':'CENTER_INSIDE'}}]})
    element_map[sid+'_e02'] = {'action':'replace scientific chart','path':path,'fit':'CENTER_INSIDE; full chart, no crop'}


def slide(sid, number, title, caption, seconds, notes, sources=()):
    if title is not None: replace(sid, sid+'_e00', title)
    if caption is not None: replace(sid, sid+'_e01', caption)
    for e in S[sid]['pageElements']:
        if 'shape' in e:
            text = ''.join(t.get('textRun',{}).get('content','') for t in e['shape'].get('text',{}).get('textElements',[])).strip()
            if text.isdigit() and int(text) <= 20 and not (sid in ['ifsc26_s20','ifsc26_regularizacao'] and e['objectId'] not in [sid+'_e01']):
                replace(sid, e['objectId'], f'{number:02d}')
    sources = list(dict.fromkeys([REPO, 'Arquivo local: results/ternary_lstm_2026_10_15/metrics.json',
        'Arquivo local: reports/lstm_ternaria_2026_10_15.md', *sources]))
    full = f'Tempo sugerido: {seconds} segundos.\n\n{notes}\n\n[Sources]\n'+'\n'.join(sources)+'\n[/Sources]'
    oid = S[sid]['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
    requests.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},
                     {'insertText':{'objectId':oid,'insertionIndex':0,'text':full}}])
    plan.append(dict(slideId=sid,number=number,title=title,caption=caption,seconds=seconds,notes=notes,notes_full=full,sources=sources))


# Every photo, image, placeholder and source decoration has a disposition before mutation.
for s in P['slides']:
    for e in s['pageElements']:
        element_map[e['objectId']] = {'action':'keep source element','kind':next((k for k in ['shape','image','line','table','elementGroup'] if k in e),'other')}

slide('ifsc26_s01',1,None,None,25,
      'Apresentar o projeto de Luiz Eduardo do Nascimento Ciarallo para Computação Científica para a Indústria, IFSC. O problema é distinguir Aedes aegypti, outro mosquito e ausência de mosquito em áudio. A apresentação compara um detector clássico implementado com duas redes recorrentes profundas também implementadas.',
      OLD['ifsc26_s01']['sources'][:3])
replace('ifsc26_s01','ifsc26_s01_e02','Detecção de Aedes aegypti\ncom redes recorrentes')
slide('ifsc26_s02',2,'Uma entrada de áudio, três respostas',None,50,
      'As asas geram padrões periódicos que podem ser analisados no áudio. A saída necessária tem três classes: Aedes aegypti, outro mosquito ou ruído sem mosquito. A fotografia é de um Aedes aegypti real. Os rótulos são de detecção acústica da espécie, sem inferir infecção. A decisão final é por grupo de gravação, reunindo suas janelas.', OLD['ifsc26_s02']['sources'])
replace('ifsc26_s02','ifsc26_s02_e02','Objetivo do detector:\n\nAedes aegypti\nOutro mosquito\nRuído / sem mosquito')
replace('ifsc26_s02','ifsc26_s02_e04','Detectar o alvo e separar o ruído.')
slide('ifsc26_s03',3,'Três estudos orientaram a proposta',None,55,
      'Mukundarajan e colaboradores contextualizam a coleta acústica com celulares. Torres e colaboradores empregam redes LSTM e bidirecionais com Monte Carlo dropout para outra tarefa, a identificação de mosquitos infectados. Aqui, essas técnicas foram adaptadas e treinadas para as três classes do detector. Supratak e colaboradores ajudam a discutir robustez ao ruído. O conjunto MosquitoSong é apenas relacionado e permanece fora do treinamento deste detector.',
      ['https://elifesciences.org/articles/27854',TORRES,'https://github.com/NakanoMariko/Dengue-Infected-Mosquito-Detection','https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121'])
replace('ifsc26_s03','ifsc26_s03_e04','Torres · 2025')
replace('ifsc26_s03','ifsc26_s03_e05','LSTM e incerteza')
replace('ifsc26_s03','ifsc26_s03_e11','Da captura acústica à modelagem temporal e à robustez ao ruído.')
path = str(W/'assets/torres_artigo.png')
images.append({'path':path,'requests':[{'replaceImage':{'imageObjectId':'ifsc26_s03_e13','url':path,'imageReplaceMethod':'CENTER_INSIDE'}}]})
element_map['ifsc26_s03_e13'] = {'action':'replace paper thumbnail','source':'actual first page of attached Torres 2025 PDF','path':path}
slide('ifsc26_s04',4,'O sinal revela padrões no tempo e na frequência',
      'Trecho real de Aedes · 0,992 s · áudio a 8 kHz · espectro do mesmo trecho.',45,
      'À esquerda está a primeira janela retida de Aedes no corpus auditado, sem seleção pelo resultado de um modelo. À direita, o módulo da FFT do mesmo trecho, com janela de Hann e nível relativo ao pico. A forma de onda mostra variações de amplitude; o espectro revela componentes periódicas e harmônicos. A frequência de amostragem do cache utilizado é oito quilohertz.',
      ['Arquivo local: output/ifsc_ternario_lstm_2026_10_15/assets/audio-example.json'])
figure('ifsc26_s04','onda_fft.png')
slide('ifsc26_s05',5,'A frequência das asas ajuda a caracterizar o voo',
      'Valores publicados a 25 °C · fêmeas e machos · Sinka et al. (2021).',50,
      'O gráfico mantém as frequências médias publicadas por Sinka e colaboradores, a 25 graus Celsius, para fêmeas e machos. Para Aedes aegypti, os valores mostrados são 458 e 715 hertz. O padrão de vários harmônicos e sua evolução temporal motivam o uso do espectrograma, além de uma única frequência. Estes números são referências da literatura, não frequências medidas em todos os arquivos do projeto.',
      OLD['ifsc26_s05']['sources'][:3])
slide('ifsc26_s06',6,'A STFT acompanha o espectro ao longo do voo',
      'FFT 1.024 · Hann 128 ms · avanço 16 ms · separação entre bins: 7,8125 Hz.',55,
      'A STFT aplica uma FFT em janelas sucessivas, preservando a posição temporal das componentes. Nesta execução o áudio tem oito quilohertz, com 1.024 amostras por Hann, equivalentes a 128 milissegundos. O avanço é de 128 amostras, ou 16 milissegundos. O quociente 8.000/1.024 dá 7,8125 hertz entre bins. O trecho de 7.936 amostras produz 55 passos com center false; o desenho exibe as frequências até 2.500 hertz para facilitar a leitura.')
figure('ifsc26_s06','stft.png')
slide('ifsc26_s07',7,'Do espectrograma à sequência mel',
      '64 filtros · 55 passos · 80–4.000 Hz · normalização ajustada apenas no treino.',55,
      'O frontend transforma a potência da STFT em 64 bandas mel entre oitenta e quatro mil hertz. A sequência tem 55 passos temporais e 64 atributos por passo. Remove-se a média e normaliza-se o RMS de cada janela; depois a potência é expressa em decibéis relativos e limitada a menos oitenta até zero. A média e o desvio por banda usados para padronizar a entrada são calculados exclusivamente no treino de cada ajuste. A matriz exibida é anterior à padronização estatística por banda.')
figure('ifsc26_s07','mel.png')
slide('ifsc26_s08',8,'Mesma população e cinco testes externos',
      '42.029 janelas · 1.494 grupos · 1.441 blocos de PCM · três classes.',70,
      'A comparação conserva a mesma população e as cinco dobras externas da cascata auditada. São 42.029 janelas de 0,992 segundo, reunidas em 1.494 grupos: 426 de ruído, 1.008 de outros mosquitos e 60 de Aedes. Nenhum dos 1.441 blocos de conteúdo PCM atravessa treino e teste. Em cada treino externo, uma divisão interna agrupada escolhe época e limiares; o modelo é então reajustado em todo o treino externo, congelado e aplicado ao teste. A seleção das redes usa um holdout interno, a primeira divisão de quatro partes; a cascata usa calibração interna OOF mais ampla. Portanto, as cinco avaliações externas e a população são iguais, e os processos internos de seleção são descritos separadamente. MosquitoSong não participa de nenhum ajuste.')
figure('ifsc26_s08','populacao.png')
slide('ifsc26_s09',9,'Redes recorrentes para as três classes',
      'Duas camadas · 128 unidades por direção · dropout 0,20 · três saídas.',85,
      'A LSTM aprende dependências entre quadros mel consecutivos, com portas que atualizam sua memória. A versão unidirecional percorre o tempo em um sentido; a Bi-LSTM processa a sequência nos dois sentidos. As duas têm duas camadas de 128 unidades por direção, dropout de 0,20 entre camadas e uma projeção linear do último passo para três escores. O softmax produz probabilidades. O projeto adapta o artigo de Torres, que usava 256 unidades por camada em outra tarefa. A LSTM tem 231.811 parâmetros, e a bidirecional 594.691. A agregação das janelas em um grupo usa pesos de atividade iguais a um menos probabilidade de ruído, elevado a 1,5.',
      [TORRES,'Arquivo local: torres_lstm.py; train_ternary_lstm.py'])
remove_image('ifsc26_s09','old CNN diagram is outside the ternary recurrent comparison')
for row, y, both in [('lstm',135,False),('bilstm',255,True)]:
    for k in range(3): arrow('ifsc26_s09',f'tern_{row}_edge{k}',151+k*137,y+28,19)
for row, y, label, mid in [('lstm',135,'LSTM · um sentido','LSTM'),('bilstm',255,'Bi-LSTM · dois sentidos','Bi-LSTM')]:
    box('ifsc26_s09',f'tern_{row}_label',label,29,y-35,540,30,21,align='START')
    for k, text in enumerate(['Sequência mel\n55 × 64',mid+('\n128 × 2' if row=='bilstm' else '\n128 unidades'),mid+('\n128 × 2' if row=='bilstm' else '\n128 unidades'),'Linear (3)\nsoftmax']):
        box('ifsc26_s09',f'tern_{row}_node{k}',text,33+k*137,y,118,57,18,fill=G if k in [1,2] else PALE,color=WHITE if k in [1,2] else INK)
slide('ifsc26_s11',10,'A cascata clássica é a referência de comparação',
      'LR + HistGB · probabilidades por janela · agregação por atividade · dois limiares.',45,
      'O detector existente combina probabilidades de regressão logística e HistGradientBoosting. O agrupamento temporal dá maior peso às janelas ativas. Um primeiro limiar separa ruído de mosquito; um segundo decide Aedes versus outro mosquito. A composição e os limiares da referência foram definidos na calibração interna, antes de cada teste externo. Essa implementação é a fonte dos 90,03 por cento de acurácia geral, e serve como comparação para as redes profundas.',
      ['Arquivo local: ternary_protocol.py; results/ternary_verified_2026_10_14_corrections/event_level_metrics.json'])
remove_image('ifsc26_s11','old 20-species MLP no longer explains the detector')
for k in range(3): arrow('ifsc26_s11',f'tern_classic_edge{k}',151+k*137,174,19)
for k, text in enumerate(['LR + HistGB\npor janela','Agregação\npor atividade','Gate 1\nruído / mosquito','Gate 2\nAedes / outro']):
    box('ifsc26_s11',f'tern_classic_node{k}',text,33+k*137,140,118,68,18,fill=G if k==0 else PALE,color=WHITE if k==0 else INK)
box('ifsc26_s11','tern_classic_result','1.345 acertos em 1.494 grupos\n90,03% de acurácia geral',45,248,520,69,29,fill=None)
slide('ifsc26_s12',11,'Adam ajusta os pesos das redes',
      'Taxa 0,0006 · lotes de 256 · norma do gradiente limitada a 3.',65,
      'A entropia cruzada mede o erro das três saídas. A retropropagação calcula gradientes; o clipping limita a norma global em três. O Adam mantém estimativas dos primeiros e segundos momentos para atualizar os pesos com taxa de 0,0006. Cada lote contém 256 janelas amostradas com reposição e pesos equilibrados. O treino pode chegar a trinta épocas, com parada quando a perda de validação não melhora durante cinco épocas. Esses são os valores efetivamente executados; esta comparação não usa AdamW, scheduler cosseno ou label smoothing. Os hiperparâmetros foram fixados antes da leitura das métricas externas, sem uma busca retrospectiva pelo melhor teste.')
remove_image('ifsc26_s12','replace stale AdamW/cosine claims with executed Adam optimization')
for k in range(3): arrow('ifsc26_s12',f'tern_adam_edge{k}',151+k*137,174,19)
for k, text in enumerate(['Entropia\ncruzada','Gradientes\nclipping = 3','Adam\n2 momentos','Atualização\ndos pesos']):
    box('ifsc26_s12',f'tern_adam_node{k}',text,33+k*137,140,118,68,19,fill=G if k==2 else PALE,color=WHITE if k==2 else INK)
box('ifsc26_s12','tern_adam_info','Taxa de aprendizado: 0,0006\n256 janelas por lote · até 30 épocas',55,245,510,70,25)
slide('ifsc26_regularizacao',12,'Regularização e seleção durante o treino',None,60,
      'O dropout de vinte por cento atua entre as duas camadas recorrentes. A amostragem ponderada mantém massa igual por classe e usa amortecimento gamma de 0,50 por duração de gravação, com teto de sessenta janelas no fator de peso. A época é escolhida pela menor entropia cruzada de grupos, equilibrada por classe, na validação interna. A paciência é cinco. O normalizador também é calculado apenas no conjunto utilizado para ajustar cada rede. Na inferência de WAVs, Monte Carlo dropout executa cem passagens e informa a discordância de decisões; os resultados OOF apresentados adiante usam a inferência determinística.')
for suffix,text in [('e03','Dropout entre camadas'),('e04','Probabilidade 0,20 nas duas arquiteturas.'),('e06','Amostragem por classe e gravação'),('e07','Gamma 0,50 · mesma massa para as três classes.'),('e09','Parada pela validação interna'),('e10','Menor perda por grupo · paciência de 5 épocas.'),('e11','Monte Carlo dropout: 100 passagens na inferência de um novo áudio.')]:
    replace('ifsc26_regularizacao','ifsc26_regularizacao_'+suffix,text)
slide('ifsc26_s14',13,'A validação escolhe a época de cada rede',
      'Ajuste final · LSTM: época 4 · Bi-LSTM: época 2 · decisão pela validação interna.',55,
      'As curvas são os históricos reais da seleção interna no ajuste final, separada dos cinco testes externos. A LSTM escolheu a época quatro; a Bi-LSTM, a época dois. A linha tracejada marca a melhor perda de validação. O treino e a validação usam níveis de agregação diferentes: janelas amostradas no treino e grupos equilibrados por classe na validação; o gráfico não deve ser interpretado como duas perdas da mesma população. Depois da seleção, cada rede é reinicializada e reajustada sobre todo o desenvolvimento pela quantidade escolhida de épocas. A LSTM final foi escolhida pela menor perda interna, sem usar a acurácia externa para escolher a arquitetura.')
figure('ifsc26_s14','aprendizado.png')
slide('ifsc26_s15',14,'A cascata mantém 90,03% de acurácia geral',
      'Mesmos 1.494 grupos · direta: maior probabilidade · calibrada: dois limiares internos.',65,
      'A linha tracejada marca os 90,03 por cento da cascata clássica. A decisão direta das redes usa a classe de maior probabilidade agregada. A LSTM direta alcançou 85,74 por cento e a bidirecional 82,33. Aplicando os dois limiares selecionados internamente, as acurácias foram 86,55 e 85,27 por cento. Portanto, as duas arquiteturas treinadas não mantiveram a acurácia geral da referência. A próxima tabela mostra como a mudança de regra afeta a sensibilidade de Aedes e os falsos positivos. Todas as configurações são mostradas, sem escolher apenas o modo que teve maior acurácia no teste.')
figure('ifsc26_s15','acuracia.png')
slide('ifsc26_s10',15,'Sensibilidade e falsos positivos mudam juntos',
      'Sensibilidade: Aedes detectados · FPR: negativos classificados como Aedes.',95,
      'A cascata detectou 47 de 60 Aedes, com zero ruídos e 23 outros mosquitos classificados como Aedes. Na decisão direta, a LSTM detectou 52 de 60 e a Bi-LSTM 53 de 60, com mais falsos positivos em ambas as classes negativas. A calibração interna maximiza a sensibilidade respeitando tetos empíricos de falso positivo no holdout de treino. Nos testes, a LSTM calibrada detectou 20 de 60 e a Bi-LSTM 17 de 60. A precisão das redes diretas para Aedes é de 48,15 e 38,69 por cento, respectivamente. A comparação demonstra o compromisso entre detectar mais alvos e emitir mais alarmes; não se deve interpretar uma melhora isolada de acurácia após calibração como melhora de detecção de Aedes. Os limites internos não garantem limites populacionais. As contagens e as duas regras foram preservadas em metrics.json.')
remove_image('ifsc26_s10','old CNN activation is replaced by a measured, native comparison table')
tid = 'tern_metrics_table'
requests.append({'createTable':{'objectId':tid,'rows':6,'columns':4,'elementProperties':props('ifsc26_s10',29,91,560,240)}})
for col,width in enumerate([185,125,125,125]):
    requests.append({'updateTableColumnProperties':{'objectId':tid,'columnIndices':[col],
        'tableColumnProperties':{'columnWidth':{'magnitude':width,'unit':'PT'}},'fields':'columnWidth'}})
table = [['Modelo / regra','Sensibilidade\nAedes','FPR ruído\npara Aedes','FPR outro\npara Aedes']]
for r in [dict(MET['reference'],label='Cascata clássica'),*[dict(r,label=('LSTM' if r['model']=='lstm' else 'Bi-LSTM')+(' · direta' if r['decision']=='argmax' else ' · calibrada')) for r in MET['neural_models']]]:
    table.append([r['label']]+[f'{100*r[k]:.2f}%'.replace('.',',') for k in ['aedes_recall','noise_to_aedes_fpr','other_to_aedes_fpr']])
for row,cells in enumerate(table):
    for col,text in enumerate(cells):
        cell = {'rowIndex':row,'columnIndex':col}
        requests.extend([{'insertText':{'objectId':tid,'cellLocation':cell,'text':text,'insertionIndex':0}},
            {'updateTextStyle':{'objectId':tid,'cellLocation':cell,'textRange':{'type':'ALL'},'style':{
                'fontFamily':'Oswald','fontSize':{'magnitude':17,'unit':'PT'},'foregroundColor':{'opaqueColor':{'rgbColor':WHITE if row==0 else INK}}},
                'fields':'fontFamily,fontSize,foregroundColor'}},
            {'updateParagraphStyle':{'objectId':tid,'cellLocation':cell,'textRange':{'type':'ALL'},'style':{
                'alignment':'START' if col==0 else 'CENTER','spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':0,'unit':'PT'}},
                'fields':'alignment,spaceAbove,spaceBelow'}},
            {'updateTableCellProperties':{'objectId':tid,'tableRange':{'location':cell,'rowSpan':1,'columnSpan':1},
                'tableCellProperties':{'tableCellBackgroundFill':{'solidFill':{'color':{'rgbColor':G if row==0 else WHITE if row%2==0 else PALE},'alpha':1}},'contentAlignment':'MIDDLE'},
                'fields':'tableCellBackgroundFill,contentAlignment'}}])
slide('ifsc26_s17',16,'Onde a cascata acerta e onde confunde',
      '1.345/1.494 acertos · 47/60 Aedes detectados · linhas: real · colunas: previsão.',70,
      'A matriz de confusão mostra a referência clássica nas três classes. A diagonal soma 1.345 acertos em 1.494 grupos. Entre sessenta Aedes, quarenta e sete foram detectados e treze classificados como outro mosquito. Nenhum foi perdido para ruído. Entre os ruídos, zero foram chamados de Aedes, mas cento e dois foram chamados de outro mosquito. Portanto, falso alarme para Aedes e falso alarme para qualquer mosquito são medidas diferentes. A sensibilidade de Aedes é 78,33 por cento, a precisão é 67,14 por cento e a acurácia equilibrada entre classes é 83,67 por cento.')
figure('ifsc26_s17','confusao.png')
slide('ifsc26_s19',17,'Referências e créditos',None,25,
      'As referências e os créditos incluem os artigos usados na motivação, na modelagem temporal e na caracterização acústica, os dados e o código do projeto, o modelo oficial e a fotografia do campus IFSC, e a fotografia do mosquito do CDC. Os links completos e a origem de cada gráfico estão nas notas. Os resultados das redes são deste projeto, sem transferir para o detector os números do artigo de infecção.',
      ['https://elifesciences.org/articles/27854',TORRES,'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121','https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13663',*OLD['ifsc26_s02']['sources']])
replace('ifsc26_s19','ifsc26_s19_e03','Torres et al. (2025) · LSTM, Bi-LSTM e MC dropout · IEEE')
slide('ifsc26_s20',18,'Discussão final: limites e próximo ciclo',None,75,
      'O detector clássico teve um resultado positivo de 90,03 por cento, e as redes recorrentes foram implementadas e comparadas de fato, como exige o trabalho. As duas redes diretas aumentaram a sensibilidade de Aedes, acompanhada de mais falsos positivos. As redes calibradas reduziram falsos positivos com redução importante da sensibilidade. Esta é uma avaliação retrospectiva de um corpus já conhecido, não uma validação independente em campo. A comparação tem as mesmas dobras externas, mas diferentes processos internos de seleção, e não há ablação para atribuir ganho causal a cada técnica. MosquitoSong continua separado como fonte externa. O próximo ciclo deve testar novas sessões, indivíduos, microfones e níveis de ruído, mantendo pesos e decisões congelados. Não foi demonstrada inferência de infecção por dengue, nem execução dessas redes em Arduino. A conclusão é que a implementação profunda cumpre o objetivo acadêmico e que a cascata permanece a referência de acurácia geral desta avaliação.')
for suffix,text in [('e03','Resultado e compromisso'),('e04','90,03% na cascata; redes diretas detectam mais Aedes.'),('e06','Avaliação retrospectiva'),('e07','Novas sessões e indivíduos ainda precisam ser avaliados.'),('e09','Próximo ciclo: robustez'),('e10','Congelar o detector e variar ruído, ambiente e microfone.'),('e11','Comparação sem ablação causal · MosquitoSong separado · redes ainda sem porte embarcado.')]:
    replace('ifsc26_s20','ifsc26_s20_'+suffix,text)

assert len(plan)==18 and {s['slideId'] for s in plan}==set(S)
assert 15*60 <= sum(s['seconds'] for s in plan) <= 20*60
assert len(element_map)==sum(len(s['pageElements']) for s in P['slides'])
for name,value in [('content',requests),('images',images),('plan',plan),('frame-map',dict(
    nativePresentationId=P['presentationId'],nativeMastersPreserved=True,nativeLayoutsPreserved=True,
    sourceRevision=P['revisionId'],pageSize=P['pageSize'],everySourceElementMapped=True,
    elementMap=element_map,finalOrder=[s['slideId'] for s in plan]))]:
    (W/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
(W/'reorder.json').write_text(json.dumps([{'updateSlidesPosition':{'slideObjectIds':['ifsc26_s10'],'insertionIndex':15}}]))
(W/'Roteiro_apresentacao_IFSC.txt').write_text('\n\n'.join(f"{s['number']:02d}. {s['title'] or 'Capa'} ({s['seconds']} s)\n{s['notes']}" for s in plan)+f"\n\nDuração sugerida: {sum(s['seconds'] for s in plan)/60:.2f} minutos.\n")
criteria = '''# Critérios do trabalho e revisão da apresentação

Autor: Luiz Eduardo do Nascimento Ciarallo. Tema: detecção acústica ternária de Aedes aegypti.

Fonte: Trabalho_Descrição-1.pdf, páginas 2–4, Computação Científica para a Indústria, IFSC Florianópolis.

| Exigência | Evidência | Slides |
|---|---|---|
| Estudo de técnicas e programa com rede neural profunda | LSTM/Bi-LSTM de duas camadas treinadas; scripts, pesos, previsões, teste de inferência e verificador | 6–13 |
| Contextualização da aplicação | Voo, detecção de Aedes e três saídas | 1–2 |
| Abordagens comuns e trabalhos correlatos | Mukundarajan, Torres, Supratak, Sinka e cascata clássica | 3, 5, 10, 17 |
| Apresentação da proposta implementada | STFT, mel, partições, arquitetura, otimização e regularização | 4–13 |
| Resultados obtidos | Curvas reais, comparação pareada de população, sensibilidade/FPR e matriz de confusão | 13–16 |
| Discussão final | Interpretação, limites, testes futuros | 18 |
| Equipe com no máximo dois integrantes | Um autor identificado por nome completo | 1 |
| Apresentação de 15–20 minutos | Roteiro com tempos por slide | 18 slides; 18 minutos aproximadamente |
| Entrega da apresentação ao professor | Slides nativos e PDF para entrega pelo autor | Artefatos de saída; envio não solicitado |

O nome, a foto real do IFSC e a identidade do modelo oficial são preservados. Os 90,03% são atribuídos somente à cascata clássica. As redes têm resultados próprios para as três classes; não são substituídos pelos resultados das 20 espécies nem pela tarefa de infecção do artigo.

Os valores de Adam, dropout, amostragem e parada são os executados, sem alegar otimização causal sem ablação. As limitações estão concentradas no encerramento.
'''
(W/'criterios_apresentacao.md').write_text(criteria)
(W/'fallback-reasons.txt').write_text('Existing native Slides edited in place via Google Drive connector; official masters/layouts preserved. Workspace dependency loader is not available in this session. Native read-to-file/parse/check/render helpers and local Poppler are used. The presentation operation marker supports pptx, not native Google Slides; the edit marker is recorded using its supported pptx format, with the actual authoring format remaining native Slides.\n')
print('Ready:',len(plan),'slides;',len(requests),'native content requests;',len(images),'image replacements;',sum(s['seconds'] for s in plan)/60,'minutes')
