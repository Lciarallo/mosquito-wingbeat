from pathlib import Path
import json,copy,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;old=W.parent/'ifsc_didatica_2026_10_15';P=json.loads((W/'raw-before.json').read_text())['structuredContent'];els={e['objectId']:e for s in P['slides'] for e in s['pageElements']};req=[]
new='ifsc26_conceitos_modelos';source=next(s for s in P['slides'] if s['objectId']=='ifsc26_s03');mapping={'ifsc26_s03':new,**{e['objectId']:e['objectId'].replace('ifsc26_s03',new) for e in source['pageElements']}}
struct=[{'duplicateObject':{'objectId':'ifsc26_s03','objectIds':mapping}},{'deleteObject':{'objectId':'ifsc26_s12'}},{'deleteObject':{'objectId':'ifsc26_regularizacao'}}]
for e in source['pageElements']:els[mapping[e['objectId']]]=copy.deepcopy(e)
def replace(oid,text):
 styles=[t['textRun'].get('style',{}) for t in els[oid]['shape']['text']['textElements'] if 'textRun'in t];assert all(s==styles[0] for s in styles),oid
 st=copy.deepcopy(styles[0]);st.pop('link',None)
 if 'weightedFontFamily'in st:st.pop('fontFamily',None)
 st={k:v for k,v in st.items() if v!={}}
 req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'text':text,'insertionIndex':0}},{'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':st,'fields':','.join(st)}}])
def geometry(oid,x,y,w,h):
 e=els[oid];sz=e['size'];factor=12700 if sz['width']['unit']=='EMU' else 1
 req.append({'updatePageElementTransform':{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w/(sz['width']['magnitude']/factor),'scaleY':h/(sz['height']['magnitude']/factor),'translateX':x,'translateY':y,'unit':'PT'}}})
texts={
'ifsc26_s04_e00':'Análise do áudio: sinal e FFT',
'ifsc26_s05_e00':'Frequência de voo por espécie',
'ifsc26_s06_e00':'Espectrograma: frequência ao longo do tempo',
'ifsc26_s06_didatica_subtitulo':'A STFT calcula as frequências de pequenos trechos do áudio.',
'ifsc26_s06_e01':'Horizontal: tempo · vertical: frequência · cor: intensidade do som.',
'ifsc26_s07_e00':'Como o áudio é preparado para a rede',
'ifsc26_s07_didatica_subtitulo':'Agrupamos as frequências em bandas e acompanhamos sua intensidade.',
'ifsc26_s07_e01':'Linhas: 64 bandas mel · colunas: 55 instantes · cor: intensidade.',
'ifsc26_s11_didatica_subtitulo':'Os modelos analisam trechos; os resultados são combinados por gravação.',
'ifsc26_s11_e01':'LR = regressão logística · HistGB = árvores em boosting.\nAgregação = média das probabilidades, com mais peso aos trechos ativos.',
 'tern_classic_node1':'Combinação\ndos trechos','tern_classic_node2':'Etapa 1\nHá mosquito?','tern_classic_node3':'Etapa 2\nÉ Aedes?',
'ifsc26_s09_e00':'Como a LSTM e a Bi-LSTM analisam o áudio',
'ifsc26_s09_didatica_subtitulo':'A rede acompanha a sequência do som para reconhecer seu padrão.',
 'tern_lstm_label':'LSTM · considera o que veio antes','tern_bilstm_label':'Bi-LSTM · analisa o trecho nos dois sentidos',
'ifsc26_s09_e01':'LSTM: um sentido · Bi-LSTM: dois sentidos · ambas usam duas camadas.',
'ifsc26_s14_e00':'Evolução do erro durante o treinamento',
'ifsc26_s14_didatica_subtitulo':'A menor perda na validação define a época utilizada.',
'ifsc26_s15_e01':'Maior acurácia observada por arquitetura · mesmos 1.494 grupos.',
'ifsc26_s10_e00':'Sensibilidade e falsos positivos',
'ifsc26_s20_e04':'Cascata: maior acurácia geral na comparação.',
new+'_e00':'Três formas de analisar o áudio',new+'_e11':'Cascata: modelos clássicos · LSTM e Bi-LSTM: redes neurais profundas.'}
for prefix in ['tern_lstm','tern_bilstm']:
 texts[prefix+'_node0']='Áudio\npreparado';texts[prefix+'_node3']='Decisão\n3 classes'
 texts[prefix+'_node1']=('Bi-LSTM' if 'bilstm'in prefix else'LSTM')+'\ncamada 1';texts[prefix+'_node2']=('Bi-LSTM' if 'bilstm'in prefix else'LSTM')+'\ncamada 2'
for oid,t in texts.items():replace(oid,t)
for k,e in enumerate(['e12','e13','e14']):req.append({'deleteObject':{'objectId':new+'_'+e}})
for k,(title,body) in enumerate([('Cascata clássica','Combina modelos\nclássicos em etapas.\nSepara mosquito,\nAedes e ruído.'),('LSTM','Rede neural\ncom memória.\nAnalisa o áudio\nem sequência.'),('Bi-LSTM','Rede neural que lê\no trecho nos dois\nsentidos para reunir\nmais contexto.')]):
 x=33+187*k;bg=new+'_square'+str(k)
 req.extend([{'createShape':{'objectId':bg,'shapeType':'RECTANGLE','elementProperties':{'pageObjectId':new,'size':{'width':{'magnitude':168,'unit':'PT'},'height':{'magnitude':168,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':125,'unit':'PT'}}}},{'updateShapeProperties':{'objectId':bg,'shapeProperties':{'shapeBackgroundFill':{'solidFill':{'color':{'rgbColor':{'red':.94,'green':.97,'blue':.94}},'alpha':1}},'outline':{'outlineFill':{'solidFill':{'color':{'rgbColor':{'red':.14,'green':.48,'blue':.18}},'alpha':1}},'weight':{'magnitude':1,'unit':'PT'}}},'fields':'shapeBackgroundFill,outline'}},{'updatePageElementsZOrder':{'pageElementObjectIds':[bg],'operation':'SEND_TO_BACK'}}])
 a=new+'_e'+str(2+2*k).zfill(2);b=new+'_e'+str(3+2*k).zfill(2);replace(a,title);replace(b,body);geometry(a,x+10,141,148,31);geometry(b,x+10,181,148,103)
# Delete the two unselected rows from the bottom upwards.
for row in [4,2]:req.append({'deleteTableRow':{'tableObjectId':'tern_metrics_table','cellLocation':{'rowIndex':row,'columnIndex':0}}})
for row,name in enumerate(['Clássica','LSTM','Bi-LSTM'],1):
 req.extend([{'deleteText':{'objectId':'tern_metrics_table','cellLocation':{'rowIndex':row,'columnIndex':0},'textRange':{'type':'ALL'}}},{'insertText':{'objectId':'tern_metrics_table','cellLocation':{'rowIndex':row,'columnIndex':0},'text':name,'insertionIndex':0}}])
rows=list(csv.DictReader(Path('results/ternary_lstm_2026_10_15/comparison.csv').open()));selected=[rows[0],rows[2],rows[4]]
(W/'assets').mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(10.8,4.77),facecolor='white');fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.20)
values=[100*float(r['accuracy']) for r in selected];names=['Clássica','LSTM','Bi-LSTM'];bars=ax.barh(names[::-1],values[::-1],color=['#75a17c','#469454','#24813b'][::-1],height=.55)
ax.set_xlim(0,100);ax.set_xlabel('Acurácia geral (%)',fontsize=15);ax.tick_params(axis='both',labelsize=14);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18)
for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
for bar,v in zip(bars,values[::-1]):ax.text(v+1,bar.get_y()+bar.get_height()/2,f'{v:.2f}%'.replace('.',','),va='center',fontsize=16,color='#15552a')
fig.savefig(W/'assets/acuracia_tres_modelos.png',dpi=160);plt.close(fig)
(W/'selection_summary.json').write_text(json.dumps({'criterion':'Maior acurácia geral observada por arquitetura; comparação retrospectiva de regras já avaliadas. Não equivale a uma nova seleção independente por validação.','selected_original_rows':selected,'visible_names':names},ensure_ascii=False,indent=2))
plan=[r for r in json.loads((old/'plan.json').read_text()) if r['slideId']not in ['ifsc26_s12','ifsc26_regularizacao']]
concept=dict(number=4,slideId=new,seconds=50,title=texts[new+'_e00'],concept='Três abordagens para classificar o áudio.',function='Situar os modelos antes dos detalhes.',why='A cascata oferece uma referência clássica; as redes aprendem sequências.',how='Cascata clássica em etapas, LSTM em um sentido e Bi-LSTM nos dois sentidos.',speech='Vamos comparar três abordagens. A cascata combina modelos clássicos em etapas. A LSTM é uma rede neural que acompanha o som em sequência. A Bi-LSTM lê o mesmo trecho nos dois sentidos, usando mais contexto.',transition='Essas escolhas foram orientadas pela literatura.',notes_full='')
plan.insert(3,concept)
source_note='\n[Sources]\nArquivo local: results/ternary_lstm_2026_10_15/comparison.csv\nArquivo local: train_ternary_lstm.py\nArquivo local: evaluate_ternary_event_pipeline.py\n[/Sources]'
for i,r in enumerate(plan,1):
 sid=r['slideId'];r['number']=i
 if sid+'_e00'in texts:r['title']=texts[sid+'_e00']
 if sid==new:r['notes_full']=r['speech']+source_note
 elif sid in ['ifsc26_s15','ifsc26_s10','ifsc26_s20']:
  r['speech']='A cascata alcançou 90,03% de acurácia geral, a LSTM 86,55% e a Bi-LSTM 85,27%. A tabela mostra as métricas da mesma configuração de cada modelo. Nessas configurações, a sensibilidade a Aedes foi 78,33%, 33,33% e 28,33%, respectivamente.'
  r['how']='Maior acurácia observada de cada arquitetura entre as regras já medidas. As duas redes usam suas regras com limiares ajustados. Seleção retrospectiva, sem novo teste independente.'
  r['notes_full']=r['speech']+'\nCritério: '+r['how']+'\nManter as métricas de cada linha juntas; não combinar sensibilidade de uma regra com acurácia de outra.'+source_note
 elif sid=='ifsc26_s14':
  r['speech']='O gráfico mostra como o erro mudou durante o treinamento. A escolha foi pela menor perda na validação. O treinamento parou após cinco épocas sem melhora. No contexto final, foram escolhidas a época quatro da LSTM e a época dois da Bi-LSTM.'
  r['notes_full']=r['speech']+'\nAdam, taxa 0,0006; lote 256; dropout 0,20; clipping 3; até 30 épocas. Mantidos no guia como apoio técnico. O método do cotovelo não foi utilizado.'+source_note
 elif sid=='ifsc26_s11':r['notes_full']+='\nEsclarecimento: etapa 1 separa ruído de mosquito; etapa 2 separa Aedes de outro mosquito. A agregação é uma média ponderada das probabilidades das janelas, com peso (1-P0)^1,5, e não voto majoritário.'
 # Source counters only, preserving template style.
 sl=source if sid==new else next(s for s in P['slides']if s['objectId']==sid)
 if i>1:
  counter=next(e['objectId'] for e in sl['pageElements'] if (t:=''.join(x.get('textRun',{}).get('content','')for x in e.get('shape',{}).get('text',{}).get('textElements',[])).strip()).isdigit() and len(t)==2)
  replace(mapping[counter] if sid==new else counter,f'{i:02d}')
 if sid!=new:
  oid=sl['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId'];req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':r['notes_full']}}])
(W/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));(W/'structure.json').write_text(json.dumps(struct));(W/'content.json').write_text(json.dumps(req,ensure_ascii=False));(W/'analise-lacunas.txt').write_text('Revisão oral autorizada: preservar apresentação aprovada e modelo oficial IFSC; substituir siglas sem explicação por termos naturais; inserir três quadrados introdutórios; remover slides Adam e regularização. Resultados: maior acurácia por arquitetura com métricas da mesma linha; registrar seleção retrospectiva nas notas. Não modificar dados/modelos. Exemplar novo: ifsc26_s03; títulos e textos reaproveitados; três capas de artigos removidas e substituídas por três quadrados solicitados. Demais slides mantêm seu exemplar original.\n')
print(len(req),'requests',len(plan),'slides',sum(r['seconds']for r in plan),'seconds')
