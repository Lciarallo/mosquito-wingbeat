from pathlib import Path
import copy,json

W=Path(__file__).resolve().parent
ROOT=W.parents[1]
P=json.loads((W/'raw-before.json').read_text())['structuredContent']
source=next(s for s in P['slides'] if s['objectId']=='ifsc26_s02')
SID='ifsc26_introducao_problema'
mapping={source['objectId']:SID,**{e['objectId']:f'{SID}_e{i:02d}' for i,e in enumerate(source['pageElements'])}}
duplicate=[{'duplicateObject':{'objectId':source['objectId'],'objectIds':mapping}}]
requests=[]
values={'e00':'Identificar Aedes para orientar a ação', 'e01':'02',
 'e02':'Hoje: vistorias e coleta.\nIdentificação morfológica\ncom lupa ou microscópio.\nÁudio: triagem automática\ndo voo para sinalizar Aedes\ne apoiar o monitoramento.',
 'e03':'Pneus e recipientes com água · SES-RS / SESA-ES',
 'e04':'Monitorar Aedes para apoiar a prevenção da dengue.'}
def native_replace(e,oid,value):
    styles=[t['textRun'].get('style',{}) for t in e['shape']['text']['textElements'] if 'textRun' in t]
    assert all(s==styles[0] for s in styles),'mixed source styles'
    st=copy.deepcopy(styles[0]);st.pop('link',None)
    if 'weightedFontFamily' in st:st.pop('fontFamily',None)
    st={k:v for k,v in st.items() if v!={}}
    requests.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},
        {'insertText':{'objectId':oid,'insertionIndex':0,'text':value}},
        {'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':st,'fields':','.join(st)}}])
for suffix,value in values.items():
    e=next(e for e in source['pageElements'] if e['objectId']=='ifsc26_s02_'+suffix)
    native_replace(e,mapping[e['objectId']],value)

prior=json.loads((ROOT/'output/ifsc_arduino_dataset_2026_10_15/plan.json').read_text())
notes='''Tempo sugerido: 65 segundos.

O ponto de partida é a vigilância do Aedes aegypti, vetor de arboviroses como a dengue. Recipientes que acumulam água, como pneus e pratos de plantas, podem funcionar como criadouros. As fotos mostram exemplos desses ambientes; não são locais detectados pelo nosso algoritmo.

Hoje, as equipes fazem vistorias, procuram criadouros e usam armadilhas para amostrar ovos e mosquitos. Larvas ou adultos coletados podem ser examinados por profissionais: a identificação morfológica compara características do corpo com auxílio de lupa ou microscópio e chaves de identificação. Essas informações orientam as ações de controle.

A contribuição proposta pelo projeto é acrescentar a análise automática do som do voo de mosquitos adultos. O algoritmo distingue Aedes aegypti, outros mosquitos e ruído. Integrado futuramente a um dispositivo de captura, poderá gerar registros e sinalizações para apoiar o monitoramento e a priorização de inspeções pelas equipes. É uma aplicação proposta para ampliar a informação disponível à vigilância, sem atribuir ao áudio a localização de larvas na água.

[Sources]
https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/a/aedes-aegypti/vigilancia-entomologica/vigilancia-entomologica/
https://saude.se.gov.br/lacen-promove-treinamento-para-auxiliar-identificacao-de-larvas-de-mosquitos-transmissores-de-arboviroses/
https://site.ufla.br/arquivo-de-noticias/7912-ufla-intensifica-trabalhos-de-controle-e-prevencao-a-dengue-e-leishmaniose
https://saude.rs.gov.br/rs-tem-85-cidades-com-alto-risco-de-transmissao-de-dengue-zika-e-chikungunya
https://mosquito.saude.es.gov.br/Not%C3%ADcia/conheca-as-diferencas-entre-dengue-zika-e-chikungunya
Foto do pneu: Divulgação/SES-RS; foto do recipiente publicada pela SESA-ES. URLs e hashes dos originais em photo-provenance.json.
Arquivo local: results/ternary_lstm_2026_10_15/metrics.json
Apoio à vigilância e priorização de inspeções são possibilidades de aplicação do projeto, não benefícios operacionais já medidos.
[/Sources]'''
newrow=dict(slideId=SID,number=2,title=values['e00'],seconds=65,notes_full=notes)
plan=[prior[0],newrow]+[dict(s,number=s['number']+1) for s in prior[1:]]
assert len(plan)==20 and sum(s['seconds'] for s in plan)==1175
for s in P['slides'][1:]:
    # The number is a separate native shape, including row-based exemplars.
    oid=s['objectId']+'_e01' if s['objectId'] in ['ifsc26_s02','ifsc26_s03','ifsc26_regularizacao','ifsc26_s19','ifsc26_s20','ifsc26_arduino_dataset'] else s['objectId']+'_e03'
    e=next(e for e in s['pageElements'] if e['objectId']==oid)
    value=next(r['number'] for r in plan if r['slideId']==s['objectId'])
    native_replace(e,oid,f'{value:02d}')
refs=next(s for s in P['slides'] if s['objectId']=='ifsc26_s19')
native_replace(next(e for e in refs['pageElements'] if e['objectId']=='ifsc26_s19_e08'),
               'ifsc26_s19_e08','Fotos: IFSC, CDC, Arduino, SES-RS e SESA-ES')
for row in plan:
    if row['slideId']=='ifsc26_s19':
        row['notes_full']=row['notes_full'].replace('[/Sources]',
          'https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/a/aedes-aegypti/vigilancia-entomologica/vigilancia-entomologica/\nhttps://saude.se.gov.br/lacen-promove-treinamento-para-auxiliar-identificacao-de-larvas-de-mosquitos-transmissores-de-arboviroses/\nhttps://saude.rs.gov.br/rs-tem-85-cidades-com-alto-risco-de-transmissao-de-dengue-zika-e-chikungunya\nhttps://mosquito.saude.es.gov.br/Not%C3%ADcia/conheca-as-diferencas-entre-dengue-zika-e-chikungunya\nFotos adicionais de criadouros: SES-RS e SESA-ES.\n[/Sources]')
        notesid=refs['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
        requests.extend([{'deleteText':{'objectId':notesid,'textRange':{'type':'ALL'}}},
            {'insertText':{'objectId':notesid,'insertionIndex':0,'text':row['notes_full']}}])

# Two unaltered photographs; derive both dimensions from each intrinsic ratio.
original=next(e for e in source['pageElements'] if 'image' in e)
base_width_pt=original['size']['width']['magnitude']/12700
width=145; height=width*450/800; center_y=180.5
requests.append({'updatePageElementTransform':{'objectId':mapping[original['objectId']],
 'applyMode':'ABSOLUTE','transform':{'scaleX':width/base_width_pt,'scaleY':width/base_width_pt,
 'translateX':280,'translateY':center_y-height/2,'unit':'PT'}}})
pneu=str(W/'assets/criadouro_pneu_rs.jpg');recipiente=str(W/'assets/criadouro_recipiente_es.jpg')
images=[dict(path=pneu,requests=[{'replaceImage':{'imageObjectId':mapping[original['objectId']],
 'url':pneu,'imageReplaceMethod':'CENTER_INSIDE'}}]),dict(path=recipiente,requests=[{'createImage':{
 'objectId':SID+'_photo_recipiente','url':recipiente,'elementProperties':{'pageObjectId':SID,
 'size':{'width':{'magnitude':145,'unit':'PT'},'height':{'magnitude':145*598/800,'unit':'PT'}},
 'transform':{'scaleX':1,'scaleY':1,'translateX':433,'translateY':center_y-145*598/800/2,'unit':'PT'}}}}])]
for name,value in [('duplicate',duplicate),('content',requests),('images',images),('plan',plan),('frame-map',dict(
 narrative='Problem, current species identification and proposed acoustic support for surveillance',
 exemplar=source['objectId'],construction='duplicate native photo-led slide; replace content and use two source photos',
 elementMap={e['objectId']:dict(target=mapping[e['objectId']],action='replace photograph and resize uniformly' if 'image' in e else 'replace native text, keep style and geometry') for e in source['pageElements']},
 allExemplarMediaMapped=True,addedMedia='second photo of a water-holding container; fit full intrinsic ratio',
 finalOrder=[s['slideId'] for s in plan],officialMastersAndLayouts='preserve unchanged'))]:
    (W/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
(W/'introduction-notes.txt').write_text(notes)
(W/'Roteiro_apresentacao_IFSC.txt').write_text('\n\n'.join(f"{s['number']:02d}. {s.get('title') or 'Capa'} ({s['seconds']} s)\n{s['notes_full']}" for s in plan)+'\n\nTempo total sugerido: 19min35s.\n')
(W/'analise-lacunas.txt').write_text('Falta uma introdução antes da definição das três classes que conecte os criadouros, a identificação entomológica atual e a contribuição do detector. Inserir um slide após a capa com fotos reais de criadouros, texto breve e tom construtivo. Fontes oficiais fundamentam a prática atual. Benefícios do áudio são apresentados como apoio proposto à triagem e à vigilância, sem confundir identificação de adultos com localização de focos ou resultados de campo. Preservar resultados, modelo IFSC, autoria, foto do campus e a discussão no encerramento.\n')
(W/'fallback-reasons.txt').write_text('Existing native deck edited in place using the Google Drive connector. The dependency loader is unavailable; use the native read/parse/check/render helpers and local Poppler. The marker records a supported pptx-format edit while authoring remains native. Previous verified export established bounded legacy compatibility because streaming file references lack workspace_path; reuse that adapter with the unchanged export helper.\n')
print('Ready: 20 slides; 19min35s; two sourced, unaltered breeding-site photographs.')
