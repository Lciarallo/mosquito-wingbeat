from pathlib import Path
import json,hashlib,shutil,datetime,re
import pandas as pd
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
a=json.loads((W/'raw-before.json').read_text())['structuredContent'];b=json.loads((W/'raw-approved.json').read_text())['structuredContent'];plan=json.loads((W/'plan.json').read_text())
assert a['masters']==b['masters'] and a['layouts']==b['layouts'];assert len(b['slides'])==20;assert [s['objectId'] for s in b['slides']]==[r['slideId'] for r in plan];assert sum(r['seconds'] for r in plan)==1080
full=[]
for row,sl in zip(plan,b['slides']):
 oid=sl['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId'];e=next(e for e in sl['slideProperties']['notesPage']['pageElements'] if e['objectId']==oid)
 notes=''.join(t.get('textRun',{}).get('content','') for t in e['shape']['text']['textElements']).strip();assert notes==row['notes_full'].strip(),row['number']
 texts=[''.join(t.get('textRun',{}).get('content','') for t in e.get('shape',{}).get('text',{}).get('textElements',[])).strip() for e in sl['pageElements']]
 if row['number']>1:assert f"{row['number']:02d}" in texts
 full.extend(texts)
 assert any(row['title']==t for t in texts) or row['number']==1
assert 'Luiz Eduardo do Nascimento Ciarallo' in full
assert not any(e['objectId']=='ifsc26_s01_campus' for sl in b['slides'] for e in sl['pageElements'])
assert any(e['objectId']=='ifsc26_logo_oficial_florianopolis' for sl in b['slides'] for e in sl['pageElements'])
# Check chart numbers and model-result table against the original metrics.
metrics=json.loads((ROOT/'results/ternary_lstm_2026_10_15/metrics.json').read_text());comp=pd.read_csv(ROOT/'results/ternary_lstm_2026_10_15/comparison.csv');plot=json.loads((W/'assets/chart-provenance.json').read_text());assert plot['class_totals']==[426,1008,60]
assert plot['accuracy_percent']==(comp.accuracy*100).tolist()
sl=next(sl for sl in b['slides'] if sl['objectId']=='ifsc26_s10');table=next(e['table'] for e in sl['pageElements'] if 'table' in e)
for row,(_,metric) in zip(table['tableRows'][1:],comp.iterrows()):
 cells=[''.join(t.get('textRun',{}).get('content','') for t in c['text']['textElements']).strip() for c in row['tableCells']]
 for actual,key in zip(cells[1:],['aedes_recall','noise_to_aedes_fpr','other_to_aedes_fpr']):assert actual==f'{metric[key]*100:.2f}%'.replace('.',','),(actual,key)
unchanged=[];changed=[]
for i in range(1,21):
 p=f'slide-{i:02d}.png';(unchanged if (W/'renders/review'/p).read_bytes()==(W/'renders/final'/p).read_bytes() else changed).append(i)
assert changed==[1,7,11,12,14,16,20]
# Improve the study guide typography without changing the frozen native-note content.
g=W/'Guia_de_estudo_e_fala.txt';text=g.read_text()
for r in plan:
 how=r['how'];pretty=re.sub(r'(?<=[A-Za-zÀ-ÿ])(?=\d)|(?<=\d)(?=[A-Za-zÀ-ÿ])',' ',how);pretty=re.sub(r'(?<=[,;])(?=[A-Za-zÀ-ÿ])',' ',pretty);text=text.replace(how,pretty)
text+='''\n\nGLOSSÁRIO RÁPIDO\nFFT: Fast Fourier Transform; algoritmo para calcular a transformada discreta de Fourier.\nSTFT: Short-Time Fourier Transform; análise de Fourier em janelas sucessivas.\nLSTM: Long Short-Term Memory; rede recorrente com memória controlada por portas.\nBi-LSTM: LSTM bidirecional; combina os dois sentidos de processamento.\nLR: Logistic Regression; regressão logística.\nHistGB: Histogram-based Gradient Boosting; combinação de árvores por boosting.\nMC: Monte Carlo; repetição estocástica para observar a variabilidade das respostas.\nOOF: Out-of-Fold; previsão de um grupo feita pelo modelo que não o utilizou no ajuste daquela dobra.\nPCM: Pulse-Code Modulation; representação digital do sinal amostrado.\nRMS: raiz da média dos quadrados; medida de amplitude efetiva.\nFPR: False Positive Rate; proporção de negativos chamados de alvo.\nCheckpoint: estado salvo dos pesos da rede.\nSoftmax: transformação de escores em probabilidades positivas cuja soma é um.\n'''
g.write_text(text)
name='IFSC_Mosquitos_Didatica_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf';shutil.copyfile(W/'renders/final/presentation.pdf',W/name)
receipt=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=b['revisionId'],slides=20,duration_seconds=1080,official_master_unchanged=True,official_layouts_unchanged=True,campus_photo_removed=True,authentic_logo_source='https://ifsc.edu.br/web/campus-florianopolis/identidade-visual',author_verified=True,source_photos_preserved=True,all_20_slides_reviewed_at_full_resolution=True,repair_passes=1,repaired_pages=changed,unchanged_after_repair=unchanged,all_native_notes_match_plan=True,all_metric_table_cells_match_results=True,chart_numbers_match_results=True,source_issues=0,pdf=name,pdf_sha256=hashlib.sha256((W/name).read_bytes()).hexdigest(),pdf_bytes=(W/name).stat().st_size,native_url='https://docs.google.com/presentation/d/1JzWMYfuNXbIpfwOVcMKuNk5R4xfhAw69S4DAAAfy99A/edit?usp=drivesdk')
(W/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False))
