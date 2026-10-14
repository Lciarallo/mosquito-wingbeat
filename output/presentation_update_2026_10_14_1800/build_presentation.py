from pathlib import Path
import json, hashlib, csv, io
from datetime import datetime, timezone
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).parent
OUT=ROOT/'output/pdf/Apresentacao_Ensemble_v4_Auditoria_Atualizada_2026_10_14.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
A=WORK/'assets'
for name,file in [('Inter','Inter-Regular.ttf'),('InterBold','Inter-Bold.ttf')]:
 pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/rsms-inter-fonts/'+file))
W,H=960,540
BG='#F5F7F5'; DARK='#0A211D'; GREEN='#008269'; RED='#C4311C'; MUTED='#566E66'; CARD='#E7EFEC'
c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('Ensemble v4 e auditoria bioacústica - mosquito-wingbeat');c.setAuthor('Luiz Eduardo do Nascimento Ciarallo - IFSC')
page=0

def box(x,y,w,h,fill=BG,r=0):
 c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(fill));c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=0)
def text(s,x,y,w,size=18,color=DARK,bold=False,maxh=300):
 st=ParagraphStyle('s',fontName='InterBold' if bold else 'Inter',fontSize=size,leading=size*1.28,textColor=HexColor(color),spaceAfter=0)
 p=Paragraph(s,st);pw,ph=p.wrap(w,maxh)
 while ph>maxh and st.fontSize>14:
  st.fontSize-=.5;st.leading=st.fontSize*1.28;p=Paragraph(s,st);pw,ph=p.wrap(w,maxh)
 if ph>maxh:raise ValueError(f'overflow slide {page}: {s[:60]} ({ph}>{maxh})')
 p.drawOn(c,x,H-y-ph);return ph

def begin(title,source):
 global page
 page+=1;box(0,0,W,H)
 text(title,56,39,848,30,bold=True,maxh=85)
 text(source,56,505,810,9,color=MUTED,maxh=25)
 text(f'{page:02d} / 13',865,505,50,9,color=MUTED,maxh=25)

def end():c.showPage()
def card(x,title,body,y=153,w=270,h=333,color=GREEN):
 box(x,y,w,h,CARD,16);text(title,x+18,y+19,w-36,22,color,bold=True,maxh=62);text(body,x+18,y+82,w-36,16,maxh=h-94)
def image(path,x,y,w,h,cover=False):
 im=Image.open(path).convert('RGB')
 if cover:
  ratio=max(w/im.width,h/im.height);sz=(round(im.width*ratio),round(im.height*ratio));im=im.resize(sz);left=(im.width-w)/2;top=(im.height-h)/2;im=im.crop((int(left),int(top),int(left+w),int(top+h)))
 c.drawImage(ImageReader(im),x,H-y-h,w,h,preserveAspectRatio=not cover,anchor='c',mask='auto')
def table(headers,rows,x,y,width,colwidths=None,size=16,rowh=43):
 widths=np.array(colwidths or [1]*len(headers),float);widths=width*widths/widths.sum()
 for j,row in enumerate([headers]+rows):
  box(x,y+j*rowh,width,rowh,DARK if j==0 else (CARD if j%2 else BG))
  xx=x
  for k,v in enumerate(row):
   text(str(v),xx+12,y+j*rowh+11,widths[k]-20,size,'#FFFFFF' if j==0 else DARK,bold=j==0,maxh=rowh-12);xx+=widths[k]

# 1 - retain the original title photograph and palette.
page+=1;image(A/'p1_5.jpeg',0,0,W,H,True)
c.saveState();c.setFillColor(HexColor(DARK));c.setFillAlpha(.64);c.rect(0,0,W,H,fill=1,stroke=0);c.restoreState()
text('Ensemble de Espécies v4<br/>e Diagnóstico Bioacústico',56,178,800,39,'#FFFFFF',True,120)
text('Resultados congelados v1-v4 e reexecução dos scripts do Drive',56,309,820,21,'#BFD0C8',maxh=70)
text('Luiz Eduardo do Nascimento Ciarallo · IFSC · 14 de outubro de 2026',56,493,830,12,'#BFD0C8')
end()
# 2
begin('560 blocos PCM: separação por origem antes da avaliação','Fontes: relatório v4 · source_pcm_audit.csv · freeze.json · verify_history')
card(56,'Acervo Dryad','<b>24.405 janelas de 1 s</b> em 20 espécies e 818 arquivos.<br/><br/>613 grupos: 569 positivos e 44 de ruído.<br/><br/>53 colisões de PCM reunidas em <b>560 blocos</b>.')
card(345,'Treino aninhado','<b>3 dobras externas + modelo final.</b><br/><br/>Fit, calibração e teste disjuntos por bloco PCM.<br/><br/>SVM RBF, ExtraTrees e HistGradientBoosting.')
card(634,'Congelamento v4','<b>14/10/2026 às 05:45:30 UTC.</b><br/><br/>Pesos e limiares congelados antes das previsões de teste.<br/><br/>1.239 arquivos históricos verificados nesta atualização.',color=RED)
end()
# 3
begin('A v4 corrige o frontend e a calibração; ganho externo não é garantido','Fontes: acoustic_features_v4.py · nested_species_ensemble_v4.py · relatório v4')
card(56,'Espectro relativo','Piso local como <b>razão de potências em dB</b>; evita zerar bins antes do log.<br/><br/>Mel relativo ao piso; mediana por frame; C0 zerado, C1/C2 preservados.<br/><br/>STFT de 100 e 200 ms.')
card(345,'MIL seletivo','Saliência harmônica e nitidez da previsão, ponderadas por presença.<br/><br/>Top-fração: <b>20%, 35%, 50% ou 65%</b> em gravações longas.<br/><br/>Escolha apenas na calibração.')
card(634,'Taxonomia e prior','Correção de cardinalidade por gênero e potência temperada.<br/><br/>Prior de espécie e limiar de gênero ajustados na calibração.<br/><br/>Invariância exata a ruído aditivo <b>não foi demonstrada</b>.',color=RED)
end()
# 4 - actual chart from frozen comparison CSV.
rows=list(csv.DictReader((ROOT/'results/species_nested_v4_2026_10_14/comparison_metrics.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.facecolor':BG,'figure.facecolor':BG,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(13.5,4.8),sharey=True)
colors=['#64748B','#B65B42',GREEN,'#153B32']
for ax,scope,label in zip(axs,['dryad','external'],['Dryad OOF (569 gravações positivas)','BioDCASE D3 (17 arquivos pareados)']):
 for i,version in enumerate(['v1','v2','v3','v4']):
  vals=[100*float(next(r for r in rows if r['scope']==scope and r['version']==version and r['task']==task)['accuracy']) for task in ['species','genus_marginalized']]
  bars=ax.bar(np.arange(2)+(i-1.5)*.18,vals,width=.17,color=colors[i],label=version)
  for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.1f}',ha='center',fontsize=9)
 ax.set_xticks([0,1],['Espécie','Gênero agregado']);ax.set_ylim(0,108);ax.set_title(label,fontsize=12,fontweight='bold',pad=18);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Acurácia por gravação (%)');axs[1].legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False)
fig.tight_layout();fig.savefig(A/'frozen_comparison.png',dpi=200,bbox_inches='tight');plt.close(fig)
begin('A v4 não resgata a generalização do ensemble congelado','Fonte: results/species_nested_v4_2026_10_14/comparison_metrics.csv · gênero = agregação de cada versão')
image(A/'frozen_comparison.png',56,137,848,305)
text('Espécie: 4/17 em todas as rodadas. Gênero v4: 5/17; v3: 6/17; v2: 8/17.',56,459,848,20,RED,True,35)
end()
# 5
begin('A rejeição precisa ser lida junto com cobertura e acerto aceito','Fonte: comparison_metrics.csv v4 · sem gate adicional de presença · valores percentuais')
table(['v4 / nível','Acc.','Bal.','F1 estrito','Cobertura','Acc. aceita'],[
 ['Dryad / espécie','59,23','65,55','57,20','11,07','77,78'],
 ['Dryad / gênero','86,47','87,19','83,65','89,63','88,24'],
 ['D3 / espécie','23,53','20,00','11,50','41,18','42,86'],
 ['D3 / gênero','29,41','20,83','12,50','100,00','29,41'],
 ['Drive / ambos','0,00','0,00','0,00','0,00','n/a'],
 ],56,160,848,[2.5,1,1,1.3,1.4,1.4],size=16,rowh=45)
text('<b>Macro-F1 estrito</b> inclui categorias previstas fora do suporte verdadeiro. No D3, espécie: suporte 23,00% versus união 11,50%.',56,440,848,17,color=MUTED,maxh=48)
end()
# 6
begin('Aedes aegypti segue sem acerto no D3 do ensemble v4','Fontes: relatórios v3/v4 · per_taxon.csv · fotografias preservadas da apresentação original (CDC)')
for x,f in zip([56,345,634],['p1_2.jpeg','p1_3.jpeg','p1_4.jpeg']):image(A/f,x,155,270,169,True)
text('Aedes (4 espécies)',56,345,270,22,GREEN,True,35)
text('v4: <b>0/4 Ae. aegypti</b> nos 17 externos.<br/><br/>Os novos lotes de Ae. aegypti também não foram resgatados.',56,389,270,17,maxh=100)
text('Anopheles (12)',345,345,270,22,GREEN,True,35)
text('A soma simples carrega o tamanho do catálogo.<br/><br/>A v4 corrige cardinalidade; isso <b>não garante</b> acerto em novo canal.',345,389,270,17,maxh=100)
text('Culex e Culiseta (4)',634,345,270,22,GREEN,True,35)
text('Táxons fora do suporte penalizam o F1 estrito.<br/><br/>Os lotes externos cobrem somente <b>5 das 20 espécies</b>.',634,389,270,17,maxh=100)
end()
# 7
begin('Scripts recentes do Drive: 26/28 não significa 100% confiável','Fonte: revisao_resultados_drive_2026_10_14_161041.md · reexecução encerrada às 16:20:49 UTC')
table(['Script neural/heurístico','Anterior','Atual'],[
 ['28 conhecidos / espécie','12/28 (42,86%)','26/28 (92,86%)'],
 ['17 pareados / espécie','6/17 (35,29%)','6/17 (35,29%)'],
 ['17 pareados / gênero','7/17 (41,18%)','10/17 (58,82%)'],
 ],56,160,848,[2.5,1.6,1.6],size=18,rowh=53)
text('<b>Conjunto conhecido, apenas cinco espécies.</b> Usa pesos neurais históricos e bônus bioacústicos; não é um ensemble v5 retreinado.',56,396,848,19,maxh=60)
text('Espécie nos 17: IC95% 17,65-58,82%. Ganho de gênero: +17,65 pp, IC pareado -5,88 a +41,18 pp.',56,460,848,16,RED,maxh=40)
end()
# 8
begin('Áudio privado: o diagnóstico harmônico mudou; hardware não foi validado','Fonte: três resumos auditados em drive_review_2026_10_14_161041/ · imagem Arduino preservada do PDF original')
image(A/'p1_7.jpeg',650,140,255,337)
table(['Modo','Voo','Enxame','Ae. aegypti*'],[
 ['Brasil emulado','139','125','132 / 118'],
 ['Brasil bruto','139','125','132 / 117'],
 ['Global bruto','139','125','112 / 92'],
 ],56,170,566,[1.8,.8,1,1.55],size=15,rowh=48)
text('<b>106,24 s · 211 janelas sobrepostas.</b><br/>Mediana de F0 nas janelas detectadas: <b>600 Hz</b>.',56,384,560,19,maxh=70)
text('*Candidatos / persistência temporal. São contagens do algoritmo, sem anotação independente por janela. Emulação PDM é offline.',56,456,560,13,MUTED,maxh=45)
end()
# 9 - exact audited temporal intervals.
active=[(25.5,26),(35.5,37.5),(39,39.5),(41,47.5),(48,91.5),(92,95),(95.5,99.5),(100,105.992)]
fig,ax=plt.subplots(figsize=(13.7,2.1));ax.broken_barh([(0,105.992)],(0,.8),facecolors='#D6E0DB')
ax.broken_barh([(a,b-a) for a,b in active],(0,.8),facecolors=GREEN);ax.axvspan(105.992,106.24,color=RED,alpha=.8)
ax.set_xlim(0,106.24);ax.set_ylim(-.1,1);ax.set_yticks([]);ax.set_xticks([0,20,40,60,80,100,106.24]);ax.set_xlabel('Tempo (s)');ax.spines['left'].set_visible(False);ax.spines['bottom'].set_visible(False);fig.tight_layout();fig.savefig(A/'timeline.png',dpi=200,bbox_inches='tight');plt.close(fig)
begin('A cronologia marca padrão de enxame; não comprova presença de Aedes','Fonte: causal_swarm_timeline dos três modos auditados · sinal ativo em verde; restante analisado em cinza')
image(A/'timeline.png',56,137,848,138)
text('<b>Intervalos ativos (s)</b><br/>25,5-26,0 · 35,5-37,5 · 39,0-39,5 · 41,0-47,5<br/>48,0-91,5 · 92,0-95,0 · 95,5-99,5 · 100,0-105,992',56,305,848,22,maxh=105)
text('65,992 s marcados como ativos. Os últimos 0,248 s não foram analisados.<br/>“Ausente” é saída do algoritmo. Sem rótulos temporais, não há precisão ou sensibilidade de enxame.',56,430,848,17,MUTED,maxh=65)
end()
# 10
begin('A auditoria refuta a independência entre janelas do script recente','Fonte: batch_inductivity_counterexample.json · scripts/predict_audio.py · auditoria de 16:20 UTC')
card(56,'Mesma janela','PCM e features da janela de <b>25,5 s</b> permaneceram idênticos.<br/><br/>A intervenção modificou quatro features em outra linha do lote.<br/><br/>Maior mudança de probabilidade: <b>0,578855</b>.')
card(345,'Resultado mudou','<b>Ae. aegypti → Ae. albopictus.</b><br/><br/>O contexto usa médias da gravação inteira para alterar bônus.<br/><br/>A persistência causal não torna a classificação inteira indutiva.',color=RED)
card(634,'Consequência','Os ganhos são <b>exploratórios</b>, sem calibração recebida das heurísticas.<br/><br/>Não extrapolar 26/28 a 20 espécies, novos aparelhos ou campo.<br/><br/>“Confirmado” significa persistência, não confirmação biológica.')
end()
# 11
begin('Preservação passou; a última suíte auditada teve uma falha nova','Fonte: pytest_complete · historical_verification.json · verify_freeze/verify_history · snapshot de 16:20 UTC')
card(56,'Histórico intacto','Rodadas <b>v1, v2, v3 e v4</b> preservadas.<br/><br/>Hashes e manifestos conferidos antes e depois da atualização.<br/><br/>Sem alteração de firmware, instalador ou pesos congelados.')
card(345,'Testes auditados','<b>215 passaram; 1 falhou.</b><br/><br/>Os 202 anteriores passaram; um caso novo v5 falhou na norma cepstral.<br/><br/>Resultado recebido de RMS: 0,2895; esperado pelo teste: 2,5.',color=RED)
card(634,'Limite do snapshot','A suíte refere-se aos arquivos auditados às <b>16:20 UTC</b>.<br/><br/>Não certifica revisões posteriores no Drive.<br/><br/>Não há OOF, freeze e modelo v5 verificados para substituir a v4.')
end()
# 12
page+=1;box(0,0,W,H,DARK)
text('Confiável para explorar padrões.<br/>Ainda experimental para identificar espécies.',56,139,850,34,'#FFFFFF',True,120)
text('O ensemble v4 não atingiu generalização externa. O script recente ajuda a inspecionar harmônicos e segmentar o áudio, mas mantém dependência entre janelas e regras não calibradas.',56,278,840,21,'#BFD0C8',maxh=110)
text('Próxima evidência: corrigir o contexto → calibrar com grupos disjuntos → congelar → avaliar nova aquisição, com negativos e referência entomológica.',56,414,840,17,'#BFD0C8',maxh=72)
text('Uso offline · Sem liberação para Arduino · github.com/Lciarallo/mosquito-wingbeat',56,510,840,10,'#BFD0C8',maxh=20)
end()
# 13 source appendix accessible readable links.
begin('Fontes e escopo desta atualização','Versão derivada do PDF Apresentacao_Ensemble_v3_Impeccable.pdf · original preservado')
text('<b>Ensemble congelado v4</b><br/>reports/ensemble_especies_v4_2026_10_14.md<br/>results/species_nested_v4_2026_10_14/comparison_metrics.csv<br/>acoustic_features_v4.py · nested_species_ensemble_v4.py',56,145,848,18,maxh=125)
text('<b>Auditoria dos scripts recebidos do Drive</b><br/>reports/revisao_resultados_drive_2026_10_14_161041.md<br/>results/drive_review_2026_10_14_161041/<br/>metrics.json · bootstrap_intervals.json · resumos e contraexemplo',56,292,848,18,maxh=125)
text('As imagens do mosquito e da placa foram preservadas do PDF fornecido.<br/>Os gráficos foram recalculados com as métricas auditadas; nenhum resultado foi inventado.',56,435,848,16,MUTED,maxh=54)
c.linkURL('https://drive.google.com/file/d/14F3mynyh8fgNdZn40CZccO3O1ZAIQ4V1/view', (56,H-268,900,H-140),relative=0)
c.linkURL('https://drive.google.com/file/d/1wg0jV9N1Zmvjg1ZYm776Ky_kn3VvUpnQ/view', (56,H-415,900,H-289),relative=0)
end();c.save()
assert page==13
print(OUT)
