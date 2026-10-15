from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;R=W.parents[1];A=W/'assets';A.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22,'axes.labelsize':22,'xtick.labelsize':20,'ytick.labelsize':23,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False})
metrics=json.loads((R/'results/ternary_lstm_2026_10_15/metrics.json').read_text());ref=metrics['reference'];sources=ref['source_breakdown'];cm=np.asarray(ref['confusion_matrix_raw']);assert cm.sum()==1494
fig,axes=plt.subplots(1,2,figsize=(16,6.5),constrained_layout=True,gridspec_kw={'width_ratios':[1,1.25]})
for ax,labels,values,title,colors in [(axes[0],['Dryad','HumBug','ESC-50'],[sources[x]['groups'] for x in ['Dryad','HumBug','ESC50']],'Origem das gravações',['#287d32']*3),(axes[1],['Ruído','Outro mosquito','Aedes aegypti'],cm.sum(axis=1).tolist(),'Classes de referência',['#087f8c','#287d32','#e2a900'])]:
 pos=np.arange(3);ax.barh(pos,values,color=colors,height=.52);ax.set_yticks(pos,labels);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.22);ax.set_title(title,pad=23,fontweight='bold',fontsize=23);ax.set_xlabel('Grupos de gravação',labelpad=14);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
 for y,v in zip(pos,values):ax.text(v+max(values)*.025,y,f'{v:,}'.replace(',','.'),va='center',fontweight='bold',fontsize=24)
fig.savefig(A/'dados_e_classes.png',dpi=150,facecolor='white');plt.close(fig)
c=pd.read_csv(R/'results/ternary_lstm_2026_10_15/comparison.csv');expected=c.accuracy.to_numpy()*100
fig,ax=plt.subplots(figsize=(16,6.5),constrained_layout=True);labels=['Cascata clássica','LSTM · direta','LSTM · calibrada','Bi-LSTM · direta','Bi-LSTM · calibrada'];y=np.arange(5);colors=['#287d32','#087f8c','#deb027','#087f8c','#deb027'];ax.barh(y,expected,color=colors,height=.57);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,103);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('Acurácia geral por grupo (%)',labelpad=12);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
for yi,v in zip(y,expected):ax.text(v+1,yi,f'{v:.2f}%'.replace('.',','),va='center',fontweight='bold',fontsize=24)
fig.savefig(A/'acuracia_cinco_regras.png',dpi=150,facecolor='white');plt.close(fig)
(A/'chart-provenance.json').write_text(json.dumps({'source':'results/ternary_lstm_2026_10_15/metrics.json + comparison.csv','sources':{x:sources[x]['groups'] for x in sources},'class_totals':cm.sum(axis=1).tolist(),'accuracy_percent':expected.tolist(),'no_new_training':True},indent=2))
print('2 figures generated from actual results')
