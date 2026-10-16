"""Shared, data-derived figures for the notebook and native Slides presentation."""
import numpy as np
import matplotlib.pyplot as plt

MODELS=['cnn','mlp','svm_rbf','extra_trees','histgb']
NAMES={'cnn':'CNN 2D','mlp':'MLP','svm_rbf':'SVM RBF','extra_trees':'Extra Trees','histgb':'HistGB'}
COLORS=['#238139','#82A43B','#087F8C','#73967B','#BBCB9C']
SOURCES=['dryad','humbug','torres','esc50']


def figure(reference, kind):
    with plt.rc_context({'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','axes.titleweight':'bold'}):
        fig=_figure(reference,kind)
    return fig


def _figure(r,kind):
    m=r['metrics'];fig=plt.figure(figsize=(10.8,4.5),layout='constrained')
    if kind=='accuracy':
        ax=fig.subplots();values=[100*m[n]['argmax']['all']['accuracy']for n in MODELS]
        bars=ax.barh([NAMES[n]for n in MODELS][::-1],values[::-1],color=COLORS[::-1],height=.62)
        ax.bar_label(bars,labels=[f'{v:.2f}%'.replace('.',',')for v in values[::-1]],padding=7)
        ax.set(xlim=(0,105),xlabel='Acurácia geral (%)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    elif kind=='datasets':
        ax=fig.subplots();v=np.array([[100*m[n]['argmax'][s]['accuracy']for s in SOURCES]for n in MODELS])
        ax.imshow(v,vmin=0,vmax=100,cmap='Greens',aspect='auto')
        ax.set(xticks=range(4),xticklabels=['Dryad · 613','HumBug · 592','Torres · 300','ESC-50 · 289'],yticks=range(5),yticklabels=[NAMES[n]for n in MODELS])
        for i in range(5):
            for j in range(4):ax.text(j,i,f'{v[i,j]:.2f}%'.replace('.',','),ha='center',va='center',color='white'if v[i,j]>80 else '#15291B',fontsize=15)
    elif kind in ['aedes','holdout']:
        ax=fig.subplots();sources=['dryad','humbug','torres'];x=np.arange(3)
        for shift,n,color in [(-.18,'cnn',COLORS[0]),(.18,'mlp',COLORS[2])]:
            stats=[m[n]['argmax'][s]if kind=='aedes'else r['source_holdout'][s][n]['argmax']for s in sources]
            bars=ax.bar(x+shift,[100*v['aedes_recall']for v in stats],.34,label=NAMES[n],color=color)
            ax.bar_label(bars,labels=[f"{v['aedes_tp']}/{v['aedes_total']}\n{100*v['aedes_recall']:.1f}%"for v in stats],padding=4,fontsize=12)
        ax.set(xticks=x,xticklabels=['Dryad','HumBug','Torres'],ylim=(0,122),ylabel='Aedes reconhecidos (%)');ax.legend(loc='upper left',ncol=2)
    elif kind=='split':
        ax=fig.subplots();left=0
        for v,label,color in [(216,'Treino interno · 216',COLORS[0]),(72,'Validação · 72','#9FBC70'),(72,'Teste · 72',COLORS[2])]:
            ax.barh([0],v,left=left,color=color,height=.45);ax.text(left+v/2,0,label.replace(' · ','\n'),ha='center',va='center',color='white'if v==216 else '#15291B',fontsize=16);left+=v
        ax.set(yticks=[],xlim=(0,360),xlabel='Exemplos Aedes em uma dobra externa',ylim=(-.7,.8));ax.text(144,.45,'Desenvolvimento: 288 (80%)',ha='center',fontsize=17);ax.text(324,.45,'Teste: 20%',ha='center',fontsize=15)
        ax.text(180,-.5,'Após selecionar a época, o modelo é reiniciado e treinado nos 288.',ha='center',fontsize=13)
    elif kind=='mel':
        ax=fig.subplots();z=np.asarray(r['example']['sequence'])
        im=ax.imshow(z.T,origin='lower',aspect='auto',extent=(0,179,0,16),vmin=-80,vmax=0,cmap='magma')
        ax.set(xlabel='Instante da sequência · 179 quadros reais',ylabel='Banda Mel · 16 bandas abaixo de 4 kHz');fig.colorbar(im,ax=ax,label='Intensidade relativa (dB)')
    elif kind=='learning':
        axes=fig.subplots(1,2)
        for ax,n in zip(axes,['cnn','mlp']):
            v=r['neural_histories'][n]['0'];h=v['history'];best=v['selection']['best_epoch']
            ax.plot([x['epoch']for x in h],[x['train_loss']for x in h],label='Treino · janelas',color=COLORS[0])
            ax.plot([x['epoch']for x in h],[x['validation_loss']for x in h],label='Validação · grupos',color='#C08A00')
            ax.axvline(best,ls='--',color=COLORS[2],alpha=.8);ax.set(title=f'{NAMES[n]} · época {best}',xlabel='Época',ylabel='Perda');ax.legend(fontsize=10)
    elif kind=='confusion':
        axes=fig.subplots(1,2)
        for ax,n in zip(axes,['cnn','mlp']):
            cm=np.asarray(m[n]['argmax']['all']['confusion_matrix']);v=cm/cm.sum(1,keepdims=True)
            ax.imshow(v,cmap='Greens',vmin=0,vmax=1)
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{100*v[i,j]:.1f}%',ha='center',va='center',color='white'if v[i,j]>.6 else '#15291B',fontsize=13)
            ax.set(title=NAMES[n],xticks=range(3),xticklabels=['Ruído','Outro','Aedes'],yticks=range(3),yticklabels=['Ruído','Outro','Aedes'],xlabel='Classe prevista',ylabel='Classe real')
    elif kind=='rules':
        axes=fig.subplots(1,3);x=np.arange(2)
        for ax,key,scope,title in zip(axes,['aedes_recall','noise_fpr','other_fpr'],['original_corpus','all','all'],['Aedes originais','Ruído → Aedes','Outro → Aedes']):
            for shift,rule,label,color in [(-.18,'argmax','Maior escore',COLORS[0]),(.18,'calibrated','Limiares ajustados',COLORS[2])]:
                values=[100*m[n][rule][scope][key]for n in ['cnn','mlp']]
                bars=ax.bar(x+shift,values,.34,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
            ax.set(xticks=x,xticklabels=['CNN','MLP'],ylim=(0,110 if key=='aedes_recall'else 13),title=title,ylabel='%')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=11)
    elif kind=='false_positives':
        ax=fig.subplots();x=np.arange(5)
        for shift,key,label,color in [(-.18,'noise_fpr','Ruído → Aedes',COLORS[0]),(.18,'other_fpr','Outro → Aedes',COLORS[2])]:
            v=[100*m[n]['argmax']['all'][key]for n in MODELS];bars=ax.bar(x+shift,v,.34,label=label,color=color);ax.bar_label(bars,fmt='%.2f',fontsize=10,padding=3)
        ax.set(xticks=x,xticklabels=[NAMES[n]for n in MODELS],ylim=(0,25),ylabel='Falsos positivos (%)');ax.legend()
    else:raise ValueError(kind)
    return fig
