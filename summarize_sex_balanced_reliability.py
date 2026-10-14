"""Final report/notebook for the feedback adaptation and its NEW frozen test."""
import json
import pandas as pd,numpy as np,nbformat
from train_sex_balanced_reliability import ROOT,OUT
from evaluate_sex_balanced_reliability import verify
from summarize_reliability import pct,table


def main():
    fr=verify();d=json.loads((OUT/'reliability_decision.json').read_text());m=pd.read_csv(OUT/'fresh_metrics.csv')
    chosen=d['selected_model'];s=m[m.selected];allnew=s[s.scope.eq('all_fresh')].iloc[0]
    manifest=json.loads((OUT/'fresh_selection.json').read_text());search=json.loads((OUT/'validation_search.json').read_text())
    ci=json.loads((OUT/'fresh_group_bootstrap.json').read_text());private=pd.read_csv(OUT/'private_audio_aggregate_exploratory.csv')
    hist=pd.read_csv(OUT/'historical_metrics_exploratory.csv');paired=json.loads((OUT/'paired_improvement_exploratory.json').read_text())
    state='Ainda não confiável: não atingiu os mínimos definidos antes do novo teste.' if not d['offline_minimum_criteria_passed'] else 'Superou os mínimos nesta amostra offline; validação independente nas condições de uso continua pendente.'
    body=[f'''# Detector binário de Aedes: relatório final da adaptação — 01/10/2026

**{state}** O modelo escolhido antes do teste foi **{chosen}**. Precisão no novo teste: **{pct(allnew.aedes_precision)}**; sensibilidade: **{pct(allnew.aedes_recall)}**; F1 Aedes: **{pct(allnew.aedes_f1)}**; macro-F1: **{pct(allnew.macro_f1)}**. Acurácia geral: {pct(allnew.accuracy)}. A decisão não usa apenas a última métrica.

## Acurácia não basta para dizer “confiável”

Foi considerado o [artigo indicado pelo usuário](https://www.dio.me/articles/por-que-a-acuracia-accuracy-nao-e-a-melhor-metrica-para-avaliar-o-desempenho-de-classificadores): uma maioria de negativos pode esconder falhas graves na detecção. Precisão mede quantos alertas Aedes estão corretos; sensibilidade mede quantos Aedes são encontrados. F1 equilibra os dois; macro-F1 e especificidade incluem a classe negativa. As definições foram verificadas na [documentação oficial do Google](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall).

Registramos antes do teste: precisão, sensibilidade, macro-F1 e especificidade de pelo menos 60% no conjunto completo, entre apenas mosquitos e no domínio D3 com positivos e negativos; limites inferiores de 95% também de pelo menos 60% nessas métricas; sensibilidade Aedes e especificidade de cada táxon negativo de pelo menos 60%. São mínimos experimentais, não certificação. Inválidos não contam como acertos; desconhecidos não viram negativos.

A precisão depende da proporção de Aedes na amostra. Cenários de prevalência não são uma medição em campo. O artigo também inverte o efeito do limiar na sua explicação final: aumentar o limiar não aumenta a sensibilidade para scores fixos. Melhorar o modelo pode aumentar precisão e sensibilidade juntas.

## O que foi executado

As rodadas anteriores, incluindo reproduções de Fernandes e MosquitoSong+, redes residuais, misturas de treinamento e ensembles, estão documentadas no [relatório ampliado](confiabilidade_binaria_aedes_2026_10_01.md). A rodada anterior adquiriu 670 gravações para treino/validação/calibração e treinou quatro redes, de até 9,65 milhões de parâmetros, mais três modelos clássicos. Todos foram congelados antes de baixar 182 arquivos adicionais para teste. A rede PCEN escolhida pela validação teve apenas 2,7% de sensibilidade nesse teste; foi reprovada.

Depois dessa reprovação, esta nova rodada:

- Incorporou **987 janelas de três gravações de fêmeas de outras espécies** antes usadas na validação MosquitoSong. Elas agora são **treino**, nunca validação independente. O treino passou para 25.495 janelas válidas e rotuladas; o corpus total mantém 49.713 janelas, incluindo as partições reservadas e exemplos desconhecidos.
- Reajustou ExtraTrees regularizado, ExtraTrees sem limite de profundidade e HistGradientBoosting com os novos negativos.
- Extraiu representações de 768 dimensões do encoder PCEN de 9,58 milhões de parâmetros, preservando seus pesos, e treinou duas cabeças logísticas com essas representações e as 138 features acústicas. A normalização foi ajustada apenas no treino. Esse classificador continua usando a rede grande na inferência; não é um modelo preparado para Arduino.
- Corrigiu a comparação de candidatos: a validação agora usa **o limiar final calibrado de cada candidato**, em vez de comparar todos por score bruto 0,5. Isso é uma melhoria da seleção, não a correção de um vazamento do teste.
- Manteve o áudio privado e todos os Aedes CD-MSC/UFRGS fora do treino. Não houve seleção, calibração ou filtro guiado pelo novo teste.

As representações neurais foram extraídas sem reajustar o encoder. A escolha do checkpoint original usou a validação da rodada anterior; os resultados dessa validação já são parte do desenvolvimento. Após a adaptação, não restam fêmeas negativas MosquitoSong independentes na validação; esta passa a usar Dryad, HumBug e ESC-50. Portanto, as métricas de validação anteriores e atuais não são comparações na mesma amostra.

Pré-processamento: mono, reamostragem, remoção de DC, normalização de RMS, PCEN/log-Mel/contraste e features bioacústicas. Áudio real de 0,3 ou 0,992 s, sem repetição ou extensão da onda. Ruídos e misturas artificiais usados em rodadas anteriores eram apenas aumentos de treino, nunca novos indivíduos ou enxames reais.
''']
    body.append(table(['Candidato','Precisão validação','Recall validação','Macro-F1 validação','Menor recall por fonte/classe','Limiar final'],[[r['name'],pct(r['validation_precision']),pct(r['validation_recall']),pct(r['validation_macro_f1']),pct(r['validation_minimum_recall']),f"{r['cut']:.4f}"] for r in search]))
    body.append(f'''Congelamento: **{fr['frozen_utc']}**. Seleção do novo teste: **{manifest['selection_utc']}**. O vencedor maximiza o mínimo de precisão, sensibilidade e recalls por fonte/classe na validação, com pequeno termo de macro-F1. Calibração e escolha do limiar usam apenas a partição de calibração. Os modelos e pesos anteriores permaneceram idênticos por SHA-256.

## Outro teste realmente novo após o congelamento

Foram adquiridos **{len(manifest['files'])} arquivos**, dos quais {d['fresh_files']} foram avaliados após a regra conservadora de exclusão de colisões de conteúdo/PCM. Nenhum grupo de aquisição ou arquivo exato coincide com o teste anterior. Não substituímos gravações depois de observar resultados.

Aedes positivos e negativos Aedes albopictus/Culex/Anopheles vêm do mesmo domínio acústico CD-MSC D3. Incluímos ainda novos grupos HumBug e fontes ESC-50 do fold 5 não utilizadas anteriormente. Não usamos o domínio D5 óptico. Fontes: [CD-MSC 2026](https://zenodo.org/records/19095788), [HumBugDB](https://zenodo.org/records/4904800) e [ESC-50](https://github.com/karolpiczak/ESC-50).

**Limites importantes:** identidade dos indivíduos/sessões CD-MSC indisponível; novidade por arquivo não comprova uma campanha ou indivíduo independente. HumBug adiciona negativos novos, sem positivos novos da mesma fonte. O rótulo público é do arquivo, sem anotação de presença de batimento em cada janela. Os resultados são concordância com essa referência, não identificação entomológica de cada som. Janelas de um arquivo são correlacionadas.

O campo `duplicate_file` da auditoria marca qualquer colisão de janela PCM quantizada, não necessariamente duplicação do arquivo original. Na rodada anterior, 16 exclusões ESC-50 foram explicadas por 37 janelas silenciosas/constantes iguais; a regra conservadora foi mantida. Nesta rodada, a lista de exclusões é preservada no JSON de decisão, sem substituição por score.
''')
    mf=pd.DataFrame(manifest['files']);body.append(table(['Fonte','Arquivos selecionados','Segundos'],[[source,len(part),f'{part.seconds.sum():.1f}'] for source,part in mf.groupby('source')]))
    body.append('## Comparação no mesmo teste novo\n\nSomente os quatro candidatos da adaptação participaram da escolha na validação. `pcen_large` é a rede previamente escolhida na rodada anterior e `previous_round3_extra_trees_regularized` é o modelo escolhido na rodada 3: ambos são referências preservadas. Não promovemos um vencedor retrospectivamente pelo teste.')
    body.append(table(['Modelo','Escolhido antes do teste','Precisão','Recall Aedes','F1 Aedes','Macro-F1','Especificidade','Acurácia'],[[r.model,'sim' if r.selected else 'não',pct(r.aedes_precision),pct(r.aedes_recall),pct(r.aedes_f1),pct(r.macro_f1),pct(r.specificity),pct(r.accuracy)] for r in m[m.scope.eq('all_fresh')].itertuples()]))
    negative_total=int(allnew.true_negative+allnew.false_positive+allnew.negative_invalid)
    body.append(f"Neste teste, responder sempre ‘não Aedes’ teria **{pct(negative_total/allnew.known_windows)} de acurácia e zero sensibilidade**. Isso demonstra por que uma taxa geral perto de 60% não torna o detector confiável. A ROC-AUC dos scores brutos do escolhido foi {allnew.roc_auc:.3f}, com average precision {allnew.average_precision:.3f}; a prevalência observada de Aedes é {pct(1-negative_total/allnew.known_windows)}. Esses diagnósticos também não mostram separação forte. Não escolhemos um novo limiar pelas curvas do teste.")
    body.append('## Resultado por condição e táxon')
    body.append(table(['Condição','Janelas','Precisão binária','Recall Aedes','Especificidade','Falsos positivos','Macro-F1'],[[r.scope,int(r.known_windows),pct(r.aedes_precision) if pd.notna(r.macro_f1) else 'só uma classe',pct(r.aedes_recall),pct(r.specificity),pct(r.emitted_false_positive_rate),pct(r.macro_f1)] for r in s[~s.scope.str.startswith('taxon:')].itertuples()]))
    body.append(table(['Referência','Grupos/arquivos','Janelas','Aedes emitido','Não Aedes emitido','Inválidos','Acerto relevante'],[[r.scope.removeprefix('taxon:'),int(r.groups),int(r.known_windows),int(r.true_positive+r.false_positive),int(r.true_negative+r.false_negative),int(r.invalid),pct(r.aedes_recall if r.scope=='taxon:Aedes aegypti' else r.specificity)] for r in s[s.scope.str.startswith('taxon:')].itertuples()]))
    body.append('Esse detector só responde Aedes/não Aedes. A tabela por táxon mede rejeição dos negativos conhecidos; não demonstra identificação multiclasse. Complexos taxonômicos permanecem incompletamente identificados.')
    matrix=json.loads(allnew.confusion);body.append('## Matriz de confusão operacional')
    body.append(table(['Referência','Não Aedes','Aedes','Inválido'],[[label,*matrix[i]] for i,label in enumerate(['Não Aedes','Aedes'])]))
    body.append('## Incerteza e melhora comparada\n\nIntervalos de 95% com 2.000 reamostragens de gravações/arquivos inteiros, estratificados por fonte; CD-MSC não fornece identidade biológica. Não tratar janelas como amostras independentes.')
    body.append(table(['Condição','Métrica','IC 95%'],[[scope,k,f"{pct(v['lower95'])}–{pct(v['upper95'])}"] for scope,items in ci[chosen].items() for k,v in items['metrics'].items()]))
    body.append('A comparação pareada usa os mesmos arquivos e os mesmos sorteios de grupos: novo detector menos ExtraTrees escolhido na rodada 3. Diferenças são exploratórias, condicionais a essa amostra. Ganho parcial não demonstra confiabilidade geral.')
    body.append(table(['Condição','Métrica','Mudança em pontos percentuais','IC 95% da mudança'],[[scope,k,f"{100*v['change']:+.1f}",f"{100*v['lower95']:+.1f} a {100*v['upper95']:+.1f}"] for scope,items in paired['differences'].items() for k,v in items.items()]))
    body.append('## Áudio privado do Drive\n\nReferência do usuário: Aedes em todo o enxame, caixa de vidro e Nano 33 BLE Sense. São 107 janelas de uma gravação conhecida, com duas inválidas no denominador. Não é teste cego e só mede sensibilidade nesse áudio. Precisão aparente de 100% em um arquivo só de positivos não comprova precisão binária. A gravação não entrou no treino e não será publicada.')
    body.append(table(['Modelo','Selecionado','Aedes detectados /107','Sensibilidade nesse áudio'],[[r.model,'sim' if r.selected else 'não',f'{int(r.true_positive)}/{int(r.known_windows)}',pct(r.aedes_recall)] for r in private.itertuples()]))
    body.append('## Bases históricas\n\nDiagnóstico de regressões; estas bases já foram examinadas no desenvolvimento. Não substituem o novo teste.')
    body.append(table(['Fonte','Precisão','Recall Aedes','Especificidade','Macro-F1'],[[r.scope,pct(r.aedes_precision) if pd.notna(r.macro_f1) else 'só Aedes',pct(r.aedes_recall),pct(r.specificity),pct(r.macro_f1)] for r in hist[hist.selected].itertuples()]))
    body.append('## Explicabilidade\n\nA cada janela pública, registramos score de presença de mosquito, score condicional de Aedes e score final. Isso localiza em qual estágio o alerta falha; não é prova de calibração em campo ou de relevância biológica causal. O encoder combina PCEN, log-Mel e contraste espectral; as features físicas incluem frequência, harmônicos, MFCC e dinâmica, também sensíveis a sexo, temperatura e microfone.')
    stages=pd.read_csv(OUT/'stage_explanations.csv');body.append(table(['Referência','Presença, mediana','Aedes condicional, mediana','Final calibrado, mediana'],[[r.reference_species,f'{r.presence_stage_score:.3f}',f'{r.conditional_species_score:.3f}',f'{r.score:.3f}'] for r in stages[stages.model.eq(chosen)].itertuples()]))
    body.append('A importância por impureza das árvores está em `classical_feature_importance_exploratory.csv`; é sensível a correlação e não explica causalmente a rede. Não afirmamos que um harmônico ou feature identifique exclusivamente Aedes.')
    body.append('## Decisão\n\n'+state+'\n\n'+('\n'.join('- '+x for x in d['failure_reasons']) if d['failure_reasons'] else 'Os mínimos foram atendidos nesta amostra, dentro das limitações descritas.'))
    body.append('''Não podemos garantir que repetir treinamentos nos mesmos corpora produza um detector confiável. Escolher por testes já vistos apenas aumenta o risco de memorizar suas condições. São necessárias novas gravações com identidade/sessão conhecidas, Aedes e negativos confirmados no mesmo aparelho e caixa, incluindo Aedes albopictus e Culex, ruídos, sexos e temperaturas variados. Esses dados devem ser separados por sessão/indivíduo antes da segmentação.

Nenhum firmware, Arduino ou instalador foi modificado: 26 arquivos históricos foram verificados por SHA-256. A comparação é dos modelos em computador, sem compactação. Não afirmamos superar os estudos: tarefa, protocolos e condições de teste não são os mesmos.

## Artefatos e execução

Este notebook apenas lê resultados e produz gráficos. Para reavaliar os modelos congelados localmente:

```bash
.venv/bin/python evaluate_sex_balanced_reliability.py
.venv/bin/python -m unittest tests.test_sex_balanced_reliability tests.test_reliability tests.test_reproduction tests.test_corrected_validation
.venv/bin/python summarize_sex_balanced_reliability.py
```

Os pesos novos e o cache são locais, ignorados pelo Git. Os JSONs/CSVs de `results/sex_balanced_reliability_2026_10_01/` guardam partições, seleção anterior ao teste, hashes, horários, intervalos e decisão. Uma nova rodada deve usar outro protocolo, diretório e outro teste após congelamento.
''')
    report=ROOT/'reports/confiabilidade_binaria_aedes_adaptacao_2026_10_01.md';report.write_text('\n\n'.join(body))
    nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell(body[0]),nbformat.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd,numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import precision_recall_curve\nROOT=Path.cwd()\nif not (ROOT/'results/sex_balanced_reliability_2026_10_01').exists(): ROOT=ROOT.parent\nO=ROOT/'results/sex_balanced_reliability_2026_10_01'\ndecision=json.loads((O/'reliability_decision.json').read_text())\nselected=decision['selected_model']\nmetrics=pd.read_csv(O/'fresh_metrics.csv')\ndisplay(decision)"),
        nbformat.v4.new_markdown_cell('## Comparação e matriz de confusão: o modelo escolhido na validação permanece identificado'),
        nbformat.v4.new_code_cell("display(pd.read_json(O/'validation_search.json')[['name','validation_precision','validation_recall','validation_macro_f1','validation_minimum_recall','cut']])\ndisplay(metrics[metrics.scope.eq('all_fresh')][['model','selected','known_windows','aedes_precision','aedes_recall','aedes_f1','macro_f1','specificity','accuracy']])\ndisplay(metrics[metrics.selected & metrics.scope.str.startswith('taxon:')])\nr=metrics[metrics.selected & metrics.scope.eq('all_fresh')].iloc[0]\na=np.array(json.loads(r.confusion))\nfig,ax=plt.subplots(figsize=(6,3));ax.imshow(a,cmap='Blues')\nfor (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>a.max()/2 else 'black')\nax.set_xticks([0,1,2],['Não Aedes','Aedes','Inválido']);ax.set_yticks([0,1],['Não Aedes','Aedes'])\nax.set(xlabel='Emissão',ylabel='Referência',title=selected);fig.tight_layout()\nassets=ROOT/'reports/assets';assets.mkdir(exist_ok=True)\nfig.savefig(assets/'aedes_adaptation_confusion.png',dpi=200)\nplt.show()"),
        nbformat.v4.new_markdown_cell('## Curvas precisão–sensibilidade\n\nScores brutos, sem escolher novo limiar por estas curvas; a matriz operacional mantém os inválidos.'),
        nbformat.v4.new_code_cell("pred=pd.read_csv(O/'fresh_predictions.csv.gz')\nfig,axes=plt.subplots(1,2,figsize=(12,4))\nfor ax,scope in zip(axes,['all_fresh','mosquito_only']):\n    sample=pred if scope=='all_fresh' else pred[pred.reference_species.ne('noise')]\n    for name,s in sample.groupby('model'):\n        precision,recall,_=precision_recall_curve(s.label,s.score)\n        ax.plot(recall,precision,label=name,linewidth=2 if name==selected else 1)\n    ax.axhline(sample[sample.model.eq(selected)].label.mean(),color='gray',linestyle='--',label='prevalência')\n    ax.set(xlabel='Sensibilidade',ylabel='Precisão',title=scope,xlim=(0,1),ylim=(0,1))\naxes[1].legend(fontsize=7,bbox_to_anchor=(1.02,1));fig.tight_layout()\nfig.savefig(assets/'aedes_adaptation_precision_recall.png',dpi=200,bbox_inches='tight')\nplt.show()"),
        nbformat.v4.new_markdown_cell('## Áudio do Drive e incerteza\n\nUma gravação positiva conhecida mede sensibilidade, não especificidade/precisão binária.'),
        nbformat.v4.new_code_cell("display(pd.read_csv(O/'private_audio_aggregate_exploratory.csv')[['model','selected','true_positive','known_windows','aedes_recall','invalid','precision_informative_for_binary_detection']])\ndisplay(json.loads((O/'fresh_group_bootstrap.json').read_text())[selected])\ndisplay(json.loads((O/'paired_improvement_exploratory.json').read_text()))\ndisplay(pd.read_csv(O/'stage_explanations.csv').query('model==@selected'))"),
        nbformat.v4.new_markdown_cell('\n\n'.join(body[1:])),nbformat.v4.new_markdown_cell('## Decisão final: somente leitura'),nbformat.v4.new_code_cell('decision')]
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(nb,ROOT/'confiabilidade_binaria_aedes_adaptacao_2026_10_01.ipynb');print('ADAPTATION REPORT AND NOTEBOOK GENERATED',flush=True)


if __name__=='__main__':main()
