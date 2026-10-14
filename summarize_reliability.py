"""Create report and executable read-only notebook from audited frozen outputs."""
import json
import numpy as np,pandas as pd,nbformat
from train_reproduction import ROOT,sha
from fetch_reliability_data import OUT
from evaluate_reliability import verify


def pct(x):return '—' if x is None or pd.isna(x) else f'{100*x:.1f}%'
def table(cols,rows):
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    freeze=verify();decision=json.loads((OUT/'reliability_decision.json').read_text());fresh=pd.read_csv(OUT/'fresh_metrics.csv')
    private=pd.read_csv(OUT/'private_audio_aggregate_exploratory.csv');history=pd.read_csv(OUT/'historical_metrics_exploratory.csv')
    plan=json.loads((OUT/'fresh_selection.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text());search=json.loads((OUT/'validation_search.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text());ci=json.loads((OUT/'fresh_group_bootstrap.json').read_text());chosen=decision['selected_model']
    selected=fresh[fresh.selected];allnew=selected[selected.scope.eq('all_fresh')].iloc[0];mosq=selected[selected.scope.eq('mosquito_only')].iloc[0]
    taxonomy=selected[selected.scope.str.startswith('taxon:')];manifest=pd.DataFrame(plan['files']);added=json.loads((OUT/'additional_selection.json').read_text())
    state='Superou os mínimos offline nesta amostra; a validação nas condições de uso permanece pendente.' if decision['offline_minimum_criteria_passed'] else 'Ainda não confiável: falhou nos mínimos offline definidos antes do teste.'
    body=[f'''# Confiabilidade do detector binário de Aedes aegypti — 01/10/2026

**{state}** O modelo selecionado na validação foi **{chosen}**. No novo teste, sua precisão foi {pct(allnew.aedes_precision)}, sensibilidade {pct(allnew.aedes_recall)}, F1 de Aedes {pct(allnew.aedes_f1)} e macro-F1 {pct(allnew.macro_f1)}. Entre apenas mosquitos, a especificidade foi {pct(mosq.specificity)}. Não houve compilação, redução de modelo ou alteração de Arduino/instalador.

## O que conta como evidência de confiabilidade

Foi incorporado o [artigo indicado pelo usuário sobre os limites da acurácia](https://www.dio.me/articles/por-que-a-acuracia-accuracy-nao-e-a-melhor-metrica-para-avaliar-o-desempenho-de-classificadores). Se 99% da amostra for negativa, dizer sempre “não Aedes” dá 99% de acurácia e zero sensibilidade. Por isso, não usamos acurácia média acima de 60% como aprovação. As definições foram conferidas na [documentação oficial do Google](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall).

- Precisão: dos alertas Aedes emitidos, quantos eram Aedes; depende da composição da amostra.
- Sensibilidade/recall: dos Aedes presentes, quantos foram detectados.
- Especificidade e falsos positivos: capacidade de rejeitar outros mosquitos e ruídos.
- F1 de Aedes e macro-F1: mostram o equilíbrio entre acertos/erros, incluindo as duas classes.
- Matriz de confusão, inválidos e cobertura: preservam os erros e as falhas de emissão no denominador.

O mínimo solicitado de 60% foi registrado no protocolo antes do teste: precisão, sensibilidade, macro-F1 e especificidade devem superar esse patamar tanto no conjunto completo quanto entre apenas mosquitos; cada táxon negativo deve ter especificidade de ao menos 60%. Exigimos também limites inferiores de 95% de ao menos 60% nas métricas principais por reamostragem de gravações. Esses mínimos são um filtro experimental, não uma certificação de uso.

O artigo contém uma inversão na explicação final do limiar: com scores fixos, aumentar o limiar nunca aumenta a sensibilidade. A precisão costuma aumentar, mas não é garantidamente monotônica; melhorar o modelo pode melhorar precisão e sensibilidade simultaneamente. [Explicação oficial do limiar](https://developers.google.com/machine-learning/crash-course/classification/thresholding).
''']
    body.append(f'''## Treinamento ampliado e estratégias executadas

Foram adquiridas **{len(added['files'])} gravações novas** para treino/validação/calibração, por regra de metadados registrada antes da inferência. Incluem HumBug e ESC-50; o conjunto ampliado tem **49.713 janelas** em todas as partições, não 49.713 observações biológicas independentes. Entraram também 30 negativos HumBug usados nos testes das rodadas anteriores: agora são adaptação, nunca teste inédito desta rodada.

Comparação pré-definida: quatro redes neurais e três modelos clássicos. Todas as redes têm cabeça de presença de mosquito e cabeça condicional de espécie, combinadas por P(mosquito) × P(Aedes | mosquito), com calibração Platt posterior. Esse produto é uma regra do detector, não prova de calibração probabilística perfeita. Os modelos clássicos usam os mesmos dois estágios, treinados separadamente.

- CNN residual expandida: 2.792.899 parâmetros.
- PCEN residual: 9.583.874 parâmetros; PCEN normaliza energia com uma média móvel causal, combinado com log-Mel relativo e contraste espectral.
- PCEN híbrida: 9.653.122 parâmetros, com 138 features acústicas adicionais, normalizadas somente no treino.
- CNN 1D em áudio bruto: 348.163 parâmetros, adaptada da arquitetura MosquitoSong+.
- ExtraTrees com profundidade 16, ExtraTrees sem limite de profundidade e HistGradientBoosting: alternativas em features acústicas.

Os lotes têm 48 Aedes, 48 outros mosquitos e 32 ruídos, com equilíbrio por fonte, táxon e gravação. O treino inclui ruído real em SNR 5–30 dB, equalização aleatória para simular diferenças de microfone e superposição de sons da mesma classe binária. Essas misturas são **simulações de treinamento**, não gravações independentes de enxames. Nenhum áudio privado, UFRGS ou CD-MSC D3 de Aedes entrou no treino.

Pré-processamento: mono, reamostragem real, retirada de componente DC e normalização de RMS. Entradas reais de 0,3 s ou 0,992 s; não repetimos, preenchemos ou concatenamos áudio para fabricar uma observação. A CNN 1D usa três recortes reais disjuntos quando recebe uma janela longa. O modelo híbrido combina features do trecho original com a onda aumentada durante treino; as features não são recalculadas após cada aumento.

Treino, validação, calibração e teste foram separados por gravação e hashes; as 212 janelas de mosquito sem espécie continuam não rotuladas. Complexos cujo gênero é comprovadamente não Aedes podem ser negativos binários, mas não equivalem a identificação exata de espécie. A categoria ESC-50 “insects” foi excluída por ambiguidade de presença de mosquito.
''')
    body.append(table(['Candidato','Época selecionada','Menor recall de validação por fonte/classe','Precisão na validação','Limiar da calibração'],[[r['name'],r.get('epoch','árvores'),pct(r['validation_minimum_recall']),pct(r['validation_precision']),f"{r['cut']:.4f}"] for r in search]))
    body.append(f'''Seleção: maior mínimo de recall por fonte/classe na validação, com pequenos termos de macro-F1 e average precision. Limiar: somente calibração, sem rótulos do teste. Congelamento: **{freeze['frozen_utc']}**. Seleção dos novos áudios: **{plan['selection_utc']}**. Todos os pesos e fontes congelados têm SHA-256; os 12 checkpoints anteriores permaneceram idênticos.

## Novo teste baixado depois do congelamento

Foram baixados **{len(plan['files'])} arquivos**, dos quais {decision['fresh_files']} foram avaliados após auditoria de duplicatas. A regra de seleção foi: {plan['rule']}

Positivos: arquivos acústicos CD-MSC 2026, domínio D3, com rótulo Aedes aegypti. Negativos: novos grupos HumBug, fontes ESC-50 do fold 5 e **CD-MSC D3 com Aedes albopictus, Culex pipiens, Anopheles minimus e Anopheles stephensi**. Esse subconjunto inclui as duas classes no mesmo domínio acústico. O domínio D5 óptico foi excluído. [Dataset oficial CD-MSC 2026](https://zenodo.org/records/19095788), [HumBugDB](https://zenodo.org/records/4904800), [ESC-50 e agrupamento das fontes](https://github.com/karolpiczak/ESC-50).

**Limites:** D3 não fornece neste recorte identidade do mosquito ou sessão original; são arquivos novos, não indivíduos ou campanhas comprovadamente novos. HumBug só acrescenta negativos frescos, o que mantém confusão entre classe e fonte no agregado; reportamos também o D3 com as duas classes. O rótulo é do arquivo e não uma anotação de presença de batimento em cada instante. Isso impede extrapolar o teste a qualquer microfone, caixa ou enxame. O bootstrap por arquivo não corrige essa ausência de metadados.
''')
    body.append(table(['Fonte','Arquivos','Segundos'],[[source,len(s),f'{s.seconds.sum():.1f}'] for source,s in manifest.groupby('source')]))
    body.append('A auditoria excluiu conservadoramente 16 arquivos ESC-50 que continham alguma janela PCM quantizada igual a uma janela anterior. O esclarecimento posterior mostrou que as 37 correspondências eram janelas constantes/silenciosas, não cópias demonstradas das gravações inteiras. A coluna `duplicate_file` sinaliza esse critério de colisão de janela. Mantivemos a exclusão pré-definida e não escolhemos substitutos; os resultados não foram recalculados para favorecer o modelo. Ver `novelty_clarification.json`.')
    body.append('## Comparação de todos os candidatos congelados\n\nO vencedor foi escolhido antes do teste. Um candidato com melhor resultado neste teste não é automaticamente promovido: isso usaria o teste para seleção.')
    body.append('A linha `previous_round3_extra_trees_regularized` é o modelo previamente selecionado na rodada 3, preservado e avaliado nos mesmos áudios novos para comparação; não participou da seleção nesta rodada.')
    body.append(table(['Modelo','Selecionado antes do teste','Precisão','Recall Aedes','F1 Aedes','Macro-F1','Especificidade','Acurácia'],[[r.model,'sim' if r.selected else 'não',pct(r.aedes_precision),pct(r.aedes_recall),pct(r.aedes_f1),pct(r.macro_f1),pct(r.specificity),pct(r.accuracy)] for r in fresh[fresh.scope.eq('all_fresh')].itertuples()]))
    paired=json.loads((OUT/'paired_improvement_exploratory.json').read_text())
    body.append('## Melhorou em relação ao modelo anterior?\n\nDiferenças na mesma amostra, em pontos percentuais: nova rede escolhida na validação menos ExtraTrees escolhido na rodada 3. Intervalos pareados por arquivo/gravação, exploratórios; ganho parcial não implica aprovação de confiabilidade.')
    body.append(table(['Condição','Métrica','Mudança','IC 95% da mudança'],[[scope,k,f"{100*v['change']:+.1f} pp",f"{100*v['lower95']:+.1f} a {100*v['upper95']:+.1f} pp"] for scope,d in paired['differences'].items() for k,v in d.items()]))
    body.append('## Resultado do modelo selecionado por condição')
    body.append(table(['Condição','Janelas','Precisão','Recall Aedes','Especificidade','Falsos positivos','Macro-F1'],[[r.scope,int(r.known_windows),pct(r.aedes_precision) if pd.notna(r.macro_f1) else 'não estimável como detector binário',pct(r.aedes_recall),pct(r.specificity),pct(r.emitted_false_positive_rate),pct(r.macro_f1)] for r in selected[~selected.scope.str.startswith('taxon:')].itertuples()]))
    body.append('## Acertos e erros por táxon\n\nO detector emite somente Aedes/não Aedes; as linhas abaixo medem rejeição de cada táxon conhecido, não classificação multiclasse. Complexos mantêm sua identificação taxonômica incompleta.')
    body.append(table(['Referência','Grupos','Janelas','Aedes emitido','Não Aedes emitido','Inválidos','Taxa de acerto relevante'],[[r.scope.removeprefix('taxon:'),int(r.groups),int(r.known_windows),int(r.true_positive+r.false_positive),int(r.true_negative+r.false_negative),int(r.invalid),pct(r.aedes_recall if r.scope=='taxon:Aedes aegypti' else r.specificity)] for r in taxonomy.itertuples()]))
    matrix=np.array(json.loads(allnew.confusion.replace("'",'"'))) if isinstance(allnew.confusion,str) else np.array(allnew.confusion)
    body.append('## Matriz de confusão do novo teste\n\nLinhas são a referência; colunas são emissões. Inválidos não contam como rejeições corretas.')
    body.append(table(['Referência','Não Aedes','Aedes','Inválido'],[[label,*matrix[i]] for i,label in enumerate(['Não Aedes','Aedes'])]))
    body.append('## Intervalos de incerteza\n\nReamostragem estratificada por fonte com 2.000 repetições de grupos/arquivos inteiros. Mantém a correlação entre janelas; não comprova independência biológica dos arquivos CD-MSC.')
    body.append(table(['Condição','Métrica','IC 95% por gravação/arquivo'],[[scope,m,f"{pct(v['lower95'])}–{pct(v['upper95'])}"] for scope,details in ci[chosen].items() for m,v in details['metrics'].items()]))
    body.append('## Áudio privado do Drive\n\nReferência fornecida pelo usuário: enxame de Aedes em caixa de vidro gravado pelo Nano 33 BLE Sense, com mosquito em todo o áudio. Todos os 107 trechos permanecem no denominador; dois são inválidos. É uma gravação conhecida e não um teste cego. Não foi usada no treino e não será publicada. Como só há positivos, o acerto mede sensibilidade nessa gravação; a precisão aparente de 100% não informa falsos alarmes.')
    body.append(table(['Modelo','Selecionado','Aedes detectados','Sensibilidade nesse áudio','Inválidos'],[[r.model,'sim' if r.selected else 'não',f'{int(r.true_positive)}/{int(r.known_windows)}',pct(r.aedes_recall),int(r.invalid)] for r in private.itertuples()]))
    body.append('## Bases históricas: diagnóstico de regressões\n\nJá examinadas no desenvolvimento; não são um novo teste independente. UFRGS permanece ausente do treino. MosquitoSong challenge só tem Aedes: não fornece precisão/especificidade binárias.')
    body.append(table(['Fonte','Janelas','Precisão','Recall Aedes','Especificidade','Macro-F1'],[[r.scope,int(r.known_windows),pct(r.aedes_precision) if pd.notna(r.macro_f1) else 'somente positivos',pct(r.aedes_recall),pct(r.specificity),pct(r.macro_f1)] for r in history[history.selected].itertuples()]))
    body.append('''## Reprodução dos estudos e rodadas anteriores

As comparações anteriores desta sessão ficam em `results/reproduction_2026_10_01/`. Foram executadas adaptações da CNN Mel de Fernandes, CNN 1D de MosquitoSong+, CNN residual e alinhamento de embeddings entre fontes da mesma espécie, além de árvores. São reproduções de estratégias, **não equivalência exata aos experimentos publicados**: tarefa, recortes, agrupamento e fontes de teste diferem.

- Fernandes: convoluções 32/64/64 e Dense 256, 1.137.601 parâmetros, Mel 60×40; BCE binária em vez dos dois sigmoides originais e interpolação do espectrograma para durações reais.
- MosquitoSong+: arquitetura 1D da figura 2, áudio de 0,3 s; adaptação de quatro classes para uma saída binária.
- Residual: 2.791.873 parâmetros; ruído e misturas de mesma classe no treino. Alinhamento contrastivo inspirado em trabalhos de adaptação de domínio; não reproduz integralmente o DR-BioL.

Rodada 1: rede residual com ruído escolhida na validação. Novo teste de 19 negativos: 334/544 falsos Aedes (61,4%); nos outros mosquitos, 67/103 (65,0%). Não houve positivo acústico novo nessa rodada. Áudio privado: 20/107 detectados.

Rodada 2: busca de 105 ensembles, selecionada só na validação; melhor mínimo de recall por fonte/classe ainda 38,8%. Áudio privado: 55/107. Os negativos da rodada 1 são diagnósticos reutilizados nesta etapa.

Rodada 3: 19 negativos anteriores incorporados ao treino. ExtraTrees regularizado foi escolhido antes do novo teste de 23 arquivos, contendo 12 positivos CD-MSC novos. Detectou 216/358 Aedes (60,3%), mas confundiu 33/47 outros mosquitos (70,2%). Precisão 64,9%, F1 Aedes 62,5%, acurácia balanceada operacional 64,8%. O resultado médio passou de 60% e **continuou inadequado entre espécies**. Outro candidato de árvores foi melhor nesse teste, mas não pode substituir retrospectivamente o vencedor sem uma nova validação. Áudio privado: 77/107.

Os 12 positivos foram escolhidos no prefixo de 8 MiB do diretório central ZIP64, não na lista completa do arquivo; a limitação está registrada no adendo da seleção. A rodada ampliada usa a lista completa de 542.777 entradas, com 192 arquivos D3 de Aedes. Os 12 anteriores estão excluídos do novo teste e de todo treino.

Os resultados publicados nos estudos não comprovam confiabilidade no nosso domínio: não fizemos uma comparação na mesma amostra, rótulos e protocolo. Por isso, não afirmamos superar Fernandes ou MosquitoSong+. A pesquisa bibliográfica está em [estudos_detector_binario_aedes_2026_10_01.md](estudos_detector_binario_aedes_2026_10_01.md).
''')
    body.append('## Explicabilidade e limites\n\nRegistramos os scores dos dois estágios, presença e espécie condicional, em cada janela pública, para localizar a etapa da falha. A tabela abaixo traz medianas por táxon; não são taxas de acerto nem probabilidades comprovadamente calibradas em campo. Na CNN 1D longa, cada estágio é uma média de três recortes e o score conjunto é a média dos produtos por recorte, não o produto das médias.')
    stages=pd.read_csv(OUT/'stage_explanations.csv');stages=stages[stages.model.eq(chosen)]
    body.append(table(['Referência','Score presença, mediana','Score Aedes condicional, mediana','Score final calibrado, mediana'],[[r.reference_species,f'{r.presence_stage_score:.3f}',f'{r.conditional_species_score:.3f}',f'{r.score:.3f}'] for r in stages.itertuples()]))
    body.append('As árvores também têm importância de features por estágio em `classical_feature_importance_exploratory.csv`. É importância por redução de impureza, sensível a features correlacionadas e ao número de valores possíveis; não equivale a relevância causal ou importância fora do treino. Features de frequência fundamental, harmônicos, PSD, dinâmica e MFCC têm interpretação acústica, mas correlacionam com sexo, temperatura, microfone e ambiente. Filtros/PCEN podem remover ruído; não recuperam informação de espécie ausente na gravação. Essa explicação operacional não prova que o modelo aprendeu uma característica biológica exclusiva de Aedes.')
    body.append('## Decisão e próximos dados necessários\n\n'+state+'\n\n'+('\n'.join('- '+reason for reason in decision['failure_reasons']) if decision['failure_reasons'] else 'Não houve falha nos mínimos estatísticos desta amostra offline.'))
    body.append('''Para decidir sobre uso real, precisamos de novas sessões independentes com o sistema pretendido: Aedes e outros mosquitos confirmados, incluindo Aedes albopictus/Culex, ruídos sem mosquito, sexos e temperaturas variados, nas mesmas condições de caixa e microfone. Separar sessões inteiras antes de segmentar. Uma única gravação positiva do Arduino não mede falsos positivos nesse aparelho.

Precisão esperada em outras prevalências só pode ser apresentada como cenário. Usando sensibilidade e falsos positivos do subconjunto de mosquitos, sem supor que sejam constantes em campo:
''')
    body.append(table(['Prevalência hipotética de Aedes','Precisão calculada, não observada'],[[pct(float(p)),pct(v)] for p,v in decision['prevalence_scenarios_not_observed_precision'].items()]))
    body.append('''## Arquivos e reprodução

`results/reliability_2026_10_01/` guarda protocolo, aquisição, partições, resultados, hashes, intervalos e decisão. `confiabilidade_binaria_aedes_2026_10_01.ipynb` somente lê resultados e produz gráficos; não retreina ou compila. Pesos experimentais são arquivos locais ignorados pelo Git. Áudios públicos seguem as licenças das fontes, sem redistribuição automática; o privado permanece em `tmp/`.

Para reavaliar este experimento congelado:

```bash
.venv/bin/python evaluate_reliability.py
.venv/bin/python -m unittest tests.test_reliability tests.test_reproduction tests.test_corrected_validation
.venv/bin/python summarize_reliability.py
```

Para outra rodada, criar outro diretório, protocolo e novo teste: os scripts de treino/aquisição impedem retocar a versão congelada. Não usar o teste final para escolher um limiar ou o modelo vencedor.
''')
    report=ROOT/'reports/confiabilidade_binaria_aedes_2026_10_01.md';report.write_text('\n\n'.join(body))
    nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('\n\n'.join(body[:2])),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd, numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import precision_recall_curve\nROOT = Path.cwd()\nif not (ROOT / 'results/reliability_2026_10_01').exists(): ROOT = ROOT.parent\nO = ROOT / 'results/reliability_2026_10_01'\ndecision = json.loads((O / 'reliability_decision.json').read_text())\nselected = decision['selected_model']\nmetrics = pd.read_csv(O / 'fresh_metrics.csv')\ndisplay(decision)"),
        nbformat.v4.new_markdown_cell('## Treinamento, seleção anterior ao teste e auditoria'),
        nbformat.v4.new_code_cell("display(pd.read_json(O / 'validation_search.json')[['name','validation_minimum_recall','validation_precision','validation_recall','validation_macro_f1','cut']])\ndisplay(json.loads((O / 'freeze.json').read_text()))\ndisplay(pd.read_csv(O / 'fresh_novelty_audit.csv'))"),
        nbformat.v4.new_markdown_cell('## Precisão, sensibilidade e F1 — comparação dos modelos congelados'),
        nbformat.v4.new_code_cell("cols = ['model','selected','known_windows','aedes_precision','aedes_recall','aedes_f1','macro_f1','specificity','accuracy']\ndisplay(metrics[metrics.scope.eq('all_fresh')][cols])\ndisplay(metrics[metrics.scope.eq('mosquito_only')][cols])"),
        nbformat.v4.new_markdown_cell('## Curvas precisão–sensibilidade\n\nDiagnóstico dos scores brutos, sem escolher novo limiar; inválidos permanecem explícitos na matriz operacional. Linhas horizontais mostram a prevalência de Aedes de cada amostra.'),
        nbformat.v4.new_code_cell("pred = pd.read_csv(O / 'fresh_predictions.csv.gz')\nfig, axes = plt.subplots(1,2,figsize=(12,4))\nfor ax, scope in zip(axes,['all_fresh','mosquito_only']):\n    sample = pred if scope == 'all_fresh' else pred[pred.reference_species.ne('noise')]\n    for name, s in sample.groupby('model'):\n        precision, recall, _ = precision_recall_curve(s.label,s.score)\n        ax.plot(recall,precision,label=name,linewidth=2 if name == selected else 1,alpha=1 if name == selected else .6)\n    ax.axhline(sample[sample.model.eq(selected)].label.mean(),color='gray',linestyle='--',label='prevalência')\n    ax.set(xlabel='Sensibilidade (recall)',ylabel='Precisão',title=scope,xlim=(0,1),ylim=(0,1))\naxes[1].legend(fontsize=7,bbox_to_anchor=(1.02,1))\nfig.tight_layout()\nassets=ROOT/'reports/assets'; assets.mkdir(exist_ok=True)\nfig.savefig(assets/'aedes_reliability_precision_recall.png',dpi=200,bbox_inches='tight')\nplt.show()"),
        nbformat.v4.new_markdown_cell('## Matriz com inválidos e acerto por táxon'),
        nbformat.v4.new_code_cell("r=metrics[metrics.model.eq(selected)&metrics.scope.eq('all_fresh')].iloc[0]\nmatrix=np.array(json.loads(r.confusion))\nfig, ax=plt.subplots(figsize=(6,3)); ax.imshow(matrix,cmap='Blues')\nfor (i,j),v in np.ndenumerate(matrix): ax.text(j,i,str(v),ha='center',va='center',color='white' if v>matrix.max()/2 else 'black')\nax.set_xticks([0,1,2],['Não Aedes','Aedes','Inválido']); ax.set_yticks([0,1],['Não Aedes','Aedes'])\nax.set(xlabel='Emissão',ylabel='Referência',title=selected); fig.tight_layout()\nfig.savefig(assets/'aedes_reliability_confusion.png',dpi=200)\nplt.show()\ndisplay(metrics[metrics.model.eq(selected)&metrics.scope.str.startswith('taxon:')][['scope','groups','known_windows','true_positive','false_positive','false_negative','true_negative','invalid','aedes_recall','specificity']])"),
        nbformat.v4.new_markdown_cell('## Intervalos por gravação e áudio privado\n\nAedes no Drive é uma referência conhecida, não um teste cego; não mede especificidade ou precisão binária. CD-MSC tem identidade individual incompleta.'),
        nbformat.v4.new_code_cell("display(json.loads((O / 'fresh_group_bootstrap.json').read_text())[selected])\ndisplay(pd.read_csv(O / 'private_audio_aggregate_exploratory.csv')[['model','selected','known_windows','true_positive','aedes_recall','invalid','precision_informative_for_binary_detection']])\ndisplay(pd.read_csv(O / 'historical_metrics_exploratory.csv'))"),
        nbformat.v4.new_markdown_cell('## Explicação dos estágios e importância das árvores\n\nScores de presença/espécie são diagnósticos. Importância por impureza não é uma explicação causal e não descreve a rede neural.'),
        nbformat.v4.new_code_cell("stages = pd.read_csv(O / 'stage_explanations.csv')\ndisplay(stages[stages.model.eq(selected)])\nimportance = pd.read_csv(O / 'classical_feature_importance_exploratory.csv')\ndisplay(importance.sort_values('impurity_importance',ascending=False).groupby(['model','stage']).head(10))"),
        nbformat.v4.new_markdown_cell('\n\n'.join(body[2:])),
        nbformat.v4.new_markdown_cell('## Dados brutos de decisão\n\nEste notebook não ajusta o modelo nem altera seus limiares.'),nbformat.v4.new_code_cell('decision')]
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(nb,ROOT/'confiabilidade_binaria_aedes_2026_10_01.ipynb');print('REPORT AND NOTEBOOK GENERATED',flush=True)


if __name__=='__main__':main()
