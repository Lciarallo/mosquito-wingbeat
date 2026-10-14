# Detector binário de Aedes: relatório final da adaptação — 01/10/2026

**Ainda não confiável: não atingiu os mínimos definidos antes do novo teste.** O modelo escolhido antes do teste foi **pcen_embedding_logistic**. Precisão no novo teste: **17.5%**; sensibilidade: **2.8%**; F1 Aedes: **4.8%**; macro-F1: **37.6%**. Acurácia geral: 54.7%. A decisão não usa apenas a última métrica.

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


| Candidato | Precisão validação | Recall validação | Macro-F1 validação | Menor recall por fonte/classe | Limiar final |
| --- | --- | --- | --- | --- | --- |
| trees_regularized_female | 25.7% | 60.2% | 65.6% | 57.3% | 0.6764 |
| trees_deep_female | 25.2% | 63.9% | 65.5% | 62.0% | 0.6569 |
| hist_boost_female | 35.8% | 59.3% | 70.7% | 55.8% | 0.7830 |
| pcen_embedding_logistic | 41.3% | 71.8% | 74.8% | 67.5% | 0.7789 |

Congelamento: **2026-10-01T22:31:17.742296+00:00**. Seleção do novo teste: **2026-10-01T22:32:49.948712+00:00**. O vencedor maximiza o mínimo de precisão, sensibilidade e recalls por fonte/classe na validação, com pequeno termo de macro-F1. Calibração e escolha do limiar usam apenas a partição de calibração. Os modelos e pesos anteriores permaneceram idênticos por SHA-256.

## Outro teste realmente novo após o congelamento

Foram adquiridos **168 arquivos**, dos quais 151 foram avaliados após a regra conservadora de exclusão de colisões de conteúdo/PCM. Nenhum grupo de aquisição ou arquivo exato coincide com o teste anterior. Não substituímos gravações depois de observar resultados.

Aedes positivos e negativos Aedes albopictus/Culex/Anopheles vêm do mesmo domínio acústico CD-MSC D3. Incluímos ainda novos grupos HumBug e fontes ESC-50 do fold 5 não utilizadas anteriormente. Não usamos o domínio D5 óptico. Fontes: [CD-MSC 2026](https://zenodo.org/records/19095788), [HumBugDB](https://zenodo.org/records/4904800) e [ESC-50](https://github.com/karolpiczak/ESC-50).

**Limites importantes:** identidade dos indivíduos/sessões CD-MSC indisponível; novidade por arquivo não comprova uma campanha ou indivíduo independente. HumBug adiciona negativos novos, sem positivos novos da mesma fonte. O rótulo público é do arquivo, sem anotação de presença de batimento em cada janela. Os resultados são concordância com essa referência, não identificação entomológica de cada som. Janelas de um arquivo são correlacionadas.

O campo `duplicate_file` da auditoria marca qualquer colisão de janela PCM quantizada, não necessariamente duplicação do arquivo original. Na rodada anterior, 16 exclusões ESC-50 foram explicadas por 37 janelas silenciosas/constantes iguais; a regra conservadora foi mantida. Nesta rodada, a lista de exclusões é preservada no JSON de decisão, sem substituição por score.


| Fonte | Arquivos selecionados | Segundos |
| --- | --- | --- |
| CDMSC2026_D3 | 48 | 968.3 |
| ESC50_noise | 91 | 455.0 |
| HumBug_noise | 3 | 69.6 |
| HumBug_other_mosquito | 26 | 452.2 |

## Comparação no mesmo teste novo

Somente os quatro candidatos da adaptação participaram da escolha na validação. `pcen_large` é a rede previamente escolhida na rodada anterior e `previous_round3_extra_trees_regularized` é o modelo escolhido na rodada 3: ambos são referências preservadas. Não promovemos um vencedor retrospectivamente pelo teste.

| Modelo | Escolhido antes do teste | Precisão | Recall Aedes | F1 Aedes | Macro-F1 | Especificidade | Acurácia |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hist_boost_female | não | 34.6% | 4.8% | 8.5% | 40.3% | 93.0% | 57.2% |
| pcen_embedding_logistic | sim | 17.5% | 2.8% | 4.8% | 37.6% | 90.1% | 54.7% |
| pcen_large | não | 15.9% | 5.2% | 7.9% | 36.7% | 80.3% | 49.9% |
| previous_round3_extra_trees_regularized | não | 37.4% | 48.9% | 42.4% | 45.5% | 43.4% | 45.6% |
| trees_deep_female | não | 37.2% | 9.9% | 15.6% | 43.0% | 87.8% | 56.2% |
| trees_regularized_female | não | 35.9% | 8.8% | 14.2% | 42.4% | 88.4% | 56.2% |

Neste teste, responder sempre ‘não Aedes’ teria **59.5% de acurácia e zero sensibilidade**. Isso demonstra por que uma taxa geral perto de 60% não torna o detector confiável. A ROC-AUC dos scores brutos do escolhido foi 0.553, com average precision 0.401; a prevalência observada de Aedes é 40.5%. Esses diagnósticos também não mostram separação forte. Não escolhemos um novo limiar pelas curvas do teste.

## Resultado por condição e táxon

| Condição | Janelas | Precisão binária | Recall Aedes | Especificidade | Falsos positivos | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| all_fresh | 1843 | 17.5% | 2.8% | 90.1% | 9.0% | 37.6% |
| mosquito_only | 1404 | 17.6% | 2.8% | 85.1% | 14.9% | 31.2% |
| CDMSC2026_D3 | 961 | 61.8% | 2.8% | 93.9% | 6.1% | 20.3% |
| ESC50_noise | 370 | só uma classe | — | 97.3% | 0.3% | — |
| HumBug_noise | 69 | só uma classe | — | 100.0% | 0.0% | — |
| HumBug_other_mosquito | 443 | só uma classe | — | 80.8% | 19.2% | — |

| Referência | Grupos/arquivos | Janelas | Aedes emitido | Não Aedes emitido | Inválidos | Acerto relevante |
| --- | --- | --- | --- | --- | --- | --- |
| Aedes aegypti | 24 | 747 | 21 | 726 | 0 | 2.8% |
| Aedes albopictus | 6 | 178 | 1 | 177 | 0 | 99.4% |
| Anopheles arabiensis | 3 | 6 | 2 | 4 | 0 | 66.7% |
| Anopheles coustani | 3 | 51 | 2 | 49 | 0 | 96.1% |
| Anopheles funestus sl | 3 | 50 | 1 | 49 | 0 | 98.0% |
| Anopheles funestus ss | 3 | 68 | 41 | 27 | 0 | 39.7% |
| Anopheles harrisoni | 2 | 8 | 0 | 8 | 0 | 100.0% |
| Anopheles maculipalpis | 3 | 91 | 0 | 91 | 0 | 100.0% |
| Anopheles minimus | 6 | 12 | 0 | 12 | 0 | 100.0% |
| Anopheles squamosus | 3 | 55 | 21 | 34 | 0 | 61.8% |
| Anopheles stephensi | 6 | 12 | 0 | 12 | 0 | 100.0% |
| Culex pipiens | 6 | 12 | 12 | 0 | 0 | 0.0% |
| Culex pipiens complex | 3 | 100 | 18 | 82 | 0 | 82.0% |
| Mansonia uniformis | 3 | 14 | 0 | 14 | 0 | 100.0% |
| noise | 77 | 439 | 1 | 429 | 9 | 97.7% |

Esse detector só responde Aedes/não Aedes. A tabela por táxon mede rejeição dos negativos conhecidos; não demonstra identificação multiclasse. Complexos taxonômicos permanecem incompletamente identificados.

## Matriz de confusão operacional

| Referência | Não Aedes | Aedes | Inválido |
| --- | --- | --- | --- |
| Não Aedes | 988 | 99 | 9 |
| Aedes | 726 | 21 | 0 |

## Incerteza e melhora comparada

Intervalos de 95% com 2.000 reamostragens de gravações/arquivos inteiros, estratificados por fonte; CD-MSC não fornece identidade biológica. Não tratar janelas como amostras independentes.

| Condição | Métrica | IC 95% |
| --- | --- | --- |
| all_fresh | aedes_precision | 6.8%–35.5% |
| all_fresh | aedes_recall | 1.2%–4.6% |
| all_fresh | aedes_f1 | 2.1%–7.9% |
| all_fresh | specificity | 84.7%–94.7% |
| all_fresh | macro_f1 | 33.5%–41.3% |
| all_fresh | balanced_accuracy | 43.4%–49.0% |
| mosquito_only | aedes_precision | 7.0%–34.9% |
| mosquito_only | aedes_recall | 1.2%–4.6% |
| mosquito_only | aedes_f1 | 2.1%–7.8% |
| mosquito_only | specificity | 75.4%–92.5% |
| mosquito_only | macro_f1 | 24.8%–36.9% |
| mosquito_only | balanced_accuracy | 39.0%–47.7% |
| CDMSC2026_D3 | aedes_precision | 35.0%–84.6% |
| CDMSC2026_D3 | aedes_recall | 1.2%–4.6% |
| CDMSC2026_D3 | aedes_f1 | 2.4%–8.7% |
| CDMSC2026_D3 | specificity | 83.5%–98.0% |
| CDMSC2026_D3 | macro_f1 | 10.5%–29.4% |
| CDMSC2026_D3 | balanced_accuracy | 43.0%–50.6% |

A comparação pareada usa os mesmos arquivos e os mesmos sorteios de grupos: novo detector menos ExtraTrees escolhido na rodada 3. Diferenças são exploratórias, condicionais a essa amostra. Ganho parcial não demonstra confiabilidade geral.

| Condição | Métrica | Mudança em pontos percentuais | IC 95% da mudança |
| --- | --- | --- | --- |
| all_fresh | aedes_precision | -19.9 | -31.6 a -4.7 |
| all_fresh | aedes_recall | -46.1 | -54.4 a -38.4 |
| all_fresh | aedes_f1 | -37.5 | -45.4 a -28.4 |
| all_fresh | specificity | +46.7 | +38.7 a +53.8 |
| all_fresh | macro_f1 | -8.0 | -15.6 a -0.2 |
| all_fresh | balanced_accuracy | +0.3 | -5.3 a +5.5 |
| mosquito_only | aedes_precision | -25.0 | -36.6 a -9.1 |
| mosquito_only | aedes_recall | -46.1 | -54.4 a -38.4 |
| mosquito_only | aedes_f1 | -40.7 | -49.1 a -31.2 |
| mosquito_only | specificity | +59.7 | +49.3 a +68.2 |
| mosquito_only | macro_f1 | -5.4 | -14.7 a +3.9 |
| mosquito_only | balanced_accuracy | +6.8 | +0.1 a +12.7 |
| CDMSC2026_D3 | aedes_precision | -6.8 | -34.2 a +18.0 |
| CDMSC2026_D3 | aedes_recall | -46.1 | -54.4 a -38.4 |
| CDMSC2026_D3 | aedes_f1 | -51.7 | -59.8 a -41.8 |
| CDMSC2026_D3 | specificity | +72.0 | +60.0 a +77.2 |
| CDMSC2026_D3 | macro_f1 | -15.5 | -24.7 a -5.2 |
| CDMSC2026_D3 | balanced_accuracy | +13.0 | +5.7 a +17.6 |

## Áudio privado do Drive

Referência do usuário: Aedes em todo o enxame, caixa de vidro e Nano 33 BLE Sense. São 107 janelas de uma gravação conhecida, com duas inválidas no denominador. Não é teste cego e só mede sensibilidade nesse áudio. Precisão aparente de 100% em um arquivo só de positivos não comprova precisão binária. A gravação não entrou no treino e não será publicada.

| Modelo | Selecionado | Aedes detectados /107 | Sensibilidade nesse áudio |
| --- | --- | --- | --- |
| trees_regularized_female | não | 46/107 | 43.0% |
| trees_deep_female | não | 55/107 | 51.4% |
| hist_boost_female | não | 14/107 | 13.1% |
| pcen_embedding_logistic | sim | 0/107 | 0.0% |
| pcen_large | não | 39/107 | 36.4% |
| previous_round3_extra_trees_regularized | não | 77/107 | 72.0% |

## Bases históricas

Diagnóstico de regressões; estas bases já foram examinadas no desenvolvimento. Não substituem o novo teste.

| Fonte | Precisão | Recall Aedes | Especificidade | Macro-F1 |
| --- | --- | --- | --- | --- |
| dryad | 62.1% | 86.1% | 97.5% | 85.3% |
| humbug | 94.0% | 81.8% | 99.1% | 92.9% |
| mosquitosong | só Aedes | 12.5% | — | — |
| ufrgs | 97.8% | 16.2% | 99.3% | 41.9% |

## Explicabilidade

A cada janela pública, registramos score de presença de mosquito, score condicional de Aedes e score final. Isso localiza em qual estágio o alerta falha; não é prova de calibração em campo ou de relevância biológica causal. O encoder combina PCEN, log-Mel e contraste espectral; as features físicas incluem frequência, harmônicos, MFCC e dinâmica, também sensíveis a sexo, temperatura e microfone.

| Referência | Presença, mediana | Aedes condicional, mediana | Final calibrado, mediana |
| --- | --- | --- | --- |
| Aedes aegypti | 1.000 | 0.000 | 0.088 |
| Aedes albopictus | 1.000 | 0.000 | 0.050 |
| Anopheles arabiensis | 1.000 | 0.000 | 0.077 |
| Anopheles coustani | 0.999 | 0.000 | 0.086 |
| Anopheles funestus sl | 1.000 | 0.000 | 0.050 |
| Anopheles funestus ss | 1.000 | 0.405 | 0.866 |
| Anopheles harrisoni | 1.000 | 0.000 | 0.068 |
| Anopheles maculipalpis | 1.000 | 0.000 | 0.123 |
| Anopheles minimus | 0.678 | 0.000 | 0.058 |
| Anopheles squamosus | 1.000 | 0.036 | 0.673 |
| Anopheles stephensi | 0.999 | 0.000 | 0.086 |
| Culex pipiens | 1.000 | 0.955 | 0.958 |
| Culex pipiens complex | 1.000 | 0.000 | 0.295 |
| Mansonia uniformis | 0.998 | 0.000 | 0.119 |
| noise | 0.000 | 0.000 | 0.050 |

A importância por impureza das árvores está em `classical_feature_importance_exploratory.csv`; é sensível a correlação e não explica causalmente a rede. Não afirmamos que um harmônico ou feature identifique exclusivamente Aedes.

## Decisão

Ainda não confiável: não atingiu os mínimos definidos antes do novo teste.

- all_fresh: aedes_precision below .60
- all_fresh: aedes_precision lower95 below .60
- all_fresh: aedes_recall below .60
- all_fresh: aedes_recall lower95 below .60
- all_fresh: macro_f1 below .60
- all_fresh: macro_f1 lower95 below .60
- mosquito_only: aedes_precision below .60
- mosquito_only: aedes_precision lower95 below .60
- mosquito_only: aedes_recall below .60
- mosquito_only: aedes_recall lower95 below .60
- mosquito_only: macro_f1 below .60
- mosquito_only: macro_f1 lower95 below .60
- CDMSC2026_D3: aedes_precision lower95 below .60
- CDMSC2026_D3: aedes_recall below .60
- CDMSC2026_D3: aedes_recall lower95 below .60
- CDMSC2026_D3: macro_f1 below .60
- CDMSC2026_D3: macro_f1 lower95 below .60
- taxon:Aedes aegypti: aedes_recall below .60
- taxon:Anopheles funestus ss: specificity below .60
- taxon:Culex pipiens: specificity below .60

Não podemos garantir que repetir treinamentos nos mesmos corpora produza um detector confiável. Escolher por testes já vistos apenas aumenta o risco de memorizar suas condições. São necessárias novas gravações com identidade/sessão conhecidas, Aedes e negativos confirmados no mesmo aparelho e caixa, incluindo Aedes albopictus e Culex, ruídos, sexos e temperaturas variados. Esses dados devem ser separados por sessão/indivíduo antes da segmentação.

Nenhum firmware, Arduino ou instalador foi modificado: 26 arquivos históricos foram verificados por SHA-256. A comparação é dos modelos em computador, sem compactação. Não afirmamos superar os estudos: tarefa, protocolos e condições de teste não são os mesmos.

## Artefatos e execução

Este notebook apenas lê resultados e produz gráficos. Para reavaliar os modelos congelados localmente:

```bash
.venv/bin/python evaluate_sex_balanced_reliability.py
.venv/bin/python -m unittest tests.test_sex_balanced_reliability tests.test_reliability tests.test_reproduction tests.test_corrected_validation
.venv/bin/python summarize_sex_balanced_reliability.py
```

Os pesos novos e o cache são locais, ignorados pelo Git. Os JSONs/CSVs de `results/sex_balanced_reliability_2026_10_01/` guardam partições, seleção anterior ao teste, hashes, horários, intervalos e decisão. Uma nova rodada deve usar outro protocolo, diretório e outro teste após congelamento.
