# Confiabilidade do detector binário de Aedes aegypti — 01/10/2026

**Ainda não confiável: falhou nos mínimos offline definidos antes do teste.** O modelo selecionado na validação foi **pcen_large**. No novo teste, sua precisão foi 10.3%, sensibilidade 2.7%, F1 de Aedes 4.2% e macro-F1 35.5%. Entre apenas mosquitos, a especificidade foi 75.7%. Não houve compilação, redução de modelo ou alteração de Arduino/instalador.

## O que conta como evidência de confiabilidade

Foi incorporado o [artigo indicado pelo usuário sobre os limites da acurácia](https://www.dio.me/articles/por-que-a-acuracia-accuracy-nao-e-a-melhor-metrica-para-avaliar-o-desempenho-de-classificadores). Se 99% da amostra for negativa, dizer sempre “não Aedes” dá 99% de acurácia e zero sensibilidade. Por isso, não usamos acurácia média acima de 60% como aprovação. As definições foram conferidas na [documentação oficial do Google](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall).

- Precisão: dos alertas Aedes emitidos, quantos eram Aedes; depende da composição da amostra.
- Sensibilidade/recall: dos Aedes presentes, quantos foram detectados.
- Especificidade e falsos positivos: capacidade de rejeitar outros mosquitos e ruídos.
- F1 de Aedes e macro-F1: mostram o equilíbrio entre acertos/erros, incluindo as duas classes.
- Matriz de confusão, inválidos e cobertura: preservam os erros e as falhas de emissão no denominador.

O mínimo solicitado de 60% foi registrado no protocolo antes do teste: precisão, sensibilidade, macro-F1 e especificidade devem superar esse patamar tanto no conjunto completo quanto entre apenas mosquitos; cada táxon negativo deve ter especificidade de ao menos 60%. Exigimos também limites inferiores de 95% de ao menos 60% nas métricas principais por reamostragem de gravações. Esses mínimos são um filtro experimental, não uma certificação de uso.

O artigo contém uma inversão na explicação final do limiar: com scores fixos, aumentar o limiar nunca aumenta a sensibilidade. A precisão costuma aumentar, mas não é garantidamente monotônica; melhorar o modelo pode melhorar precisão e sensibilidade simultaneamente. [Explicação oficial do limiar](https://developers.google.com/machine-learning/crash-course/classification/thresholding).


## Treinamento ampliado e estratégias executadas

Foram adquiridas **670 gravações novas** para treino/validação/calibração, por regra de metadados registrada antes da inferência. Incluem HumBug e ESC-50; o conjunto ampliado tem **49.713 janelas** em todas as partições, não 49.713 observações biológicas independentes. Entraram também 30 negativos HumBug usados nos testes das rodadas anteriores: agora são adaptação, nunca teste inédito desta rodada.

Comparação pré-definida: quatro redes neurais e três modelos clássicos. Todas as redes têm cabeça de presença de mosquito e cabeça condicional de espécie, combinadas por P(mosquito) × P(Aedes | mosquito), com calibração Platt posterior. Esse produto é uma regra do detector, não prova de calibração probabilística perfeita. Os modelos clássicos usam os mesmos dois estágios, treinados separadamente.

- CNN residual expandida: 2.792.899 parâmetros.
- PCEN residual: 9.583.874 parâmetros; PCEN normaliza energia com uma média móvel causal, combinado com log-Mel relativo e contraste espectral.
- PCEN híbrida: 9.653.122 parâmetros, com 138 features acústicas adicionais, normalizadas somente no treino.
- CNN 1D em áudio bruto: 348.163 parâmetros, adaptada da arquitetura MosquitoSong+.
- ExtraTrees com profundidade 16, ExtraTrees sem limite de profundidade e HistGradientBoosting: alternativas em features acústicas.

Os lotes têm 48 Aedes, 48 outros mosquitos e 32 ruídos, com equilíbrio por fonte, táxon e gravação. O treino inclui ruído real em SNR 5–30 dB, equalização aleatória para simular diferenças de microfone e superposição de sons da mesma classe binária. Essas misturas são **simulações de treinamento**, não gravações independentes de enxames. Nenhum áudio privado, UFRGS ou CD-MSC D3 de Aedes entrou no treino.

Pré-processamento: mono, reamostragem real, retirada de componente DC e normalização de RMS. Entradas reais de 0,3 s ou 0,992 s; não repetimos, preenchemos ou concatenamos áudio para fabricar uma observação. A CNN 1D usa três recortes reais disjuntos quando recebe uma janela longa. O modelo híbrido combina features do trecho original com a onda aumentada durante treino; as features não são recalculadas após cada aumento.

Treino, validação, calibração e teste foram separados por gravação e hashes; as 212 janelas de mosquito sem espécie continuam não rotuladas. Complexos cujo gênero é comprovadamente não Aedes podem ser negativos binários, mas não equivalem a identificação exata de espécie. A categoria ESC-50 “insects” foi excluída por ambiguidade de presença de mosquito.


| Candidato | Época selecionada | Menor recall de validação por fonte/classe | Precisão na validação | Limiar da calibração |
| --- | --- | --- | --- | --- |
| residual_expanded | 11 | 36.4% | 15.1% | 0.3488 |
| pcen_large | 13 | 42.7% | 15.8% | 0.5064 |
| pcen_hybrid_large | 1 | 33.1% | 13.1% | 0.4269 |
| raw_expanded | 10 | 33.3% | 6.8% | 0.5518 |
| trees_regularized | árvores | 14.0% | 7.4% | 0.3419 |
| trees_deep | árvores | 10.1% | 7.1% | 0.4075 |
| hist_gradient_boosting | árvores | 30.2% | 11.9% | 0.3880 |

Seleção: maior mínimo de recall por fonte/classe na validação, com pequenos termos de macro-F1 e average precision. Limiar: somente calibração, sem rótulos do teste. Congelamento: **2026-10-01T22:16:05.106583+00:00**. Seleção dos novos áudios: **2026-10-01T22:16:32.502892+00:00**. Todos os pesos e fontes congelados têm SHA-256; os 12 checkpoints anteriores permaneceram idênticos.

## Novo teste baixado depois do congelamento

Foram baixados **182 arquivos**, dos quais 166 foram avaliados após auditoria de duplicatas. A regra de seleção foi: First 24 unused acoustic class-1/domain-3 WAVs in full archive numeric order, plus first6/domain3 each of classes2,7,8,9 as SAME-domain negatives; first 3 unused HumBug recording groups per known non-Aedes taxon and noise; ESC50 fold5 first2 distinct original sources/category excluding flying insects.

Positivos: arquivos acústicos CD-MSC 2026, domínio D3, com rótulo Aedes aegypti. Negativos: novos grupos HumBug, fontes ESC-50 do fold 5 e **CD-MSC D3 com Aedes albopictus, Culex pipiens, Anopheles minimus e Anopheles stephensi**. Esse subconjunto inclui as duas classes no mesmo domínio acústico. O domínio D5 óptico foi excluído. [Dataset oficial CD-MSC 2026](https://zenodo.org/records/19095788), [HumBugDB](https://zenodo.org/records/4904800), [ESC-50 e agrupamento das fontes](https://github.com/karolpiczak/ESC-50).

**Limites:** D3 não fornece neste recorte identidade do mosquito ou sessão original; são arquivos novos, não indivíduos ou campanhas comprovadamente novos. HumBug só acrescenta negativos frescos, o que mantém confusão entre classe e fonte no agregado; reportamos também o D3 com as duas classes. O rótulo é do arquivo e não uma anotação de presença de batimento em cada instante. Isso impede extrapolar o teste a qualquer microfone, caixa ou enxame. O bootstrap por arquivo não corrige essa ausência de metadados.


| Fonte | Arquivos | Segundos |
| --- | --- | --- |
| CDMSC2026_D3 | 48 | 1008.4 |
| ESC50_noise | 97 | 485.0 |
| HumBug_noise | 3 | 83.8 |
| HumBug_other_mosquito | 34 | 455.4 |

A auditoria excluiu conservadoramente 16 arquivos ESC-50 que continham alguma janela PCM quantizada igual a uma janela anterior. O esclarecimento posterior mostrou que as 37 correspondências eram janelas constantes/silenciosas, não cópias demonstradas das gravações inteiras. A coluna `duplicate_file` sinaliza esse critério de colisão de janela. Mantivemos a exclusão pré-definida e não escolhemos substitutos; os resultados não foram recalculados para favorecer o modelo. Ver `novelty_clarification.json`.

## Comparação de todos os candidatos congelados

O vencedor foi escolhido antes do teste. Um candidato com melhor resultado neste teste não é automaticamente promovido: isso usaria o teste para seleção.

A linha `previous_round3_extra_trees_regularized` é o modelo previamente selecionado na rodada 3, preservado e avaliado nos mesmos áudios novos para comparação; não participou da seleção nesta rodada.

| Modelo | Selecionado antes do teste | Precisão | Recall Aedes | F1 Aedes | Macro-F1 | Especificidade | Acurácia |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | não | 37.9% | 14.8% | 21.2% | 45.0% | 82.8% | 55.1% |
| pcen_hybrid_large | não | 35.2% | 14.6% | 20.7% | 44.1% | 80.9% | 54.0% |
| pcen_large | sim | 10.3% | 2.7% | 4.2% | 35.5% | 83.4% | 50.6% |
| previous_round3_extra_trees_regularized | não | 45.1% | 57.9% | 50.7% | 53.7% | 51.1% | 53.9% |
| raw_expanded | não | 18.7% | 7.5% | 10.7% | 37.4% | 77.1% | 48.8% |
| residual_expanded | não | 48.0% | 35.8% | 41.0% | 54.1% | 72.9% | 57.8% |
| trees_deep | não | 55.7% | 46.3% | 50.6% | 60.4% | 74.1% | 62.8% |
| trees_regularized | não | 56.5% | 59.4% | 57.9% | 63.7% | 68.0% | 64.5% |

## Melhorou em relação ao modelo anterior?

Diferenças na mesma amostra, em pontos percentuais: nova rede escolhida na validação menos ExtraTrees escolhido na rodada 3. Intervalos pareados por arquivo/gravação, exploratórios; ganho parcial não implica aprovação de confiabilidade.

| Condição | Métrica | Mudança | IC 95% da mudança |
| --- | --- | --- | --- |
| all_fresh | aedes_precision | -34.8 pp | -47.6 a -20.2 pp |
| all_fresh | aedes_recall | -55.2 pp | -62.9 a -47.2 pp |
| all_fresh | aedes_f1 | -46.5 pp | -54.9 a -36.8 pp |
| all_fresh | specificity | +32.3 pp | +23.5 a +40.7 pp |
| all_fresh | macro_f1 | -18.3 pp | -25.9 a -9.9 pp |
| all_fresh | balanced_accuracy | -11.4 pp | -17.3 a -5.8 pp |
| mosquito_only | aedes_precision | -41.1 pp | -56.0 a -25.0 pp |
| mosquito_only | aedes_recall | -55.2 pp | -62.9 a -47.2 pp |
| mosquito_only | aedes_f1 | -50.8 pp | -59.6 a -40.7 pp |
| mosquito_only | specificity | +37.8 pp | +23.5 a +50.2 pp |
| mosquito_only | macro_f1 | -19.6 pp | -29.0 a -9.7 pp |
| mosquito_only | balanced_accuracy | -8.7 pp | -16.9 a -1.2 pp |
| CDMSC2026_D3 | aedes_precision | -25.4 pp | -64.0 a -0.9 pp |
| CDMSC2026_D3 | aedes_recall | -55.2 pp | -62.9 a -47.2 pp |
| CDMSC2026_D3 | aedes_f1 | -61.8 pp | -69.8 a -52.7 pp |
| CDMSC2026_D3 | specificity | +46.8 pp | +27.0 a +66.5 pp |
| CDMSC2026_D3 | macro_f1 | -29.2 pp | -37.1 a -19.3 pp |
| CDMSC2026_D3 | balanced_accuracy | -4.2 pp | -15.3 a +6.5 pp |

## Resultado do modelo selecionado por condição

| Condição | Janelas | Precisão | Recall Aedes | Especificidade | Falsos positivos | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| all_fresh | 1933 | 10.3% | 2.7% | 83.4% | 16.0% | 35.5% |
| mosquito_only | 1445 | 11.6% | 2.7% | 75.7% | 24.3% | 28.1% |
| CDMSC2026_D3 | 1002 | 53.8% | 2.7% | 91.7% | 8.3% | 19.3% |
| ESC50_noise | 405 | não estimável como detector binário | — | 92.6% | 5.7% | — |
| HumBug_noise | 83 | não estimável como detector binário | — | 100.0% | 0.0% | — |
| HumBug_other_mosquito | 443 | não estimável como detector binário | — | 67.9% | 32.1% | — |

## Acertos e erros por táxon

O detector emite somente Aedes/não Aedes; as linhas abaixo medem rejeição de cada táxon conhecido, não classificação multiclasse. Complexos mantêm sua identificação taxonômica incompleta.

| Referência | Grupos | Janelas | Aedes emitido | Não Aedes emitido | Inválidos | Taxa de acerto relevante |
| --- | --- | --- | --- | --- | --- | --- |
| Aedes aegypti | 24 | 786 | 21 | 765 | 0 | 2.7% |
| Aedes albopictus | 6 | 180 | 4 | 176 | 0 | 97.8% |
| Anopheles arabiensis | 3 | 6 | 0 | 6 | 0 | 100.0% |
| Anopheles barbirostris | 1 | 11 | 0 | 11 | 0 | 100.0% |
| Anopheles coustani | 3 | 11 | 8 | 3 | 0 | 27.3% |
| Anopheles dirus | 1 | 2 | 0 | 2 | 0 | 100.0% |
| Anopheles funestus sl | 3 | 22 | 21 | 1 | 0 | 4.5% |
| Anopheles funestus ss | 3 | 81 | 78 | 3 | 0 | 3.7% |
| Anopheles harrisoni | 3 | 28 | 0 | 28 | 0 | 100.0% |
| Anopheles maculatus | 3 | 18 | 0 | 18 | 0 | 100.0% |
| Anopheles maculipalpis | 3 | 110 | 4 | 106 | 0 | 96.4% |
| Anopheles minimus | 6 | 12 | 3 | 9 | 0 | 75.0% |
| Anopheles squamosus | 3 | 19 | 0 | 19 | 0 | 100.0% |
| Anopheles stephensi | 6 | 12 | 1 | 11 | 0 | 91.7% |
| Culex pipiens | 6 | 12 | 10 | 2 | 0 | 16.7% |
| Culex pipiens complex | 3 | 114 | 27 | 87 | 0 | 76.3% |
| Mansonia africanus | 2 | 7 | 2 | 5 | 0 | 71.4% |
| Mansonia uniformis | 3 | 14 | 2 | 12 | 0 | 85.7% |
| noise | 84 | 488 | 23 | 458 | 7 | 93.9% |

## Matriz de confusão do novo teste

Linhas são a referência; colunas são emissões. Inválidos não contam como rejeições corretas.

| Referência | Não Aedes | Aedes | Inválido |
| --- | --- | --- | --- |
| Não Aedes | 957 | 183 | 7 |
| Aedes | 765 | 21 | 0 |

## Intervalos de incerteza

Reamostragem estratificada por fonte com 2.000 repetições de grupos/arquivos inteiros. Mantém a correlação entre janelas; não comprova independência biológica dos arquivos CD-MSC.

| Condição | Métrica | IC 95% por gravação/arquivo |
| --- | --- | --- |
| all_fresh | aedes_precision | 1.6%–25.8% |
| all_fresh | aedes_recall | 0.4%–6.2% |
| all_fresh | aedes_f1 | 0.7%–9.6% |
| all_fresh | specificity | 74.7%–90.7% |
| all_fresh | macro_f1 | 30.7%–40.5% |
| all_fresh | balanced_accuracy | 38.4%–47.1% |
| mosquito_only | aedes_precision | 1.9%–29.7% |
| mosquito_only | aedes_recall | 0.4%–6.2% |
| mosquito_only | aedes_f1 | 0.7%–9.9% |
| mosquito_only | specificity | 60.3%–87.3% |
| mosquito_only | macro_f1 | 20.8%–34.9% |
| mosquito_only | balanced_accuracy | 31.4%–45.4% |
| CDMSC2026_D3 | aedes_precision | 14.3%–79.2% |
| CDMSC2026_D3 | aedes_recall | 0.4%–6.2% |
| CDMSC2026_D3 | aedes_f1 | 0.9%–11.4% |
| CDMSC2026_D3 | specificity | 81.1%–96.0% |
| CDMSC2026_D3 | macro_f1 | 9.4%–28.9% |
| CDMSC2026_D3 | balanced_accuracy | 41.9%–50.1% |

## Áudio privado do Drive

Referência fornecida pelo usuário: enxame de Aedes em caixa de vidro gravado pelo Nano 33 BLE Sense, com mosquito em todo o áudio. Todos os 107 trechos permanecem no denominador; dois são inválidos. É uma gravação conhecida e não um teste cego. Não foi usada no treino e não será publicada. Como só há positivos, o acerto mede sensibilidade nessa gravação; a precisão aparente de 100% não informa falsos alarmes.

| Modelo | Selecionado | Aedes detectados | Sensibilidade nesse áudio | Inválidos |
| --- | --- | --- | --- | --- |
| residual_expanded | não | 37/107 | 34.6% | 2 |
| pcen_large | sim | 39/107 | 36.4% | 2 |
| pcen_hybrid_large | não | 11/107 | 10.3% | 2 |
| raw_expanded | não | 17/107 | 15.9% | 2 |
| trees_regularized | não | 74/107 | 69.2% | 2 |
| trees_deep | não | 81/107 | 75.7% | 2 |
| hist_gradient_boosting | não | 53/107 | 49.5% | 2 |
| previous_round3_extra_trees_regularized | não | 77/107 | 72.0% | 2 |

## Bases históricas: diagnóstico de regressões

Já examinadas no desenvolvimento; não são um novo teste independente. UFRGS permanece ausente do treino. MosquitoSong challenge só tem Aedes: não fornece precisão/especificidade binárias.

| Fonte | Janelas | Precisão | Recall Aedes | Especificidade | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| dryad | 7901 | 47.9% | 91.7% | 95.2% | 80.1% |
| humbug | 1298 | 34.3% | 98.1% | 74.6% | 68.1% |
| mosquitosong | 352 | somente positivos | 31.8% | — | — |
| ufrgs | 417 | 99.0% | 36.5% | 99.3% | 58.0% |

## Reprodução dos estudos e rodadas anteriores

As comparações anteriores desta sessão ficam em `results/reproduction_2026_10_01/`. Foram executadas adaptações da CNN Mel de Fernandes, CNN 1D de MosquitoSong+, CNN residual e alinhamento de embeddings entre fontes da mesma espécie, além de árvores. São reproduções de estratégias, **não equivalência exata aos experimentos publicados**: tarefa, recortes, agrupamento e fontes de teste diferem.

- Fernandes: convoluções 32/64/64 e Dense 256, 1.137.601 parâmetros, Mel 60×40; BCE binária em vez dos dois sigmoides originais e interpolação do espectrograma para durações reais.
- MosquitoSong+: arquitetura 1D da figura 2, áudio de 0,3 s; adaptação de quatro classes para uma saída binária.
- Residual: 2.791.873 parâmetros; ruído e misturas de mesma classe no treino. Alinhamento contrastivo inspirado em trabalhos de adaptação de domínio; não reproduz integralmente o DR-BioL.

Rodada 1: rede residual com ruído escolhida na validação. Novo teste de 19 negativos: 334/544 falsos Aedes (61,4%); nos outros mosquitos, 67/103 (65,0%). Não houve positivo acústico novo nessa rodada. Áudio privado: 20/107 detectados.

Rodada 2: busca de 105 ensembles, selecionada só na validação; melhor mínimo de recall por fonte/classe ainda 38,8%. Áudio privado: 55/107. Os negativos da rodada 1 são diagnósticos reutilizados nesta etapa.

Rodada 3: 19 negativos anteriores incorporados ao treino. ExtraTrees regularizado foi escolhido antes do novo teste de 23 arquivos, contendo 12 positivos CD-MSC novos. Detectou 216/358 Aedes (60,3%), mas confundiu 33/47 outros mosquitos (70,2%). Precisão 64,9%, F1 Aedes 62,5%, acurácia balanceada operacional 64,8%. O resultado médio passou de 60% e **continuou inadequado entre espécies**. Outro candidato de árvores foi melhor nesse teste, mas não pode substituir retrospectivamente o vencedor sem uma nova validação. Áudio privado: 77/107.

Os 12 positivos foram escolhidos no prefixo de 8 MiB do diretório central ZIP64, não na lista completa do arquivo; a limitação está registrada no adendo da seleção. A rodada ampliada usa a lista completa de 542.777 entradas, com 192 arquivos D3 de Aedes. Os 12 anteriores estão excluídos do novo teste e de todo treino.

Os resultados publicados nos estudos não comprovam confiabilidade no nosso domínio: não fizemos uma comparação na mesma amostra, rótulos e protocolo. Por isso, não afirmamos superar Fernandes ou MosquitoSong+. A pesquisa bibliográfica está em [estudos_detector_binario_aedes_2026_10_01.md](estudos_detector_binario_aedes_2026_10_01.md).


## Explicabilidade e limites

Registramos os scores dos dois estágios, presença e espécie condicional, em cada janela pública, para localizar a etapa da falha. A tabela abaixo traz medianas por táxon; não são taxas de acerto nem probabilidades comprovadamente calibradas em campo. Na CNN 1D longa, cada estágio é uma média de três recortes e o score conjunto é a média dos produtos por recorte, não o produto das médias.

| Referência | Score presença, mediana | Score Aedes condicional, mediana | Score final calibrado, mediana |
| --- | --- | --- | --- |
| Aedes aegypti | 0.975 | 0.005 | 0.067 |
| Aedes albopictus | 0.985 | 0.006 | 0.077 |
| Anopheles arabiensis | 0.968 | 0.003 | 0.067 |
| Anopheles barbirostris | 0.995 | 0.001 | 0.025 |
| Anopheles coustani | 0.998 | 0.330 | 0.663 |
| Anopheles dirus | 0.674 | 0.003 | 0.041 |
| Anopheles funestus sl | 0.999 | 0.818 | 0.890 |
| Anopheles funestus ss | 0.999 | 0.913 | 0.932 |
| Anopheles harrisoni | 0.997 | 0.001 | 0.027 |
| Anopheles maculatus | 0.995 | 0.001 | 0.033 |
| Anopheles maculipalpis | 1.000 | 0.002 | 0.055 |
| Anopheles minimus | 0.777 | 0.036 | 0.222 |
| Anopheles squamosus | 1.000 | 0.001 | 0.023 |
| Anopheles stephensi | 0.704 | 0.006 | 0.065 |
| Culex pipiens | 0.999 | 0.691 | 0.838 |
| Culex pipiens complex | 0.999 | 0.008 | 0.122 |
| Mansonia africanus | 0.999 | 0.000 | 0.013 |
| Mansonia uniformis | 0.998 | 0.052 | 0.316 |
| noise | 0.015 | 0.229 | 0.061 |

As árvores também têm importância de features por estágio em `classical_feature_importance_exploratory.csv`. É importância por redução de impureza, sensível a features correlacionadas e ao número de valores possíveis; não equivale a relevância causal ou importância fora do treino. Features de frequência fundamental, harmônicos, PSD, dinâmica e MFCC têm interpretação acústica, mas correlacionam com sexo, temperatura, microfone e ambiente. Filtros/PCEN podem remover ruído; não recuperam informação de espécie ausente na gravação. Essa explicação operacional não prova que o modelo aprendeu uma característica biológica exclusiva de Aedes.

## Decisão e próximos dados necessários

Ainda não confiável: falhou nos mínimos offline definidos antes do teste.

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
- CDMSC2026_D3: aedes_precision below .60
- CDMSC2026_D3: aedes_precision lower95 below .60
- CDMSC2026_D3: aedes_recall below .60
- CDMSC2026_D3: aedes_recall lower95 below .60
- CDMSC2026_D3: macro_f1 below .60
- CDMSC2026_D3: macro_f1 lower95 below .60
- taxon:Aedes aegypti: aedes_recall below .60
- taxon:Anopheles coustani: specificity below .60
- taxon:Anopheles funestus sl: specificity below .60
- taxon:Anopheles funestus ss: specificity below .60
- taxon:Culex pipiens: specificity below .60

Para decidir sobre uso real, precisamos de novas sessões independentes com o sistema pretendido: Aedes e outros mosquitos confirmados, incluindo Aedes albopictus/Culex, ruídos sem mosquito, sexos e temperaturas variados, nas mesmas condições de caixa e microfone. Separar sessões inteiras antes de segmentar. Uma única gravação positiva do Arduino não mede falsos positivos nesse aparelho.

Precisão esperada em outras prevalências só pode ser apresentada como cenário. Usando sensibilidade e falsos positivos do subconjunto de mosquitos, sem supor que sejam constantes em campo:


| Prevalência hipotética de Aedes | Precisão calculada, não observada |
| --- | --- |
| 1.0% | 0.1% |
| 5.0% | 0.6% |
| 20.0% | 2.7% |
| 50.0% | 9.9% |

## Arquivos e reprodução

`results/reliability_2026_10_01/` guarda protocolo, aquisição, partições, resultados, hashes, intervalos e decisão. `confiabilidade_binaria_aedes_2026_10_01.ipynb` somente lê resultados e produz gráficos; não retreina ou compila. Pesos experimentais são arquivos locais ignorados pelo Git. Áudios públicos seguem as licenças das fontes, sem redistribuição automática; o privado permanece em `tmp/`.

Para reavaliar este experimento congelado:

```bash
.venv/bin/python evaluate_reliability.py
.venv/bin/python -m unittest tests.test_reliability tests.test_reproduction tests.test_corrected_validation
.venv/bin/python summarize_reliability.py
```

Para outra rodada, criar outro diretório, protocolo e novo teste: os scripts de treino/aquisição impedem retocar a versão congelada. Não usar o teste final para escolher um limiar ou o modelo vencedor.
