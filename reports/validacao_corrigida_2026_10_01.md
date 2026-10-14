# Validação corrigida dos modelos de mosquito — 01/10/2026

**Correções implementadas e experimento executado. O desempenho corrigido não sustenta a conclusão anterior de alta confiabilidade.** Esta é uma avaliação dos modelos no computador. Não houve trabalho de firmware, compilação, adaptação ou compactação para Arduino.

## O que foi corrigido

- As cinco dobras do stacking agora usam `StratifiedGroupKFold`: todas as janelas de uma mesma gravação ficam na mesma dobra. São 17.391 janelas em 355 grupos, sem grupo compartilhado entre treino e validação OOF. Antes, 99,95% das janelas OOF compartilhavam o grupo com o treino da dobra.
- O meta-modelo recebe probabilidades brutas das árvores tanto no OOF quanto na inferência. A versão anterior era treinada com probabilidades brutas e testada com probabilidades recalibradas, em escalas diferentes.
- O ensemble substitui os ranks do lote por distribuições empíricas ponderadas e congeladas no treino. O limiar é escolhido exclusivamente na calibração principal. Gravações inéditas não alteram a normalização nem o limiar.
- O Super-Comitê recebe limiares derivados da calibração principal, em vez do par manual 0,35/0,65. Sua meta de até 5% de falsos positivos é ponderada por origem e espécie na calibração; não é garantia de 5% no teste.
- As 212 janelas HumBug de espécie desconhecida continuam sem rótulo binário; não viram negativos.
- Os 436 controles de ruído que repetiam o treino foram excluídos. Foram avaliados 971 trechos das partições de teste e, depois do congelamento, novos ruídos baixados da internet.
- No domínio óptico D5, a calibração usa 400 arquivos da lista oficial de validação; o teste usa outros 400 da lista oficial de teste. Os 400 arquivos examinados anteriormente foram excluídos de ambos. Não há IDs, conteúdo binário ou PCM idênticos entre calibração e teste. Isso comprova separação de clipes, não separação por indivíduo: a identidade de cada mosquito não é fornecida nesse recorte.

As quatro árvores foram retreinadas com os mesmos hiperparâmetros e partições principais; o stacking foi reconstruído com o OOF corrigido. A rede profunda e o SVM existentes foram preservados e reavaliados: o problema identificado no OOF não exigia retreinar esses modelos. Os 12 checkpoints anteriores permaneceram idênticos por SHA-256. Todos os resultados novos ficam em `results/corrected_2026_10_01/`; predições do áudio privado ficam em `tmp/`.

**Limitação do frontend preservado:** a rede profunda do outro modelo repete internamente entradas curtas de 0,3 s até 0,992 s, conforme sua implementação anterior. Mantivemos esse comportamento para avaliar os mesmos pesos, sem apresentá-lo como uma melhoria de pré-processamento. Isso não acrescenta informação acústica e pode alterar os padrões temporais; os resultados D5 e os diagnósticos de arquivos curtos devem ser lidos com essa ressalva. Uma arquitetura que aceite a duração real precisa de treinamento e novo teste após congelamento. Os novos áudios acústicos desta rodada e o áudio privado foram avaliados em janelas completas de 0,992 s, sem essa extensão interna.


Congelamento dos modelos e limiares: **2026-10-01T19:22:43.947457+00:00**. Seleção dos novos áudios: **2026-10-01T19:26:16.617590+00:00**, posterior ao congelamento. Não houve reajuste após examinar os testes.

## Comparação nas mesmas bases históricas

As bases abaixo já haviam sido examinadas, portanto são comparações históricas. Cada célula mostra **sensibilidade de emissão / falsos positivos emitidos**. Os limiares estão congelados; incertos e inválidos permanecem no denominador e não contam como acertos. As acurácias balanceadas forçadas, scores e coberturas estão nos CSVs.


| Modelo | Dryad: sensibilidade / FP | HumBug: sensibilidade / FP | UFRGS: sensibilidade / FP |
| --- | --- | --- | --- |
| LightGBM | 80.3% / 3.8% | 96.1% / 7.4% | 37.3% / 10.3% |
| HistGB | 82.5% / 3.7% | 96.1% / 6.4% | 41.7% / 10.3% |
| ExtraTrees novo | 71.9% / 3.7% | 90.9% / 3.3% | 48.3% / 11.0% |
| Random Forest | 55.8% / 2.4% | 70.8% / 0.4% | 12.9% / 6.8% |
| Soft Voting | 78.6% / 3.5% | 94.8% / 3.9% | 41.3% / 10.3% |
| Stacking corrigido | 37.8% / 1.6% | 35.7% / 0.2% | 4.8% / 5.5% |
| Rede profunda preservada | 88.1% / 3.4% | 97.4% / 7.4% | 65.3% / 19.2% |
| Super-Comitê recalibrado | 88.6% / 3.5% | 96.8% / 5.9% | 64.2% / 14.4% |
| Ensemble com CDF fixa | 98.6% / 13.7% | 99.4% / 18.3% | 87.1% / 69.9% |
| SVM anterior | 81.1% / 3.6% | 96.8% / 5.2% | 38.0% / 11.6% |
| ExtraTrees anterior | 75.0% / 5.0% | 88.3% / 3.4% | 81.2% / 29.5% |
| Boosting anterior | 79.4% / 4.7% | 94.2% / 4.3% | 76.0% / 29.5% |

### Stacking: antes e depois

Comparação de acurácia balanceada no corte 0,5, preservando os rótulos desconhecidos em ambas as versões:


| Base | Anterior | Corrigido |
| --- | --- | --- |
| dryad | 90.6% | 69.5% |
| humbug | 95.7% | 70.6% |
| ufrgs | 75.0% | 49.7% |

A queda do stacking acompanha duas correções: dobras por gravação e escala consistente das entradas do meta-modelo. Não atribuímos toda a diferença exclusivamente ao vazamento. As árvores isoladas e a rede mantêm suas características/hiperparâmetros; o Super-Comitê também mudou os limiares, então mudanças na emissão não significam que aprendeu novas características.

## Enxames sintéticos: comparação exploratória

O código público da base usa recortes do **Abuzz**, já utilizado no projeto, inseridos em posições diferentes do arquivo. O JSON indica presença da espécie no arquivo inteiro; não identifica os instantes em que ela está audível. Por isso, esta rodada mede presença por arquivo, não acurácia de espécie por janela, e não trata essas misturas como validação externa independente.


| Modelo | Arquivos | Acurácia balanceada | Sensibilidade emitida | Falsos positivos emitidos |
| --- | --- | --- | --- | --- |
| LightGBM | 315 | 54.7% | 18.0% | 14.9% |
| HistGB | 315 | 52.6% | 21.0% | 15.3% |
| ExtraTrees novo | 315 | 51.2% | 3.0% | 6.0% |
| Random Forest | 315 | 50.9% | 1.0% | 3.7% |
| SVM anterior | 315 | 63.2% | 5.0% | 2.8% |
| ExtraTrees anterior | 315 | 54.0% | 12.0% | 7.9% |
| Boosting anterior | 315 | 53.7% | 33.0% | 25.6% |
| Soft Voting | 315 | 51.6% | 14.0% | 11.6% |
| Stacking corrigido | 315 | 49.4% | 0.0% | 1.9% |
| Rede profunda preservada | 315 | 56.0% | 18.0% | 6.0% |
| Super-Comitê recalibrado | 315 | 57.3% | 25.0% | 14.9% |
| Ensemble com CDF fixa | 315 | 49.9% | 98.0% | 98.1% |

O ensemble corrigido ficou em **49.9%**, frente aos 65,55% anteriores obtidos com ranks do próprio lote e limiar otimizado no próprio teste. Seu limiar corrigido é 0.421865, escolhido na calibração acústica principal. Ele ainda não generaliza adequadamente a essas misturas. Aplicar uma média de scores por arquivo é uma regra fixa deste diagnóstico; não houve calibração nas misturas.

A estabilidade entre lotes foi verificada em 1575 janelas: diferença máxima 1.11e-16 ao processar cada janela sozinha e 0.00e+00 ao processar por arquivo. O defeito de dependência do lote foi resolvido, sem que isso garanta qualidade de classificação.

## D5: novo teste óptico, com calibração separada

200 Aedes e 200 não-Aedes na calibração oficial; outros 200 Aedes e 200 não-Aedes no teste oficial. Seleção por ordenação de hashes com semente pré-fixada, sem consultar as predições. Os limiares específicos do D5 são uma adaptação de domínio feita na validação; não são limiares escolhidos no teste.


| Modelo | AUC | Acurácia balanceada | Sensibilidade | Falsos positivos |
| --- | --- | --- | --- | --- |
| LightGBM | 0.594 | 57.5% | 81.0% | 66.0% |
| HistGB | 0.596 | 57.8% | 79.5% | 64.0% |
| ExtraTrees novo | 0.616 | 59.5% | 83.0% | 64.0% |
| Random Forest | 0.608 | 61.3% | 68.5% | 46.0% |
| SVM anterior | 0.548 | 58.0% | 83.0% | 67.0% |
| ExtraTrees anterior | 0.625 | 61.3% | 71.5% | 49.0% |
| Boosting anterior | 0.608 | 59.8% | 64.5% | 45.0% |
| Soft Voting | 0.610 | 58.8% | 80.0% | 62.5% |
| Stacking corrigido | 0.599 | 58.8% | 67.5% | 50.0% |
| Rede profunda preservada | 0.543 | 54.5% | 75.0% | 66.0% |
| Super-Comitê recalibrado | 0.594 | 58.5% | 83.0% | 66.0% |
| Ensemble com CDF fixa | 0.562 | 58.0% | 72.0% | 56.0% |

O intervalo observado de acurácia balanceada foi **54,5% a 61,25%**, inferior aos 73,5% destacados anteriormente. Não é a mesma amostra antiga, portanto essa diferença também inclui mudança de amostra/distribuição. Os resultados com os limiares acústicos originais também estão em `d5_test_metrics.csv` e apresentam muitos falsos positivos. Esta avaliação óptica não mede diretamente desempenho em áudio de microfone.

## Novos áudios baixados após o congelamento

Foram selecionados, sem consultar resultados, todos os 12 arquivos de ruído ambiental/ventiladores disponíveis no repositório MosquitoSong e os três primeiros grupos HumBug ainda não utilizados segundo a regra registrada. Os três são Anopheles arabiensis: IDs 206579, 206590 e 206597. Sua identidade de gravação não aparece nos grupos anteriores e seus arquivos não são cópias binárias dos áudios já baixados. São exemplos novos da mesma campanha HumBug, não um novo domínio de aquisição.

Os ruídos totalizam **161 janelas** e os três mosquitos **seis janelas**. São amostras pequenas e correlacionadas dentro de cada arquivo. As gravações de ventiladores por microfones diferentes também não devem ser tratadas como eventos independentes.

Não restaram grupos acústicos de Aedes aegypti inéditos no HumBug que cumprissem a duração mínima. Isso foi registrado antes da inferência, sem reutilizar positivos antigos. Logo, esta rodada externa acústica estima falsos alarmes, **não sensibilidade em Aedes inédito**. O novo teste D5 tem positivos, mas é óptico.


| Modelo | Falsos Aedes no ruído novo | Falsos Aedes em An. arabiensis novo | Incerto em An. arabiensis |
| --- | --- | --- | --- |
| LightGBM | 10/161 (6.2%) | 2/6 (33.3%) | 2 |
| HistGB | 24/161 (14.9%) | 2/6 (33.3%) | 1 |
| ExtraTrees novo | 1/161 (0.6%) | 3/6 (50.0%) | 2 |
| Random Forest | 1/161 (0.6%) | 1/6 (16.7%) | 5 |
| Soft Voting | 6/161 (3.7%) | 3/6 (50.0%) | 2 |
| Stacking corrigido | 2/161 (1.2%) | 2/6 (33.3%) | 4 |
| Rede profunda preservada | 6/161 (3.7%) | 1/6 (16.7%) | 0 |
| Super-Comitê recalibrado | 7/161 (4.3%) | 1/6 (16.7%) | 1 |
| Ensemble com CDF fixa | 23/161 (14.3%) | 5/6 (83.3%) | 0 |
| SVM anterior | 14/161 (8.7%) | 0/6 (0.0%) | 4 |
| ExtraTrees anterior | 5/161 (3.1%) | 5/6 (83.3%) | 0 |
| Boosting anterior | 30/161 (18.6%) | 2/6 (33.3%) | 3 |

O SVM não emitiu Aedes nas seis janelas novas de Anopheles, mas rejeitou quatro como incertas; isso não corresponde a seis classificações corretas confirmadas. O ensemble com CDF fixa emitiu Aedes em cinco das seis, apesar de detectar muitos trechos do áudio privado: a sensibilidade maior vem acompanhada de alarmes falsos.

## Áudio privado: referência do usuário em todas as 107 janelas

Todo o arquivo foi informado como enxame de Aedes aegypti em caixa de vidro gravado pelo Arduino. Essa é a referência fornecida pelo usuário, sem verificação entomológica independente. Existem duas janelas inválidas, mantidas no total. A gravação já era conhecida antes desta rodada e não é um teste cego nem uma amostra de 107 gravações diferentes.


| Modelo | Aedes detectado /107 | Não-Aedes | Incerto | Inválido |
| --- | --- | --- | --- | --- |
| LightGBM | 38/107 (35.5%) | 35 | 32 | 2 |
| HistGB | 36/107 (33.6%) | 45 | 24 | 2 |
| ExtraTrees novo | 48/107 (44.9%) | 30 | 27 | 2 |
| Random Forest | 32/107 (29.9%) | 18 | 55 | 2 |
| SVM anterior | 25/107 (23.4%) | 41 | 39 | 2 |
| ExtraTrees anterior | 66/107 (61.7%) | 26 | 13 | 2 |
| Boosting anterior | 55/107 (51.4%) | 23 | 27 | 2 |
| Soft Voting | 42/107 (39.3%) | 33 | 30 | 2 |
| Stacking corrigido | 18/107 (16.8%) | 5 | 82 | 2 |
| Rede profunda preservada | 59/107 (55.1%) | 32 | 14 | 2 |
| Super-Comitê recalibrado | 59/107 (55.1%) | 41 | 5 | 2 |
| Ensemble com CDF fixa | 70/107 (65.4%) | 35 | 0 | 2 |

O ensemble corrigido detecta 70/107, mas sua taxa de falsos positivos na UFRGS chega a 69,9% e nos seis trechos inéditos de Anopheles a 83,3%. A detecção no áudio privado, isoladamente, não demonstra reconhecimento confiável da espécie. A rede preservada continua com 59/107; o stacking corrigido ficou com 18/107 e 82 incertos.

## Ruído de teste e incerteza

Nos 971 trechos de ruído efetivamente reservados para teste, o Super-Comitê recalibrado emitiu 57 falsos Aedes (5,9%), a rede 70 (7,2%), o stacking corrigido quatro (0,4%) e o ensemble com CDF fixa 195 (20,1%). Uma taxa baixa de alarmes acompanhada de muitas rejeições/positivos perdidos não basta para declarar o detector confiável.

`recording_cluster_intervals.csv` inclui intervalos exploratórios por reamostragem de gravações, em vez de tratar janelas como observações independentes. Quando há menos de dois grupos de origem para uma classe, o intervalo é recusado. Identidades biológicas/sessões incompletas continuam sendo uma limitação. O áudio privado e os seis novos trechos de Anopheles não permitem estimar confiabilidade geral com precisão.

## Estado final e reprodução

**Estado: experimental; generalização insuficiente.** As falhas de avaliação identificadas foram corrigidas. Os resultados agora mostram limitações importantes nos modelos, particularmente no ensemble e no stacking. Não houve seleção de um vencedor por estes testes, reajuste de limiares após avaliação nem atualização de firmware.

Arquivos de entrada, modelos, seleção, hashes, horários e auditorias estão em `results/corrected_2026_10_01/`. Os modelos anteriores permanecem preservados. Há sete testes de regressão das falhas corrigidas e 20 testes prévios, totalizando 27 testes aprovados. O notebook `validacao_corrigida_2026_10_01.ipynb` apenas lê os resultados e não treina nem compila.

Em uma reprodução nova, execute primeiro `models_trees.py`, depois `models_committee.py` e `corrected_validation.py --phase fit`. Depois execute `corrected_validation.py --phase evaluate`, `fetch_corrected_fresh.py`, `evaluate_corrected_fresh.py` e `summarize_corrected_validation.py`. Use `.venv/bin/python` na raiz do projeto. O experimento já congelado bloqueia novos ajustes: outra rodada de treinamento precisa de um novo diretório/protocolo. Os avaliadores anteriores agora encaminham a execução ao protocolo corrigido, preservando os resultados antigos.

## Referências de metodologia e origem

- [StratifiedGroupKFold: preservação dos grupos nas dobras](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [Gerador original das misturas a partir de Abuzz](https://github.com/duydinhthai27/synthetic-swarm-mosquito/blob/main/mosquito_swarm.ipynb).
- [MosquitoSong: origem dos novos controles ambientais e de ventiladores](https://github.com/akaraspt/mosquitosongp).
- [HumBugDB: origem dos três novos arquivos de Anopheles](https://zenodo.org/records/4904800).
- [BioDCASE: arquivos e listas oficiais de validação/teste](https://zenodo.org/records/20478577).
