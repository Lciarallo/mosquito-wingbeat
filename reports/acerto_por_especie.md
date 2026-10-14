# Taxas de acerto por espécie

Rodada 09/10/2026: novas bases com calibração separada e teste após congelamento em [ensemble corrigido](ensemble_especies_calibracao_separada_2026_10_09.md). O bom resultado interno em *Ae. aegypti* (19/20 gravações corretas) não se repetiu nos quatro arquivos externos (0/4). [CSV interno](../results/species_nested_2026_10_09/per_species.csv) · [CSV externo](../results/species_nested_2026_10_09/external_per_species.csv). As tabelas abaixo continuam referentes à CNN multifonte histórica.

Atualização 08/10/2026: esta tabela descreve a CNN multifonte histórica. O novo benchmark recebido do Git usa outros modelos e está na [revisão por espécie](revisao_git_classificacao_especies_2026_10_08.md), com [CSV por janela/gravação](../results/git_review_2026_10_08/per_species.csv). Nenhuma das rodadas demonstrou identificação confiável em novas condições de aquisição.

CNN multifonte atual, com pré-processamento original mantido. Contagens de janelas dos testes já examinados, não de mosquitos independentes. Acerto sem rejeição é recall da espécie real ao forçar a escolha de uma das 20 classes; não é a probabilidade de uma previsão individual estar correta. Identificação correta / todas as janelas inclui as rejeições no denominador. Precisão quando a espécie é emitida usa todas as emissões dessa espécie no corpus, incluindo erros vindos de outras espécies e rótulos desconhecidos. Não misturar taxas de corpora distintos.

## dryad

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 6 | 240/360 | 66.7% | 126/360 (35.0%) | 100.0% |
| Aedes albopictus | 3 | 99/180 | 55.0% | 45/180 (25.0%) | 93.8% |
| Aedes mediovittatus | 1 | 50/60 | 83.3% | 37/60 (61.7%) | 100.0% |
| Aedes sierrensis | 20 | 51/76 | 67.1% | 7/76 (9.2%) | 100.0% |
| Anopheles albimanus | 14 | 408/738 | 55.3% | 98/738 (13.3%) | 86.7% |
| Anopheles arabiensis | 4 | 89/240 | 37.1% | 50/240 (20.8%) | 73.5% |
| Anopheles atroparvus | 2 | 70/120 | 58.3% | 41/120 (34.2%) | 91.1% |
| Anopheles dirus | 19 | 463/776 | 59.7% | 168/776 (21.6%) | 94.9% |
| Anopheles farauti | 15 | 409/725 | 56.4% | 127/725 (17.5%) | 95.5% |
| Anopheles freeborni | 18 | 563/840 | 67.0% | 252/840 (30.0%) | 90.6% |
| Anopheles gambiae | 25 | 317/532 | 59.6% | 64/532 (12.0%) | 95.5% |
| Anopheles merus | 2 | 88/120 | 73.3% | 35/120 (29.2%) | 94.6% |
| Anopheles minimus | 17 | 344/901 | 38.2% | 35/901 (3.9%) | 97.2% |
| Anopheles quadriannulatus | 2 | 39/120 | 32.5% | 22/120 (18.3%) | 91.7% |
| Anopheles quadrimaculatus | 2 | 120/120 | 100.0% | 98/120 (81.7%) | 100.0% |
| Anopheles stephensi | 19 | 654/901 | 72.6% | 308/901 (34.2%) | 95.4% |
| Culex pipiens | 3 | 148/164 | 90.2% | 111/164 (67.7%) | 79.3% |
| Culex quinquefasciatus | 4 | 117/195 | 60.0% | 41/195 (21.0%) | 100.0% |
| Culex tarsalis | 4 | 196/217 | 90.3% | 108/217 (49.8%) | 100.0% |
| Culiseta incidens | 10 | 446/460 | 97.0% | 182/460 (39.6%) | 98.9% |

## humbug

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 7 | 143/154 | 92.9% | 99/154 (64.3%) | 88.4% |
| Anopheles arabiensis | 8 | 120/130 | 92.3% | 66/130 (50.8%) | 97.1% |
| Anopheles dirus | 8 | 83/89 | 93.3% | 42/89 (47.2%) | 72.4% |
| Anopheles minimus | 1 | 0/10 | 0.0% | 0/10 (0.0%) | Sem emissão |

## ufrgs

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 2 | 18/271 | 6.6% | 0/271 (0.0%) | Sem emissão |

## mosquitosong

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 2 | 142/352 | 40.3% | 2/352 (0.6%) | 100.0% |

MosquitoSong é desafio retrospectivo, com identidade individual entre idades não comprovada. UFRGS permaneceu inteira fora do treino, validação e calibração. No áudio privado informado como enxame de Aedes, nenhuma identificação foi emitida. Esse áudio não fornece referência das outras 19 espécies nem anotação de eventos por janela.

[CSV com os seis pré-processamentos, por espécie e corpus](../results/preprocessing/per_species.csv). Nenhum treinamento, ajuste de pesos/limiares ou compilação foi executado para produzir estas tabelas.
