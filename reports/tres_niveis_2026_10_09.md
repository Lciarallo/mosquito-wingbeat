# Presença, gênero e espécie — 09/10/2026

Recalculadas as decisões salvas dos três níveis, sem novo treinamento, mudança de limiares ou aquisição. O modelo de presença é o detector histórico separado. Gênero e espécie usam o ensemble `calibrated` da rodada de 09/10, com os mesmos 569 grupos internos e os mesmos 17 arquivos externos. **Não é uma comparação dos três estágios de um único sistema integrado.**

| Tarefa | Corpus conhecido | Avaliação externa disponível | Parecer |
|---|---|---|---|
| Presença de mosquito | Sensibilidade 72,63%; falsos positivos em ruído 29,17%; acurácia balanceada 71,73%, por janela, no modo padrão calibrado para FPR 10% | Não há nesta comparação teste externo com mosquito e ausência/ruído anotados | Detecta parte dos sinais, mas o excesso de alarmes impede afirmar confiabilidade |
| Gênero | 549/569 gravações corretas: 96,49%; acurácia balanceada 93,66% | 7/17 corretas: 41,18%; acurácia balanceada 29,17% | Forte no corpus conhecido, não transferiu bem |
| Espécie | 408/569 gravações corretas: 71,70%; acurácia balanceada 76,60% | 4/17 corretas: 23,53%; acurácia balanceada 20,00% | Mais difícil e também sem confiabilidade externa |

Os percentuais de gênero/espécie são de escolha forçada. Presença usa rótulos fracos por arquivo, sem identificação temporal de cada voo. Presença é medida por janela; gênero/espécie acima são por gravação. Não comparar diretamente seus percentuais como se fossem o mesmo evento, modelo ou conjunto.

## Presença: trocar o limiar não resolveu os dois tipos de erro

O modelo foi escolhido na validação interna histórica, sem selecionar um vencedor pelos números desta comparação. As três regras disponíveis nas previsões salvas são:

| Regra | Sensibilidade por janela | Falsos positivos por janela de ruído | Acurácia balanceada |
|---|---:|---:|---:|
| Limiar 0,5 | 82,68% | 35,23% | 73,73% |
| Calibração para FPR 5% | 48,88% | 20,27% | 64,31% |
| Calibração para FPR 10% | 72,63% | 29,17% | 71,73% |

Há 23.877 janelas rotuladas como positivas em 569 grupos e apenas 528 de ruído em 44 grupos. A prevalência artificialmente alta de positivos impede interpretar a precisão dessa amostra como precisão esperada no uso cotidiano. As metas nominais de 5% e 10% de FPR não foram alcançadas no teste. Falsos positivos por janela não são falsos alarmes por hora. A confirmação temporal foi estudada em um subconjunto com somente duas fontes de ruído, insuficiente para demonstrar robustez geral.

O ensemble de espécies atual não tem saída de presença validada em ruído. Uma escolha forçada de gênero/espécie sempre retorna uma classe de mosquito e não demonstra que havia mosquito. O teste externo de 17 arquivos só contém espécies de mosquito; não permite medir especificidade de presença.

## Gênero: o acerto interno alto não se manteve no externo

As probabilidades das 20 espécies foram somadas entre membros do mesmo gênero. Não houve treino dedicado ou limiar de rejeição calibrado para gênero. No teste externo, os acertos por gênero foram:

| Gênero real | Corretos / arquivos | Sensibilidade |
|---|---:|---:|
| Aedes | 0/5 | 0% |
| Anopheles | 7/8 | 87,5% |
| Culex | 0/4 | 0% |

Não há Culiseta externo. Sempre responder Anopheles acertaria 8/17 nesta amostra, mais que os 7/17 do modelo. Portanto, a acurácia geral de 41,18% não demonstra discriminação útil entre os gêneros. A estimativa balanceada de 29,17% fica abaixo do referencial de 33,33% para os três gêneros reais representados. São comparações descritivas, sem teste de significância ou garantia de independência biológica entre os arquivos.

## Espécie: a rejeição também não garantiu acerto

No corpus conhecido, o modelo emite 351 identificações em 569 gravações (61,69% de cobertura); 312 são corretas (88,89% entre emissões). No externo, emite em 8/17 arquivos (47,06% de cobertura) e acerta apenas 2/8 (25%). O acerto forçado externo é 4/17; duas dessas quatro respostas corretas foram rejeitadas. Assim, citar só o acerto interno entre emissões esconderia a falha de transferência.

O teste externo tem cinco espécies de três gêneros, não todas as 20 espécies de treino. Os mesmos arquivos já foram examinados na análise de gênero: esta comparação é retrospectiva, não uma nova confirmação independente.

## Conclusão e próximo teste

Os dados mostram que identificar o gênero é mais fácil que identificar a espécie **dentro do corpus conhecido**. Fora dele, ambos falham nesta amostra; o modelo não pode ser considerado bom para gênero em geral. Presença tem evidência própria de sensibilidade parcial e falsos positivos elevados, mas não foi avaliada externamente em condições comparáveis ao ensemble atual.

Para testar um sistema completo, é necessário fixar os três estágios e seus limiares antes de uma coleta no sensor/ambiente pretendidos, com períodos sem mosquitos, outras fontes sonoras e indivíduos de espécie confirmada. Avaliar cada etapa e também o resultado final por evento, incluindo abstenções, sinais inválidos e erros de presença. Nenhuma das três tarefas está aprovada para uso confiável em campo por esta comparação.

## Rastreabilidade

[Métricas recalculadas](../results/three_levels_2026_10_09/metrics.csv) · [Recall por gênero](../results/three_levels_2026_10_09/per_genus.csv) · [Hashes e escopo](../results/three_levels_2026_10_09/audit.json).

`compare_detection_levels.py` confere as decisões do detector de presença contra scores/limiares, recalcula as métricas, verifica coincidência de arquivos entre gênero/espécie e compara os valores com os resultados anteriores. Confere os hashes do ensemble congelado antes e depois da análise. A saída já existente não pode ser sobrescrita. Os modelos, pesos, limiares, firmware e análises anteriores foram preservados.
