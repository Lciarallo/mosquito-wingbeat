# Revisão da atualização Git: classificação por espécie — 08/10/2026

**Os resultados são úteis para pesquisa dentro do corpus, mas ainda não sustentam identificação confiável de espécies em gravações novas.** Gênero tem resultados melhores que espécie. A nova calibração do ensemble apresenta uma dependência indevida da dobra externa; seu resultado precisa de validação aninhada antes de ser tratado como estimativa sem vazamento.

## Versão e trabalho conferido

Atualizado `main` por fast-forward de `971b534` para `1ab88b8`, incorporando quatro commits: emulador acústico/comparação de algoritmos, pipeline hierárquico, documentação e experimento TinyML + score Dryad. As alterações locais anteriores foram reaplicadas; o conflito em `.gitignore` foi resolvido preservando ambas as listas.

A revisão recalculou as métricas de seis abordagens a partir das probabilidades salvas e repetiu a inferência do pipeline atual no áudio já conhecido. Não houve novo treinamento, coleta independente ou compilação de firmware. As tabelas de origem continuam preservadas; [manifesto e hashes](../results/git_review_2026_10_08/manifest.json) identificam as entradas.

## Os números de espécie são bons?

Avaliação fechada das 20 espécies Dryad, sem rejeição, com 23.877 janelas em 569 grupos. Todos os arquivos já foram examinados em rodadas anteriores. A agregação por gravação usa a média de suas janelas; não representa uma decisão imediata no microfone.

| Abordagem | Acerto por janela | Acerto por gravação | Recall médio das 20 espécies por gravação |
|---|---:|---:|---:|
| SVM RBF | 46,13% | 65,03% | 76,66% |
| ExtraTrees com contraste | 47,74% | 61,16% | 72,84% |
| HistGradientBoosting com contraste | 52,69% | 73,46% | 79,44% |
| MLP com contraste | 40,74% | 59,75% | 72,77% |
| Ensemble fixo | 50,72% | 69,24% | 77,87% |
| Ensemble calibrado, com ressalva metodológica | 52,49% | 74,34% | 80,60% |

O HistGradientBoosting é um baseline promissor: supera a escolha aleatória e alcança 418/569 gravações corretas. Ainda erra 151 gravações e quase metade das janelas. A vantagem nominal do ensemble calibrado é pequena: 423/569 corretas, cinco a mais; não houve teste confirmatório de superioridade, e a dependência descrita abaixo impede aprová-lo pela diferença.

Gênero é uma tarefa mais fácil. Os 95,43% por gravação do ensemble calibrado se referem a quatro gêneros e carregam a mesma ressalva; não são acurácia de espécie. Os 85,15% com prior regional são acerto por janela em um subconjunto de três espécies escolhido pelo rótulo verdadeiro, com pesos regionais definidos manualmente. Não medem classificação de todas as espécies em campo, nem distinguem negativos/espécies desconhecidas. *Ae. aegypti* e *Ae. albopictus* não devem ser descritos como espécies endêmicas brasileiras nesse resultado.

## Resultados por espécie do baseline HistGradientBoosting

Precisão responde quantas emissões daquela espécie estavam corretas; recall responde quantas gravações daquela espécie foram reconhecidas. Ambos são necessários. Estas contagens são grupos de gravação, não indivíduos cuja independência foi confirmada.

| Espécie | Corretas / gravações | Recall | Precisão |
|---|---:|---:|---:|
| Aedes aegypti | 19/20 | 95,0% | 86,4% |
| Aedes albopictus | 5/7 | 71,4% | 45,5% |
| Aedes sierrensis | 37/60 | 61,7% | 100,0% |
| Anopheles arabiensis | 7/13 | 53,8% | 70,0% |
| Anopheles gambiae | 43/76 | 56,6% | 75,4% |
| Anopheles minimus | 31/50 | 62,0% | 60,8% |
| Culex quinquefasciatus | 11/12 | 91,7% | 64,7% |
| Culex tarsalis | 12/12 | 100,0% | 92,3% |
| Culiseta incidens | 30/30 | 100,0% | 90,9% |

*Ae. aegypti* tem um sinal interno interessante. *Ae. albopictus* recebe muitas previsões erradas vindas de outras espécies: 5 acertos em 11 emissões. Isso não permite afirmar que a distinção entre as duas está resolvida. Amostras pequenas e 100% de recall não garantem transferência. [CSV completo das 20 espécies e seis abordagens, por janela e gravação](../results/git_review_2026_10_08/per_species.csv); matrizes de confusão na mesma pasta.

## Problema da calibração e alcance do certificado

Em `optimize_ensemble_oof.py`, ao avaliar a dobra externa `k`, o otimizador ajusta pesos e priors com linhas OOF das outras dobras `j != k`. Essas linhas foram produzidas por modelos-base treinados em todas as dobras exceto `j`, portanto incluindo `k`. Há um caminho indireto dos dados/rótulos da dobra externa até a calibração. Deixar os rótulos de `k` fora de `y_train` do otimizador não elimina esse caminho.

Correção necessária: para cada dobra externa, refazer as predições de calibração em dobras internas restritas ao treino externo; então treinar os modelos-base somente no treino externo e avaliar `k` uma vez. Outra opção é congelar pesos fixos antes de qualquer teste confirmatório. Esta revisão não refez esses treinamentos e não estima quanto a dependência alterou as métricas.

`leakage_auditor.audit_splits` passou na verificação de grupos, caminhos e hashes declarados. Não inspeciona a linhagem das previsões do ensemble, não chama a verificação de scaler em `run_full_audit`, não verifica a escolha dos limiares e não comprova independência biológica ou entre bases. Os hashes são lidos do manifesto, não recalculados dos áudios nessa função. Assim, `ZERO_LEAKAGE_VERIFIED` é mais abrangente que o teste realizado.

As matrizes NPZ antigas não contêm IDs explícitos de cada linha. A ordem das classes foi conferida com o protocolo; a ordem das linhas é assumida conforme os scripts históricos. O nome de arquivo `train_hierarchical_models.py` também não representa um novo treinamento hierárquico: ele agrega probabilidades das 20 espécies já salvas e calcula as métricas.

## Reexecução no seu áudio com o código atual

Executado `predict_audio.process_audio_file` no `drive_audio.wav`, com ensemble das três MLPs Arduino históricas, hop de 0,5 s, prior global/Brasil e emulação desligada/ligada. São pesos diferentes do ensemble de sete modelos avaliado na tabela acima. A emulação é software com ruído de semente 42; não demonstra resposta do Arduino físico.

| Prior | Entrada | Janelas detectadas / 211 | Espécie candidata mais frequente | Espécie confirmada |
|---|---|---:|---|---|
| Global | Bruta | 50/211 | Culex tarsalis | Nenhuma |
| Global | Emulada | 62/211 | Culex tarsalis | Nenhuma |
| Brasil | Bruta | 50/211 | Aedes aegypti | Nenhuma |
| Brasil | Emulada | 62/211 | Aedes aegypti | Nenhuma |

O gênero com persistência também muda: Anopheles no prior global e Aedes no prior Brasil. “Confirmado” no código significa concordância temporal, não confirmação entomológica. O resultado mostra influência forte do prior e ausência de identificação aceita por espécie. A referência *Ae. aegypti* é a informação do usuário; esta revisão não encontrou anotação independente que confirme cada janela ou o intervalo 49,5–77,5 s. Esse áudio já orientou várias análises, portanto não é teste cego novo.

Os JSON históricos descrevem uma versão anterior da inferência e não coincidem com a reexecução atual. [Quatro reexecuções, CSVs e resumos](../results/git_review_2026_10_08/). Não substituímos seus resultados históricos.

## Experimento denominado mistura de datasets

A implementação treina o RandomForest apenas no treino TinyML, acrescentando como variável o score de uma rede Dryad pré-treinada. É uma combinação de features/transferência de score, não treinamento conjunto balanceado de dois corpora. Além disso, o baseline TinyML usa HistGradientBoosting e o híbrido usa RandomForest: a comparação não isola o efeito de combinar fontes. O alvo dos números divulgados é presença de qualquer mosquito contra ruído. `clf_tinyml_multi` é ajustado, mas não tem avaliação de espécie exportada.

O JSON produzido pelo experimento registra AUC híbrida **0,537 no HumBug e 0,879 no Drive**. O relatório/CSV citavam 0,668 no HumBug e o relatório atribuía 0,986 ao híbrido no Drive; esses números pertencem ao baseline TinyML. O campo chamado `macro_f1` no CSV recebe o F1 binário positivo do código. O resultado calibrado TPR 96,6% / FPR 9,0% não consta do JSON e não há etapa de calibração correspondente nesse script. Não foi validado nesta revisão.

Sem calibração, o híbrido marca 99,18% dos negativos HumBug e 82,05% dos negativos Drive como positivos, segundo o JSON. Isso não sustenta robustez operacional nem classificação por espécie. A anotação temporal Drive está fixa no próprio script, sem manifesto independente identificado. [Relatório corrigido do experimento](relatorio_experimento_mistura_datasets.md).

## Próximo passo recomendado e verificação

Usar HistGradientBoosting e ensemble fixo como referências de desenvolvimento; corrigir a validação aninhada antes de comparar o calibrado. Reservar novas sessões/indivíduos, microfones e ambientes, com espécie identificada por referência independente, incluindo outras espécies, ruído e ausência real. Fixar critérios, pesos e limiares antes da coleta confirmatória. Reportar precisão, recall, F1 e cobertura por espécie, incluindo rejeições; priors devem ser comparados separadamente e ter procedência registrada.

Verificação realizada: **123 testes existentes passaram**; as métricas dos seis métodos coincidiram com a tabela Git; os inputs históricos permaneceram com os mesmos hashes durante a auditoria. Testes de software não validam identificação em campo. Reprodução aritmética, preservando os resultados originais:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python audit_git_species.py --output tmp/revisao_especies
.venv/bin/python -m pytest -q tests
```

O script recusa repetir uma auditoria na pasta cujo manifesto já existe. Os dados e probabilidades locais são necessários. As quatro inferências foram executadas chamando `process_audio_file` com `HierarchicalMosquitoPipeline('ensemble', region)`, para `region` global/Brasil e `emulate_mic` falso/verdadeiro. Os hashes do código, áudio e modelos estão no manifesto.

Referência de escopo: o [baseline oficial BioDCASE 2026](https://arxiv.org/abs/2603.20118) trata classificação por espécie entre domínios de aquisição. Esse é um objetivo apropriado para a próxima validação; o desempenho publicado de outro modelo não valida os nossos pesos.
