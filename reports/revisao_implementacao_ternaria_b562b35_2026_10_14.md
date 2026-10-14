# Revisão da implementação ternária — b562b35

Data: 14/10/2026

Commit revisado: `b562b35f9569032acf568caa003afce0102a0640`

Referência anterior: `20d6fe9`

## Parecer

**A implementação corrigiu partes importantes do protocolo, mas ainda não atingiu as metas de desempenho e precisa de proteções adicionais para preservar o isolamento em novas execuções.**

A revisão anterior descrevia a versão `20d6fe9`. Este documento avalia o código e os artefatos novos de `b562b35`; os dois estados não devem ser confundidos.

No caminho aninhado atual, o código calcula pesos e scalers dentro do treino, seleciona `gamma` e limiares dentro do treino externo e mantém o bloco PCM conhecido em uma única dobra. Não identifiquei, na inspeção desse caminho com os arquivos presentes, a dependência anterior em que a calibração reutilizava previsões produzidas com participação da dobra externa.

As falhas de ausência de arquivos descritas abaixo são condicionais: os artefatos exigidos estavam presentes nesta revisão. Elas não demonstram que a execução publicada usou o caminho alternativo.

## 1. Resultados atuais recalculados

Unidade de avaliação: `group`, com 1.504 grupos, incluindo 64 positivos de Aedes, 1.014 de outros mosquitos e 426 negativos de mosquito. Grupos relacionados por colisões PCM continuam sendo múltiplas unidades na métrica, embora compartilhem o bloco de isolamento. Essas unidades não devem ser chamadas automaticamente de eventos biológicos independentes.

| Modelo | Acurácia balanceada | Aedes corretos | Sensibilidade Aedes | Precisão Aedes | Ruído → Aedes | Outro mosquito → Aedes |
|---|---:|---:|---:|---:|---:|---:|
| Boosting | 84,30% | 46/64 | 71,88% | 71,88% | 0/426 | 18/1014 |
| Ensemble | 87,92% | 54/64 | 84,38% | 54,55% | 2/426 | 43/1014 |
| Cascata aninhada | 82,11% | 46/64 | 71,88% | 67,65% | 0/426 | 22/1014 |

Os cinco modelos por grupo e os três modelos por janela tiveram suas matrizes recalculadas das previsões salvas. Todas coincidiram com os JSONs publicados.

### Metas da cascata

| Critério | Meta proposta | Resultado atual | Situação |
|---|---:|---:|---|
| Aedes corretos | Pelo menos 50/64 | 46/64 | Não atingida |
| Ruído identificado como Aedes | No máximo 4/426 | 0/426 | Atingida nesta amostra |
| Outro mosquito identificado como Aedes | No máximo 20/1014 | 22/1014 | Não atingida |

A matriz atual da cascata, com linhas verdadeiras e colunas preditas na ordem ruído/outro/Aedes, é:

```text
[[329,  97,  0],
 [  6, 986, 22],
 [  0,  18, 46]]
```

Os 18 Aedes perdidos terminaram em `other_mosquito`. Os 97 ruídos classificados como outro mosquito correspondem a **22,77% dos grupos negativos**. Portanto, zero ruídos classificados como Aedes continua não significando zero falsos alarmes de presença de mosquito.

### Aedes por fonte na cascata

| Fonte | Corretos / total | Sensibilidade |
|---|---:|---:|
| Dryad | 19/20 | 95,00% |
| HumBug | 26/40 | 65,00% |
| MosquitoSong | 1/4 | 25,00% |

A sensibilidade agregada não descreve adequadamente a dificuldade por fonte. O resultado MosquitoSong é especialmente incerto pelo tamanho de quatro grupos e pela concentração de muitos WAVs em cada grupo.

Na versão anterior, a cascata registrava 48/64. A versão atual registra 46/64. Essa diferença é descritiva: houve mudança de particionamento, seleção de `gamma` e configuração de treinamento, portanto não isola o efeito causal de uma correção específica.

## 2. Correções confirmadas

- **Bloco PCM conhecido isolado:** os 54 grupos e suas 1.734 janelas agora ficam juntos na dobra externa 0. Cada `pcm_block` declarado aparece em apenas uma dobra externa.
- **Pesos calculados no treino:** `compute_sample_weights` é chamado com o subconjunto de ajuste em cada divisão interna e novamente no treino externo.
- **Scaler calculado no treino:** cada divisão ajusta seu próprio `StandardScaler`, aplicado posteriormente ao subconjunto de validação correspondente.
- **Seleção interna:** a escolha entre `gamma=0.50` e `0.75` usa as previsões internas; os limiares da cascata também são selecionados dentro do treino externo. Os cinco folds selecionaram `gamma=0.50`.
- **Artefatos separados:** os resultados novos estão em `results/ternary_nested_2026_10_14/`; os resultados anteriores continuam em `results/ternary_detection_2026_10_14/`.
- **Relato de contagens:** o relatório novo inclui numeradores e denominadores inteiros, além das taxas.

O mapa PCM reaproveitado contém 613 dos 1.504 grupos atuais. Os outros 891 recebem o próprio identificador como bloco. Isso corrige a colisão conhecida, mas não equivale a uma auditoria PCM completa de todas as fontes do corpus ternário.

## 3. Achados prioritários

### P1 — Ausência do mapa PCM não interrompe o treinamento estrito

Local: [train_ternary_nested_models.py](../train_ternary_nested_models.py), linhas 84–89.

Se `COLLISION_MAP_CSV` não existir, o programa usa `group` como `pcm_block` e continua. Nessa condição, a colisão conhecida deixa de ser unida, mas o relatório ainda escreve `pcm_block_leakage: 0`.

Reprodução de leitura: substituí o caminho do mapa por um caminho inexistente apenas na memória de um processo Python. `load_and_filter_data()` continuou e retornou **1.504 blocos**, em vez dos **1.451** presentes na rodada auditada. Nenhum arquivo do projeto foi alterado nessa reprodução.

**Direcionamento:** tornar o mapa obrigatório no modo estrito; validar esquema, identidade e hash da versão usada; registrar cobertura e limites da auditoria. A afirmação de disjunção deve se referir aos blocos efetivamente auditados, sem assumir ausência de colisões desconhecidas nas demais fontes.

### P1 — Ausência dos limiares internos ativa calibração legada

Local: [evaluate_ternary_event_pipeline.py](../evaluate_ternary_event_pipeline.py), linhas 429–454.

Se `inner_thresholds.json` não existir, a avaliação chama `calibrate_neyman_pearson_oof(rec_df)`, inclusive quando foi escolhido o diretório aninhado. Essa função seleciona limiares usando as previsões externas globais dos outros folds e reintroduz a dependência metodológica corrigida pelo treinamento novo.

**Direcionamento:** exigir limiares internos completos e compatíveis com as previsões no modo aninhado. Validar os cinco folds, o identificador da rodada e hashes das entradas. Se faltar qualquer requisito, interromper a avaliação; o modo legado deve ser uma opção explícita e identificada como retrospectiva.

### P2 — A persistência temporal é calculada, mas não participa da decisão

Local: [evaluate_ternary_event_pipeline.py](../evaluate_ternary_event_pipeline.py), linhas 392–410 e 434–450.

`rolling_max_p2` é calculado e guardado em `rec_df`, mas não é usado pela cascata, pelos modelos de referência ou por um classificador de eventos. Também não é exportado como diagnóstico por grupo. A decisão continua dependendo das médias ponderadas por atividade, do peso de fusão e dos dois limiares.

**Direcionamento:** descrever a persistência como um descritor ainda não integrado. Antes de atribuir ganho de detecção a ela, definir a regra que a consome e avaliá-la por uma comparação interna controlada. Qualquer regra nova precisa ser aplicada de forma idêntica na calibração e na avaliação.

### P2 — Casos sem continuidade recebem valor de persistência positivo

Local: [evaluate_ternary_event_pipeline.py](../evaluate_ternary_event_pipeline.py), linhas 104–136.

O isolamento por WAV foi implementado, mas a semântica temporal ainda é incompleta:

- Duas janelas do mesmo arquivo em 0 s e 100 s, com scores 0,9 e 0,1, retornam persistência 0,9 por substituição pelo pico isolado, embora não exista par contíguo.
- Sem a coluna `start_s`, os mesmos scores retornam 0,9 ou 0,1 conforme a ordem das linhas. Esse descritor não é invariável à ordem do CSV quando não há tempo disponível.
- A condição `diferença entre inícios <= 1.05` não verifica duração ou cobertura temporal; também aceita inícios iguais. O CSV atual não tem duplicatas de `path/start_s`, mas o caso permanece permitido pela função.

Há **20.363 janelas sem `duration_s`** no CSV atual, embora `start_s` esteja preenchido em todas. Uma regra uniforme de duas janelas ou 1,05 s não demonstra persistência durante o mesmo intervalo nas diferentes fontes e durações de janela.

**Direcionamento:** definir persistência em segundos, reconstruir a duração real das janelas pelo extrator de origem e verificar cobertura/continuidade. Separar `peak_score` de persistência. Quando o tempo ou um par válido não estiver disponível, registrar o estado como ausente ou indisponível em vez de convertê-lo silenciosamente em evidência temporal positiva.

### P2 — O relato por fonte omite negativos e representa ausência de dados como certeza

Local: [evaluate_ternary_event_pipeline.py](../evaluate_ternary_event_pipeline.py), função `wilson_ci` e bloco `source_breakdown`.

O detalhamento inclui somente Dryad, HumBug e MosquitoSong. Os **289 grupos de ruído ESC50**, que completam os 426 negativos, são omitidos do detalhamento por fonte. As métricas globais continuam incluindo esses grupos.

Além disso, `wilson_ci(0, 0)` retorna ponto 0 e intervalo `[0, 0]`. Isso aparece, por exemplo, na ausência de negativos de ruído no MosquitoSong. Sem observações, a taxa e seu intervalo não estão estimados; não há evidência de erro exatamente zero.

**Direcionamento:** incluir todas as fontes e validar que seus denominadores somem aos totais globais. Para denominador zero, usar `null` ou indicação explícita de não estimável. O parâmetro `confidence` também deve ser respeitado ou removido, pois a função usa sempre a constante de 95%.

Os intervalos de Wilson atuais são binomiais por grupo. Como vários grupos pertencem a um mesmo bloco PCM, eles não oferecem, por si só, cobertura estatística demonstrada para esta amostra dependente. Relatar essa limitação e usar reamostragem por unidades de isolamento adequadas quando aplicável.

### P2 — Faltam artefatos para auditar e reutilizar a rodada

Local: [train_ternary_nested_models.py](../train_ternary_nested_models.py), linhas 210–246, 232–235, 389–392 e 453–480.

O diretório novo contém cinco arquivos: métricas, matrizes, limiares internos, métricas por grupo e previsões externas por janela. Não contém modelos/scalers serializados, previsões internas, partições internas identificadas, scores de ambos os valores de `gamma`, manifesto de entradas/código ou registro de congelamento.

Também existe diferença de configuração entre o boosting interno (`max_iter=150`) e externo (`max_iter=200`). Sem o número efetivo de iterações salvo, não é possível confirmar se os dois limites produziram o mesmo regime de parada. Essa diferença precisa ser eliminada ou justificada antes de tratar a calibração como correspondente à mesma configuração do modelo avaliado.

**Direcionamento:** salvar partições, previsões e seleção internas, configurações completas, iterações efetivas, modelos, scalers e hashes. Usar a mesma configuração na produção das previsões de calibração e na aplicação externa. Congelar a rodada antes de gerar as previsões de avaliação e impedir sobrescrita de uma rodada concluída.

Essas lacunas não tornam as matrizes salvas aritmeticamente erradas. Elas limitam a auditoria independente da execução e impedem reutilizar diretamente esta rodada como um modelo pronto para inferência em novos WAVs.

## 4. Testes e limites da revisão

Comando executado:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

Resultado: **249 passed, 59.933 warnings, 39,41 s**. Os avisos são depreciações e leitura de tipos mistos já emitidas pela suíte.

Também foram executadas verificações de leitura que:

- Recalcularam todas as matrizes novas, por janela e grupo, e confirmaram correspondência com os artefatos.
- Confirmaram a disjunção externa dos blocos PCM declarados e a localização do bloco conhecido na dobra 0.
- Mediram a cobertura do mapa de colisões.
- Reproduziram a continuação sem mapa PCM e os casos temporais descritos acima.
- Confirmaram a omissão dos 289 grupos ESC50 no detalhamento por fonte e a resposta para denominador zero.

O teste `test_sample_weights_strictly_within_train` verifica a massa de classe em dados sintéticos. Ele não verifica a linhagem das chamadas reais do treinamento. A nova suíte também não cobre todos os comportamentos de arquivos ausentes, tempo indisponível e persistência não consumida descritos nesta revisão.

Não houve novo treinamento, alteração dos scripts, alteração dos testes existentes ou regravação dos resultados científicos nesta revisão. As alterações desta tarefa são documentais. A leitura do código confirma a estrutura aninhada pretendida; a ausência de artefatos internos persistidos impede reconstruir integralmente a linhagem da execução publicada sem repetir o treinamento.

## 5. Próximo direcionamento

1. **Proteger o modo aninhado:** interromper quando mapa PCM, limiares internos ou metadados de compatibilidade estiverem ausentes. Cobrir os cenários com testes que exercitem o caminho real.
2. **Completar o registro da rodada:** salvar partições, previsões internas, seleção de `gamma`, modelos/scalers e hashes; alinhar as configurações interna e externa.
3. **Corrigir o relato:** incluir ESC50, tratar denominador zero, explicitar dependência entre grupos e manter a distinção entre ruído → Aedes e ruído → qualquer mosquito.
4. **Definir a persistência real:** usar continuidade no mesmo WAV e duração conhecida; integrar a regra somente dentro da seleção interna e medir sua contribuição separadamente.
5. **Comparar os modelos no mesmo ponto operacional:** ajustar boosting e ensemble sob as mesmas restrições internas de falsos positivos, avaliando cada um na mesma divisão externa.
6. **Confirmar em nova aquisição:** após congelar a solução, avaliar Aedes, outros mosquitos e ruído de sessões/indivíduos independentes. As fontes já examinadas permanecem desenvolvimento retrospectivo.

O ensemble encontra mais Aedes nesta rodada, mas 45 dos 99 alertas de Aedes estão errados. O boosting apresenta maior precisão e atende aos tetos empíricos de falsos positivos nesta amostra, porém fica abaixo da meta de sensibilidade. A cascata atual não melhora esse compromisso em relação ao boosting simples nos números agregados. Essas observações orientam comparações futuras; não autorizam escolher retrospectivamente limiares e anunciar o mesmo conjunto como confirmação independente.

## Referências

- [Revisão original do plano em 20d6fe9](direcionamento_refinamento_ternario_2026_10_14.md).
- [Métricas atuais por grupo](../results/ternary_nested_2026_10_14/event_level_metrics.json).
- [Previsões externas atuais](../results/ternary_nested_2026_10_14/predictions_oof.csv).
- [Métricas e protocolo registrados](../results/ternary_nested_2026_10_14/metrics.json).
- [Limiares internos registrados](../results/ternary_nested_2026_10_14/inner_thresholds.json).
- [Mapa PCM reaproveitado](../results/species_nested_2026_10_13/recording_predictions.csv).
- [Validação aninhada — documentação scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [Tong, Feng e Li — controle Neyman–Pearson](https://arxiv.org/abs/1608.03109).

Os limites empíricos de calibração não garantem limites populacionais no teste. Garantias probabilísticas exigem um procedimento e pressupostos específicos; o nome Neyman–Pearson no código não substitui essa demonstração.
