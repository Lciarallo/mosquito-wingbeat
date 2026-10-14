# Auditoria dos novos resultados recebidos do Google Drive — 14/10/2026

O novo script apresenta melhora no subconjunto de 28 gravações, mas **o resultado de 100% informado no JSON do Drive não foi reproduzido**. A reexecução encontrou 26/28 acertos de espécie e de gênero. Nos mesmos 17 arquivos da comparação histórica, obteve 6/17 espécies e 10/17 gêneros. A inferência continua experimental: utiliza estatísticas de outras janelas da gravação, falha em uma verificação nova da v5 e não dispõe de comprovação independente para as 20 espécies.

Esta é uma auditoria e reexecução dos arquivos recebidos. Não é um novo treinamento aninhado v5. Os resultados originais, inclusive o JSON que informa 100%, foram preservados.

## Procedência e escopo

Pasta autenticada: [mosquito-wingbeat](https://drive.google.com/drive/folders/10oUy0jGS8npQ0GbZo4UO4UMwGntTdS14). Recebemos quatro scripts da v5, dois scripts de inferência em `scripts/` e `tests/test_nested_species_v5.py`, modificados no Drive entre 15:53:13 e 15:53:59 UTC. Todos os sete downloads tiveram MD5 conferido contra os metadados do Drive e SHA-256 registrado em `source_provenance.json`. Nenhum parâmetro desses scripts foi modificado nesta auditoria.

O arquivo [multiclass_28files_v5_audit.json](https://drive.google.com/file/d/1iDyE_CA5xzvHVtgjirNBxmosJcP_rsyx/view?usp=drivesdk), modificado às 16:00:59 UTC, informa 28/28 espécies, 28/28 gêneros, 139 candidatos a Ae. aegypti no Drive em cada modo e Dryad OOF v5 de 63,62%. Sua cópia integral está em [received_multiclass_28files_v5_audit.json](../results/drive_review_2026_10_14_161041/received_multiclass_28files_v5_audit.json).

Não havia pasta `results/species_nested_v5_2026_10_14/`, modelo treinado v5, `freeze.json` v5, previsões OOF v5 ou relatório de treinamento v5 nas listagens do Drive realizadas nesta auditoria. Portanto, **63,62% no Dryad e a alegação de 25 gravações recuperadas não foram verificados**. Não devem substituir as métricas congeladas da v4.

O script `scripts/predict_audio.py` continua carregando os pesos neurais históricos `results/arduino_species/models/species_fold0.npz` e `species_fold2.npz`. Ele não carrega os ensembles SVM/ExtraTrees/HGB da v4 ou da v5. Os 12 arquivos do diretório de modelos neurais foram comparados por MD5 com o Drive e coincidiram; o SHA-256 do áudio privado também coincide com os metadados do arquivo já existente no Drive. Isso reduz divergências conhecidas de código, pesos e entrada, mas o JSON recebido não identifica sua própria linhagem por hashes.

## Protocolo de reexecução

Foi registrado [review_protocol.json](../results/drive_review_2026_10_14_161041/review_protocol.json), com timestamp UTC e hashes de entradas, pesos, fontes atuais, fontes anteriores e script de auditoria, antes das previsões desta revisão. Esse registro congela uma reexecução diagnóstica; não transforma dados conhecidos em teste cego nem substitui o congelamento de um treinamento v5.

Reexecutamos sem ajuste de parâmetros:

1. Os 28 arquivos exatos enumerados no JSON recebido, em modo global bruto.
2. Os 17 arquivos exatos enumerados no plano externo congelado da v4, em modo global bruto.
3. O áudio privado em Brasil emulado, Brasil bruto e global bruto, pelos mesmos aliases CLI solicitados anteriormente.
4. As duas comparações públicas também com os scripts locais anteriores, preservados em `output/drive_review_2026_10_14_161041/local_before/scripts/`. Cada versão usa seu próprio módulo harmônico e os mesmos pesos históricos; a localização dos pesos permanece a raiz do projeto.

Os 28 arquivos incluem 27 WAVs públicos e o áudio privado, todos já disponíveis antes desta revisão, de somente cinco espécies. Há 9 referências a Ae. aegypti, 5 a Ae. albopictus, 5 a Cx. pipiens, 5 a An. minimus e 4 a An. stephensi. O rótulo do áudio privado vem do usuário, sem confirmação entomológica independente. Os 17 pareados são outro subconjunto dessas mesmas cinco espécies. Não há negativos de presença, anotações temporais independentes ou identificação de indivíduos/sessões que permitam tratar cada arquivo como mosquito biologicamente independente.

Por gravação, `primary_species`/`primary_genus` correspondem ao candidato mais frequente nas janelas detectadas. `confirmed_species`/`confirmed_genus` correspondem ao resultado de persistência temporal mais frequente. **Candidato e emissão são avaliados separadamente**, inclusive quando divergem. Ausência de candidato conta como erro no denominador completo; `NO_CANDIDATE` também entra na união de categorias do F1 estrito, como categoria de erro e não como táxon biológico. A cobertura abaixo é a proporção de gravações com saída de persistência; o acerto aceito usa a saída realmente emitida. Persistência de gênero não equivale a um limiar de confiança de gênero calibrado.

## Métricas reproduzidas por gravação

Os nomes “anterior” e “atual” abaixo designam as duas versões do **script neural/heurístico de inferência**, não as rodadas de treinamento v4 e v5.

| Conjunto / nível | Script | Acertos / total | Acurácia | Acurácia balanceada | Macro-F1 suporte | Macro-F1 união estrita | Cobertura | Acerto das emissões |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 28 recebidos / espécie | Anterior | 12/28 | 42,86% | 41,11% | 39,24% | 28,03% | 92,86% | 46,15% |
| 28 recebidos / espécie | Atual | 26/28 | 92,86% | 95,56% | 95,68% | 79,73% | 89,29% | 92,00% |
| 28 recebidos / gênero | Anterior | 20/28 | 71,43% | 56,88% | 52,94% | 52,94% | 96,43% | 70,37% |
| 28 recebidos / gênero | Atual | 26/28 | 92,86% | 92,59% | 93,61% | 93,61% | 96,43% | 92,59% |
| 17 pareados / espécie | Anterior | 6/17 | 35,29% | 30,00% | 36,33% | 22,71% | 82,35% | 42,86% |
| 17 pareados / espécie | Atual | 6/17 | 35,29% | 30,00% | 38,22% | 27,30% | 70,59% | 50,00% |
| 17 pareados / gênero | Anterior | 7/17 | 41,18% | 41,67% | 31,85% | 23,89% | 82,35% | 50,00% |
| 17 pareados / gênero | Atual | 10/17 | 58,82% | 62,50% | 62,63% | 46,97% | 76,47% | 69,23% |

Sem o áudio privado, o script atual acerta 25/27 espécies e 25/27 gêneros públicos do novo subconjunto, ambos 92,59%. Em espécie, BA=95,00%, F1 estrito=79,44%, cobertura=24/27 e acerto emitido=22/24. Assim, a melhora não depende apenas de incluir o Drive; ainda assim, é retrospectiva e restrita a cinco espécies.

Os erros de espécie entre os 28 são `S_1_D_3_75.wav` → Ae. albopictus e `S_1_D_3_85.wav` → Ae. sierrensis. O gênero Aedes é correto nesses dois casos. As duas falhas de gênero ocorrem em outras gravações; espécie e gênero podem divergir por causa da regra de agregação taxonômica. O F1 estrito de espécie também penaliza Ae. sierrensis, fora do suporte verdadeiro desses 28 arquivos.

No subconjunto de 17, os acertos de espécie por táxon são Ae. aegypti 2/4, Ae. albopictus 0/1, Cx. pipiens 2/4, An. minimus 1/4 e An. stephensi 1/4. Portanto, a discriminação entre espécies continua limitada, inclusive dentro de Aedes.

## Intervalos e comparação pareada

Usamos 2.000 réplicas bootstrap pareadas por arquivo, estratificadas pelos rótulos observados, semente 20261014, com a união do F1 recalculada em cada réplica. Os intervalos são descritivos por arquivo; não corrigem seleção prévia, adaptação pelos arquivos conhecidos ou dependência entre indivíduos/sessões desconhecidos. A unidade passada ao algoritmo é o arquivo, não uma nova auditoria PCM de aquisição.

| Resultado | Estimativa atual | IC descritivo 95% | Diferença atual − anterior | IC pareado da diferença |
|---|---:|---:|---:|---:|
| Espécie nos 28 | 92,86% | 82,14–100,00% | +50,00 pp | +35,71 a +64,29 pp |
| Espécie nos 17 | 35,29% | 17,65–58,82% | 0,00 pp | −17,65 a +17,65 pp |
| Gênero nos 17 | 58,82% | 41,18–76,47% | +17,65 pp | −5,88 a +41,18 pp |

O ganho de gênero nos 17 não tem diferença pareada de acurácia separada de zero nesse procedimento. Aumentar o número de janelas sobrepostas não aumenta o número de gravações independentes.

## Áudio privado e cronologia do enxame

| Modo atual | Janelas | Voo detectado | Flag harmônica de enxame | Candidatos Ae. aegypti | Janelas de persistência Ae. aegypti |
|---|---:|---:|---:|---:|---:|
| Brasil emulado | 211 | 139 | 125 | 132 | 118 |
| Brasil bruto | 211 | 139 | 125 | 132 | 117 |
| Global bruto | 211 | 139 | 125 | 112 | 92 |

A frequência mediana das janelas detectadas foi 600,0 Hz. O JSON recebido informa 588,8 Hz e 139 candidatos Ae. aegypti em todos os modos; esses valores não foram reproduzidos. Outras gravações também apresentam divergências de F0, registradas individualmente em `replay_verification.json`.

No modo global, 112/139 é 80,58% dos candidatos detectados; no denominador completo são 112/211=53,08%. No Brasil, 132/139=94,96% dos detectados e 132/211=62,56% do total. Essas frações **não são acurácias entomológicas por janela**: não temos rótulo temporal independente de cada voo, ausência ou espécie. O filtro regional e os bônus heurísticos mudam as saídas.

O novo resumo contém duas cronologias. `swarm_timeline` usa preenchimento offline com uma janela futura; `causal_swarm_timeline` usa persistência dos rótulos passados. Nos três modos, os oito intervalos ativos da segunda são: 25,5–26,0 s; 35,5–37,5 s; 39,0–39,5 s; 41,0–47,5 s; 48,0–91,5 s; 92,0–95,0 s; 95,5–99,5 s; 100,0–105,992 s. Eles somam 65,992 s; a cronologia offline soma 63,492 s. Os últimos 0,248 s não formam uma janela completa e ficam sem análise. Os JSONs preservam todos os intervalos ativos e inativos, com F0 e SNR.

`swarm_present=false` é uma decisão do algoritmo, não prova de ausência de mosquito. Um padrão harmônico compatível com voo não confirma gênero ou espécie. “CONFIRMADO” no terminal significa repetição temporal da classificação, não confirmação biológica.

## Dependência entre janelas e regras não calibradas

Em `scripts/predict_audio.py`, `evaluate_windows` calcula `_recording_context` com as médias de `features[:,31]`, `features[:,46]`, `features[:,32]` e `features[:,58]`. Essas médias alteram os bônus de Ae. albopictus e Ae. aegypti na classificação de uma janela. Portanto, o script atual **não é estritamente indutivo por janela**, apesar de o frontend neural continuar recebendo PCM sem pré-condicionamento harmônico.

O primeiro diagnóstico com a janela de 50 s não mudou a saída: sua ramificação não utilizava o contexto. Um segundo diagnóstico encontrou um contraexemplo na janela real de 25,5 s: mantendo o PCM e o vetor de features dessa janela exatamente iguais e mudando somente quatro features de outra linha, a previsão mudou de Ae. aegypti para Ae. albopictus; a maior diferença de probabilidade foi 0,578855. A intervenção foi controlada sobre a matriz de features, não uma nova gravação física. O contraexemplo e os dois vetores estão em [batch_inductivity_counterexample.json](../results/drive_review_2026_10_14_161041/batch_inductivity_counterexample.json). Isso basta para refutar invariância bit a bit a outras linhas do lote.

A persistência de `causal_swarm_timeline` usa rótulos passados, mas alguns rótulos podem depender do gênero calculado com contexto da gravação inteira. Assim, esse recurso não comprova causalidade do pipeline completo em streaming.

O código recebido ampliou as regras em relação ao pedido anterior: permite bônus Aedes também com entropia ≥0,55 sob condições de `b0`, introduz bônus de 3,80 para determinados casos de Cx. pipiens/Ae. albopictus, e pode impor peso bioacústico mínimo 0,50 ou 0,65 fora da fórmula adaptativa original. Os limiares dessas regras não vêm de artefatos de calibração recebidos. Não há evidência suficiente para afirmar em que dados foram escolhidos; são tratados como heurísticas exploratórias. Nenhuma foi retocada aqui.

O ganho nos 28 é compatível com melhora naquele subconjunto conhecido, mas também com especialização em suas condições. A queda para 6/17 espécies no outro subconjunto torna indefensável extrapolar 92,86% para novos aparelhos, ambientes, sessões ou as 20 espécies.

## Testes, preservação e decisão

Comando executado: `PYTHONPATH=. .venv/bin/python -m pytest -q tests`.

**Resultado real: 215 passed, 1 failed, 49.273 warnings, 37,39 s, código de saída 1.** Os 202 testes anteriores passaram; dentre os 14 novos casos v5, 13 passaram e um falhou. Não houve ocultação da falha, alteração de teste, reescrita de snapshot ou retreinamento.

A falha é `tests/test_nested_species_v5.py::test_cepstrum_is_scale_normalized_and_retains_shape`: o teste exige RMS dos coeficientes cepstrais igual a 2,5, mas observa 0,289500. `centred_cepstrum` divide por `max(RMS,1.5)`, produzindo norma menor que 2,5 quando RMS<1,5. A invariância ao ganho passa; a expectativa de norma fixa conflita com esse piso. O código e o teste recebidos foram preservados para revisão, sem decidir silenciosamente qual especificação deve mudar. A afirmação de `passed=true` gerada por `ensure_pretraining_artifacts` também não substitui executar e inspecionar esses testes.

`verify_freeze` e `verify_history` foram executados antes e depois; v1, v2, v3 e v4 passaram. Os snapshots históricos e manifestos de conclusão permaneceram intactos. Das 1.633 referências SHA-256 da linha de base, apenas os dois scripts ativos em `scripts/` mudaram, conforme a sincronização solicitada; os 1.631 demais arquivos coincidiram. As versões anteriores dos dois scripts foram preservadas e usadas na comparação pareada. Nenhum firmware, instalador, peso histórico, documento histórico ou resultado concluído foi sobrescrito. Nenhum áudio bruto foi incluído no envio desta revisão; o arquivo privado já existente no Drive não foi alterado.

**Decisão: manter experimental, sem liberação para Arduino e sem confirmação automática de espécie.** Antes de usar estas melhorias como nova evidência de confiabilidade, será necessário resolver a dependência entre janelas, reconciliar a especificação cepstral com o teste, treinar/calibrar a rodada v5 real com grupos disjuntos, congelar todos os artefatos e avaliar outra aquisição independente. Os 28 e os 17 já examinados permanecem material exploratório.

Artefatos desta revisão: [resultados](../results/drive_review_2026_10_14_161041/metrics.json), [previsões por gravação](../results/drive_review_2026_10_14_161041/recording_predictions.csv), [intervalos bootstrap](../results/drive_review_2026_10_14_161041/bootstrap_intervals.json), [terminal e JSONs completos](../results/drive_review_2026_10_14_161041/terminal_e_jsons_completos.md), [preservação SHA-256](../results/drive_review_2026_10_14_161041/historical_verification.json). O script de reexecução é [review_drive_results_2026_10_14.py](../scripts/review_drive_results_2026_10_14.py).

Entrega no Drive: o [pacote completo](../results/drive_review_2026_10_14_161041/drive_review_complete.zip) reúne os 240 arquivos selecionados, com a hierarquia relativa do repositório e um manifesto SHA-256 interno. Inclui todos os CSVs/JSONs individuais de `per_file/`, fontes recebidas e anteriores, metadados, protocolos e logs de execução. O relatório, os três resumos do Drive e os principais resultados também são disponibilizados como arquivos individuais. A cópia inicial dos arquivos individuais foi interrompida voluntariamente para reduzir operações lentas da API, sem apagar arquivos; o ZIP contém o conjunto completo. Áudios brutos e pesos históricos não integram o pacote, pois permanecem nas localizações originais autorizadas. Extrair o pacote em uma pasta separada permite inspecionar a entrega sem substituir arquivos do repositório.
