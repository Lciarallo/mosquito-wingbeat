# Correções e avaliação da detecção ternária

Data: 14/10/2026. Base revisada: `b562b35f9569032acf568caa003afce0102a0640`.

**A implementação foi concluída e a avaliação principal compreende 1.494 grupos em validação cruzada aninhada.** A cascata detectou **47/60 grupos de Aedes (78,33%)**, com **0/426 ruídos (0%)** e **23/1008 outros mosquitos (2,28%)** classificados como Aedes. A precisão foi de **67,14%**. Esses resultados mostram um equilíbrio promissor entre sensibilidade e falsos positivos no corpus avaliado.

**Foco da análise:** por orientação posterior do usuário, o parecer principal se concentra nessa população OOF. As verificações externas com poucos grupos e arquivos são mantidas nos apêndices como observações complementares; não determinam isoladamente o julgamento desta versão. Essa mudança de ênfase não altera previsões, limiares, denominadores ou os critérios numéricos originalmente definidos. O desempenho em novas condições permanece uma questão distinta do resultado OOF medido.

Esta entrega sucede a [revisão do plano](direcionamento_refinamento_ternario_2026_10_14.md) e a [revisão de b562b35](revisao_implementacao_ternaria_b562b35_2026_10_14.md). Os números desses documentos anteriores descrevem seus respectivos estados históricos.

## 1. Análise de lacunas e correções executadas

| Lacuna identificada | Comportamento corrigido |
|---|---|
| Mapa PCM ausente permitia continuar usando apenas `group`. | O mapa conhecido é obrigatório e validado; todos os arquivos retidos passam também por auditoria de bytes e PCM. Falha de decodificação no corpus retido interrompe o treino. |
| Ausência dos limiares internos acionava calibração pelas previsões externas. | Limiares, dobras, identificador da rodada e hashes são obrigatórios. O caminho alternativo foi desativado. |
| Linhagem interna e modelos não eram integralmente persistidos. | Partições, previsões internas, candidatos, configurações, scalers e modelos de cada ajuste são salvos antes da avaliação externa. |
| Persistência aceitava picos sem continuidade e não participava da decisão. | Usa timestamps e duração real dentro de um WAV; entra explicitamente na regra, com peso selecionado internamente e opção de peso zero. |
| Janelas de durações distintas eram tratadas como se representassem o mesmo tempo. | A duração vem do cache de amostras: 0,992 s ou 0,300 s. O horizonte temporal comum é 1,984 s, com tolerância declarada de 0,010 s para lacunas. |
| Configurações diferentes do HistGB nas etapas interna e externa. | Todos os ajustes usam 200 iterações, mesma regularização e `early_stopping=False`. |
| Relatório por fonte omitia ESC50; denominadores vazios viravam resultados certos. | Todas as fontes são reconciliadas com o total. Métricas indefinidas e intervalos sem suporte são `null`; o nível de confiança solicitado é respeitado. |
| Uma fração arredondada reprovava a própria contagem permitida. | Critérios usam aritmética inteira e denominadores explícitos, incluindo a adaptação da população após a exclusão autorizada. |
| Resultados novos podiam substituir rodadas anteriores. | Treinamento exige diretório vazio; reexecução de uma avaliação já iniciada é recusada. Leitura histórica exige `--legacy` e não reescreve artefatos. |
| Inferência não estava vinculada ao modelo final congelado. | Há um comando offline que verifica hashes e usa o mesmo frontend, sem repetir, preencher ou concatenar áudio curto. Janelas inválidas são registradas. |

Os pesos e o scaler são ajustados exclusivamente nas linhas de treino de cada divisão. A massa de pesos de cada classe é `N_fit/3`. A escolha entre `gamma=0.50` e `0.75`, os pesos da fusão, a persistência e os limiares usam somente previsões internas ao treino externo.

A margem de segurança é uma **restrição empírica de calibração**: FPR interno de até 0,75% em ruído e 1,75% em outros mosquitos. Ela não garante esses limites na dobra externa nem na população. O código registra violações e uma regra que não emite Aedes quando nenhum candidato detecta positivos dentro das restrições; não relaxa os tetos para fabricar aprovação.

O plano de acrescentar vários descritores foi substituído por uma regra menor e verificável: pooling de atividade, fusão de probabilidades e persistência condicional com peso explícito. Não se atribui ganho a top-k, fração ativa ou concordância que não tenham sido incorporados à decisão.

## 2. População e protocolo desta rodada

Por instrução expressa do usuário, **MosquitoSong foi retirado de todo o desenvolvimento**, incluindo treino, seleção, normalização e calibração. Sua avaliação separada usa o modelo final congelado, sem novo ajuste.

| População de desenvolvimento | Quantidade |
|---|---:|
| Janelas retidas | 42.029 |
| Grupos avaliados | 1.494 |
| Grupos de Aedes | 60 |
| Grupos de outros mosquitos | 1.008 |
| Grupos de ruído | 426 |
| Arquivos de origem auditados | 1.749 |
| Blocos PCM de isolamento | 1.441 |
| Grupos com persistência temporal disponível | 1.390 |

Foram excluídas 2.739 janelas do MosquitoSong. Outros filtros anteriores, como fontes UFRGS/Figshare, rótulos desconhecidos e qualidade técnica, permanecem registrados em [dataset_statistics.json](../results/ternary_verified_2026_10_14_corrections/dataset_statistics.json). As contagens individuais de exclusão podem se sobrepor.

Foram usadas cinco dobras externas e quatro internas, separadas por blocos PCM. A cobertura das previsões OOF foi de 1.494/1.494 grupos retidos. As janelas excluídas não entram nessa medida de cobertura.

`group` é a unidade de decisão, e o bloco PCM é a unidade de isolamento e reamostragem. Um grupo pode reunir vários WAVs; não equivale automaticamente a um voo ou indivíduo independente. A auditoria cobre bytes e PCM não constante integral ou alinhado em fragmentos de 1,000/0,992/0,300 s, com e sem normalização de ganho. Não estabelece independência biológica nem ausência de edições ou deslocamentos arbitrários.

O frontend determinístico e as características existentes foram reutilizados. Esta execução retreinou os classificadores e seus scalers; não representa treinamento de uma nova rede profunda.

## 3. Congelamento e conferência da linhagem

Diretório da execução: [ternary_verified_2026_10_14_corrections](../results/ternary_verified_2026_10_14_corrections/).

- Congelamento: **14/10/2026 às 23:34:19 UTC** — 20:34:19 em São Paulo.
- Avaliação OOF concluída: **23:34:40 UTC**, após o congelamento.
- SHA-256 de `freeze.json`: `34b7b1f1201f3722adcd4a451b4c4282d306e2a3beafb7df88c625cea74f04bd`.
- 163 artefatos constam do manifesto congelado, além dos hashes do código e das entradas.
- Foram salvos e conferidos **54 pares LR/HistGB**, com seus scalers: 48 ajustes internos, cinco modelos externos e um modelo final de desenvolvimento.

A conferência verificou os conjuntos efetivos de linhas de cada modelo, a disjunção de linhas/grupos/caminhos/blocos PCM e a massa de pesos. As médias e variâncias dos 54 scalers foram recalculadas a partir das características brutas das respectivas linhas de ajuste, com diferença máxima zero. Os modelos internos, externos e final executaram efetivamente 200 iterações do HistGB.

Além de conferir hashes, as probabilidades OOF foram recalculadas com os cinco modelos congelados; a diferença máxima para o CSV salvo foi de `1,11e-16`. As sete decisões por grupo e matrizes de confusão coincidiram com os artefatos salvos. A evidência está em [lineage_and_history_audit.json](../results/ternary_verified_2026_10_14_corrections/lineage_and_history_audit.json).

| Contexto | Gamma escolhido internamente | Peso temporal da cascata |
|---|---:|---:|
| Dobra externa 0 | 0,75 | 0,50 |
| Dobra externa 1 | 0,50 | 0,00 |
| Dobra externa 2 | 0,50 | 0,00 |
| Dobra externa 3 | 0,75 | 0,50 |
| Dobra externa 4 | 0,75 | 0,25 |
| Modelo final de desenvolvimento | 0,75 | 0,00 |

O modelo final descartou a persistência por seleção interna, antes dos testes de transferência. Os resultados externos não foram usados para escolher uma versão ou recalibrar limiares.

## 4. Resultados OOF no mesmo protocolo

| Modelo | Acurácia balanceada | Aedes corretos | Sensibilidade | Precisão Aedes | Ruído → Aedes | Outro → Aedes |
|---|---:|---:|---:|---:|---:|---:|
| Regressão logística | 84,88% | 55/60 | 91,67% | 28,95% | 5/426 | 130/1008 |
| HistGB simples | 84,25% | 44/60 | 73,33% | 72,13% | 0/426 | 17/1008 |
| Ensemble simples LR + HistGB | 88,98% | 52/60 | 86,67% | 57,14% | 1/426 | 38/1008 |
| **Cascata com seleção temporal** | **83,67%** | **47/60** | **78,33%** | **67,14%** | **0/426** | **23/1008** |
| Cascata sem persistência | 84,86% | 49/60 | 81,67% | 68,06% | 0/426 | 23/1008 |
| HistGB com restrições | 85,14% | 50/60 | 83,33% | 68,49% | 0/426 | 23/1008 |
| Ensemble com restrições | 82,37% | 39/60 | 65,00% | 66,10% | 0/426 | 20/1008 |

O ensemble simples aumenta a sensibilidade, mas produz 38 falsos positivos entre outros mosquitos, com FPR de 3,77%. Ele não atende ao objetivo operacional proposto. Nenhum candidato da tabela foi promovido com base nesses resultados externos à seleção.

A matriz da cascata principal, com linhas verdadeiras e colunas preditas na ordem ruído/outro/Aedes, é:

```text
[[324, 102,  0],
 [ 11, 974, 23],
 [  0,  13, 47]]
```

Os 13 Aedes perdidos terminaram em `other_mosquito`. Houve **102/426 ruídos classificados como algum mosquito (23,94%)**, apesar de nenhum ter sido chamado de Aedes. Essas são métricas diferentes. O F1 de Aedes foi de **72,31%**, o F1 macro de **83,45%**, e a especificidade entre outros mosquitos de **97,72%**.

### Metas com os denominadores corretos

Os critérios preservam as taxas correspondentes às contagens originais de 50/64, 4/426 e 20/1014. Com a nova população, isso exige:

| Critério da cascata | Contagem exigida | Observado | Situação |
|---|---:|---:|---|
| Aedes detectados | Pelo menos 47/60 | 47/60 | Atingida nesta amostra |
| Ruído → Aedes | No máximo 4/426 | 0/426 | Atingida nesta amostra |
| Outro mosquito → Aedes | No máximo 19/1008 | 23/1008 | **Não atingida** |

Manter 20 erros depois de reduzir o denominador para 1008 relaxaria a taxa original. A diferença entre 19 e 20 não pode ser escondida por arredondamento. [operational_targets.json](../results/ternary_verified_2026_10_14_corrections/operational_targets.json) registra `passed_all: false` para os sete candidatos, independentemente da aprovação dos testes de software.

### Fontes e incerteza

| Fonte de desenvolvimento | Grupos | Aedes corretos | Outro → Aedes | Ruído → Aedes |
|---|---:|---:|---:|---:|
| Dryad | 613 | 17/20 | 6/549 | 0/44 |
| HumBug | 592 | 30/40 | 17/459 | 0/93 |
| ESC50 | 289 | Sem Aedes | Sem outros mosquitos | 0/289 |

As fontes somam exatamente 1.494 grupos. Ausência de uma classe é reportada como métrica não estimável, sem atribuir sensibilidade ou especificidade fictícia.

O bootstrap pareado usa 2.000 réplicas de blocos PCM, com estratificação pelo conjunto de classes do bloco. Para a cascata, o intervalo percentil de 95% da sensibilidade é **66,67%–88,33%**, e o do FPR em outros mosquitos é **1,39%–3,27%**. A diferença de sensibilidade entre cascata temporal e sem persistência foi de **−3,33 pontos percentuais**, com intervalo pareado de **−8,33 a 0,00 pontos**. Não há evidência de ganho temporal nesta rodada.

Zero erros observados pode produzir um intervalo bootstrap degenerado `[0, 0]`; isso não demonstra risco populacional zero. O intervalo Wilson descritivo para 0/426 vai até aproximadamente 0,89%, sob hipótese binomial de independência, que não é uma confirmação de independência biológica. Os limites da reamostragem e os estratos com um único bloco estão registrados nos JSONs.

Matrizes, F1 por classe, fontes e intervalos estão em [event_level_metrics.json](../results/ternary_verified_2026_10_14_corrections/event_level_metrics.json). F1 macro, especificidade, cobertura e emissões por táxon de referência estão em [supplemental_metrics.json](../results/ternary_verified_2026_10_14_corrections/supplemental_metrics.json).

## 5. Testes e preservação do histórico

A suíte completa executada terminou com **276 testes aprovados**, 60.068 avisos, em 47,69 s:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

Os contratos novos cobrem ausência e conflito do mapa PCM, disjunção, ajuste de pesos/scalers, uso efetivo da persistência, tempos reais e lacunas, denominadores vazios, ESC50, limites inteiros, frontend de inferência e um ciclo pequeno real de treino/congelamento/avaliação com detecção de adulteração. Uma suíte aprovada significa que os contratos de software passaram; as metas científicas desta rodada continuam reprovadas.

Antes das alterações foram registrados hashes de 1.158 arquivos históricos. A conferência final encontrou **1.153 arquivos idênticos** e **cinco artefatos de resultados alterados concomitantemente no workspace**. As versões originais exatas de `b562b35` e as versões paralelas sem MosquitoSong foram preservadas em [historical_snapshots](../results/ternary_verified_2026_10_14_corrections/historical_snapshots/), com um registro dos caminhos e hashes. A alteração paralela permanece no caminho em que foi encontrada.

O diretório antigo `results/ternary_nested_2026_10_14` não é a evidência congelada desta nova implementação. Seus números não foram combinados com os desta rodada. Para a entrega atual, usar o diretório `ternary_verified_2026_10_14_corrections` e os respectivos manifestos.

Durante o trabalho, o checkout avançou por uma alteração concomitante para `a6e927f`, contendo os scripts corrigidos. A base registrada no congelamento continua sendo `b562b35`. O diretório da rodada verificada permanece local, excluído do Git por `.git/info/exclude`; os links para seus artefatos neste relatório se referem ao workspace local. A versão final deste relatório acrescenta as auditorias e os resultados externos que não constavam do texto incorporado naquele commit.

## 6. Comandos e direção técnica

Para consultar novamente os resultados concluídos, com verificação de integridade e sem reescrita:

```bash
.venv/bin/python evaluate_ternary_event_pipeline.py \
  --results-dir results/ternary_verified_2026_10_14_corrections
```

Uma reprodução completa precisa de outro diretório vazio e das entradas locais descritas no protocolo:

```bash
.venv/bin/python train_ternary_nested_models.py \
  --out results/ternary_verified_reproducao --phase all --threads 4
```

A inferência offline de um WAV usa o modelo final desta rodada e um novo diretório de saída:

```bash
.venv/bin/python predict_ternary.py caminho/gravacao.wav \
  --run results/ternary_verified_2026_10_14_corrections \
  --output results/inferencia_ternaria_nova
```

A avaliação de uma fonte excluída usa `evaluate_ternary_source_transfer.py --run <rodada> --source mosquitosong`; a aquisição após congelamento usa `fetch_ternary_fresh.py --run <rodada> --search-receipt <consulta-real.json>`. Ambos recusam substituir uma saída existente.

Esta versão encerra a rodada com treinamento concluído, modelo congelado e resultados OOF auditados. A cascata apresenta sensibilidade de 78,33% e baixo FPR para Aedes nesta população; a meta estrita de falsos positivos entre outros mosquitos permaneceu não atingida. O alcance dessa conclusão é o corpus e o protocolo descritos. Firmware, Arduino e instalador permanecem preservados.

## Apêndice A. Verificação complementar: MosquitoSong excluído do desenvolvimento

O teste separado contém **10 grupos e 2.739 janelas**: quatro grupos de Aedes e seis de outros mosquitos. Foram conferidos os bytes e as assinaturas PCM dos 1.322 arquivos dessa fonte contra os arquivos do treino, sem colisões detectadas pelo método declarado.

A cascata identificou **0/4 Aedes** e classificou corretamente os seis grupos negativos como outros mosquitos. Os dez grupos foram classificados como `other_mosquito`, resultando em acurácia de 60% e acurácia balanceada de 50%. A precisão de Aedes é indefinida, pois não houve emissão de Aedes. Como não há ruído puro nessa amostra, o FPR em ruído não é estimável.

É uma avaliação de transferência de fonte **externa ao ajuste desta rodada**, mas retrospectiva: o projeto já examinou essa fonte e suas anotações anteriormente. Sua exclusão atual não transforma esses áudios em confirmação cega independente. As identidades biológicas e as sessões independentes não são estabelecidas pela auditoria de arquivos.

Evidências: [métricas](../results/ternary_verified_2026_10_14_corrections/source_transfer_mosquitosong/metrics.json), [previsões por grupo](../results/ternary_verified_2026_10_14_corrections/source_transfer_mosquitosong/event_predictions.csv) e [auditoria de sobreposição](../results/ternary_verified_2026_10_14_corrections/source_transfer_mosquitosong/overlap_audit.json).

## Apêndice B. Verificação complementar: áudios obtidos depois do congelamento

A página oficial [BioDCASE 2026 no Zenodo, versão fixada nesta rodada](https://zenodo.org/records/19095788) foi consultada depois do congelamento. A seleção seguiu uma regra previamente registrada: primeiros dois arquivos numéricos `S_1_D_3` ainda ausentes do acervo local, com pelo menos 1,984 s reais. A escolha foi encerrada antes da inferência; não houve substituição por resultado desfavorável.

| Arquivo | Duração real | Referência do corpus | Decisão da cascata |
|---|---:|---|---|
| `S_1_D_3_91.wav` | 26,6565 s | Aedes aegypti | Outro mosquito — erro |
| `S_1_D_3_92.wav` | 27,910375 s | Aedes aegypti | Outro mosquito — erro |

Resultado: **0/2 Aedes**, sem janelas tecnicamente inválidas nos dois arquivos. As quatro políticas calibradas classificaram ambos como outro mosquito. A amostra contém somente positivos e não permite estimar precisão ou especificidade.

A verificação de novidade examinou 7.627 caminhos conhecidos, com 7.386 hashes de bytes distintos. Não detectou coincidência de bytes ou das assinaturas PCM declaradas nos dois novos arquivos. Houve 301 entradas sem PCM decodificável: 41 WAVs vazios, 255 metadados AppleDouble e cinco arquivos de teste do decodificador do SciPy. Nenhuma pertence aos 1.749 arquivos do desenvolvimento. O registro original mantém `complete_decoding_coverage: false`; a [conferência complementar](../results/ternary_verified_2026_10_14_corrections/postfreeze/novelty_error_diagnosis.json) explica cada falha sem apagar a limitação.

Esses são arquivos novos de um corpus já conhecido, com referência de espécie fornecida pelo corpus. Não são uma nova campanha biológica ou validação de campo; a independência de indivíduos e sessões permanece desconhecida. Pesos, frontend, limiares e regra de seleção não foram modificados após observar os erros.

Evidências: [consulta oficial](../results/ternary_verified_2026_10_14_corrections/postfreeze/search_receipt.json), [seleção antes da inferência](../results/ternary_verified_2026_10_14_corrections/postfreeze/selection_complete.json), [auditoria de novidade](../results/ternary_verified_2026_10_14_corrections/postfreeze/novelty_audit.json) e [resultados e hashes dos arquivos](../results/ternary_verified_2026_10_14_corrections/postfreeze/results.json).
