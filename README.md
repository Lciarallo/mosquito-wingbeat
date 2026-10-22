# Detector acústico ternário de mosquitos

Classifica gravações em três classes: ruído, outro mosquito e *Aedes aegypti*.

## Estado atual: não usar operacionalmente

> **v4 (22/10/2026): sem melhora demonstrável; resposta sobre o teste novo: inconclusiva.** Rodada pré-registrada ([plano](reports/plano_v4.md), [protocolo](reanalysis/protocol_v4.json) congelado por SHA-256, emendas A1 e A2) com oito candidatos: Perch congelado, CORAL, DANN, bandas ≥ 300 Hz, descritores do batimento, aumento de dados com RIR/ruído externos e **fine-tuning do Perch na GPU**. Avaliação deixando cada corpus de fora (LODO), com IC por cluster e calibração local com 5/10/20 arquivos. **Nenhum candidato atingiu as metas** (recall ≥ 20%, FPR ≤ 2% em outros mosquitos e ≤ 1% em ruído, por corpus); o AUROC médio ficou entre 0,44 e 0,61, com IC sobrepostos, e a origem da gravação segue decodificável (92% no Perch, acaso 33%). **Não há teste novo disponível** que cumpra os critérios ([levantamento de datasets](reports/datasets_candidatos_v4.md)); o candidato descritivo foi congelado antes de qualquer dado novo. O gargalo é a diversidade de dados: só 16 clusters com Aedes no desenvolvimento. [Resultados](reports/resultados_v4.md) · [Saídas auditáveis](reanalysis/outputs/v4/).

> **Continuação v3 concluída (20/10/2026).** Foram comparados seis candidatos, com Perch congelado, MLP/MMD, descritores do batimento e HistGB. Nenhum atingiu os critérios de desenvolvimento. O candidato descritivo final detectou **0/197 Aedes** com o portão de concordância; sem esse portão, **17/197**, com **20/371** outros mosquitos chamados Aedes. A rodada não justifica substituir o modelo anterior para uso operacional. [Relatório e limites](reports/continuacao_v3.md) · [Resultados auditáveis](reanalysis/outputs/v3_continuation/).
>
> Os experimentos anteriores permanecem preservados como resultados retrospectivos dos respectivos corpora e protocolos. As novas rodadas ampliam a investigação de generalização; elas não isolam o efeito de cada mudança de arquitetura, processamento e composição dos dados.

**Resumo.** Nenhum modelo deste repositório está validado para uso em campo, em ambientes ou em aparelhos novos. Os números antigos (por exemplo, 90,03% de acurácia e 47/60 Aedes) são **retrospectivos**: foram medidos com dobras por arquivo/grupo do mesmo corpus, **sem validação por sessão, ambiente ou aparelho**, e estão documentados à parte em [Resultados históricos](reports/historico_ternario.md).

### Por que os números antigos não descrevem o estado atual

> **Reanálise v2 (20/10/2026).** Uma auditoria identificou três limitações relevantes para generalização:
> - **Atalho de fonte.** Com a fonte inteira fora do treino, a detecção de Aedes caía para 0–12%.
> - **Rótulo igual à fonte.** O Torres é 100% Aedes e o ESC-50 é 100% ruído.
> - **Dobras por arquivo, não por sessão.** Por isso as cifras de 89–90% de acurácia e 98% de Aedes **não demonstram** reconhecimento em sessões, ambientes ou aparelhos novos.
>
> A reanálise usa protocolo pré-registrado (`reanalysis/protocol_v2.json`, congelado por SHA-256), um frontend comum (8 kHz, 100–3.400 Hz, ganho normalizado) e validação por **cluster** (sessão, laboratório+aparelho ou dia de gravação). Os intervalos de confiança vêm de bootstrap por cluster. O Torres sai do conjunto, porque não há áudio bruto e todos os exemplos são Aedes.
>
> Resultados por arquivo, com IC95%:
>
> | Avaliação | Melhor AUROC Aedes | Acurácia balanceada | Detecção de Aedes (argmax) |
> |---|---|---|---|
> | E1 · CV por cluster, todas as fontes | 0,91 [0,86–0,94] (HistGB) | 0,66 [0,62–0,71] | 18–53% (precisão 14–46%) |
> | E2 · Dryad → HumBug | 0,84 [0,75–0,93] (CNN) | 0,61 [0,59–0,64] | 0–22,5% |
> | E2 · HumBug → Dryad | 0,79 [0,56–0,96] (HistGB) | ≤0,25 (chance 0,50) | 0–10% |
> | E3 · MosquitoSong (externo, nunca visto) | ≤0,44 (abaixo do acaso) | ≤0,42 (chance 0,50) | 2–12% (CNN, HistGB, logística) |
>
> **Conclusão da v2.** O critério declarado antes de rodar era IC inferior > acaso em E1 **e** E2. Nenhum modelo o cumpre nas duas direções, e todos falham no corpus externo.
> - A origem da gravação continua decodificável dos atributos (acurácia balanceada de 0,96 ao distinguir Dryad de HumBug).
> - Por oclusão de bandas, os modelos dependem principalmente da faixa de 130–280 Hz, abaixo do batimento de asa do Aedes.
> - Essa rodada **não demonstrou confiabilidade operacional** em ambientes ou aparelhos novos. Ver [`reports/reanalise_v2.md`](reports/reanalise_v2.md). Decodificar a origem ou observar dependência de uma banda espectral não prova, por si só, que toda decisão use exclusivamente o aparelho ou o dataset.
## Como reproduzir

O ponto de entrada é o [notebook de treinamento, teste e aplicação](notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb). Ele apresenta o experimento retrospectivo (1.794 grupos/exemplos, com 360 Aedes; cascata, LSTM e Bi-LSTM em um apêndice histórico de outro corpus). Os resultados mostrados nele são históricos; ver o aviso acima.

### Executar o notebook

1. Crie um ambiente Python e instale `requirements.txt`. PyTorch está incluído; para GPU, use uma distribuição compatível com sua máquina. CPU também é suportada.
2. Execute as células em ordem. Os resultados de referência aparecem sem baixar áudios ou modelos e **sem iniciar treinamento**.
   A parte principal mostra o experimento retrospectivo com 360 exemplos Aedes; o apêndice identifica explicitamente o corpus antigo com 60 grupos Aedes.
3. Para uma nova cascata, prepare as entradas locais e habilite `RUN_NEW_TRAINING`.
4. Para treinar **LSTM e Bi-LSTM**, habilite `RUN_NEURAL_TRAINING`. As redes usam a rodada clássica criada na sessão ou `NEURAL_REFERENCE_RUN`, que deve apontar para uma rodada concluída com cinco dobras externas. A mesma população e as mesmas partições são usadas na comparação.
5. Para aplicar as redes existentes, configure `EXISTING_NEURAL_RUN` com uma rodada neural concluída e `AUDIO_PATH` com um WAV. O notebook aplica as duas arquiteturas, sem retreinar nem ajustar limiares.
6. `VERIFY_NEURAL_RUN=True` reproduz as previsões OOF da rodada local. Essa verificação é opcional e pode demorar; não sobrescreve resultados existentes.
7. Para os modelos atuais, use `RUN_PATTERN_TRAINING` (novo treino), `RUN_PATTERN_VERIFY` (reprodução auditada) e `PATTERN_AUDIO_PATH` (inferência com os pesos finais locais). Os três fluxos estão explicados passo a passo no caderno e ficam desativados por padrão.

### Entradas locais para treinar

A cascata precisa do manifesto e mapa versionados, dos áudios referenciados e de `data/cache/reliability_2026_10_01/features.npy`, alinhado linha a linha ao manifesto. O protocolo também confere os caches de áudio `long_8k.npy`/`short_8k.npy` correspondentes aos buckets presentes.

As redes precisam de `segments.csv.gz`, `partitions.csv.gz` e `event_level_metrics.json` de uma rodada clássica concluída, além de `data/cache/reliability_2026_10_01/long_8k.npy`. Esse arquivo guarda janelas reais de 7.936 amostras a 8 kHz nos índices do manifesto; as sequências de 55 instantes × 64 bandas mel são extraídas automaticamente.

Áudios, caches, checkpoints e demais saídas permanecem fora do Git. **Metadados sozinhos não permitem treinar.** O repositório enxuto não inclui os antigos scripts de aquisição/reconstrução de todo o corpus; esses artefatos locais devem ser preparados ou recuperados separadamente. Não há download nem preenchimento artificial de áudio no notebook.

Uma rodada completa pode demorar bastante e cria uma pasta nova em `results/`. Não altere arquivos congelados para reaproveitar um resultado antigo: os hashes protegem código, partições, pesos e normalizadores. O verificador neural pode ler fontes originais arquivadas na rodada, sem substituir o código do checkout.

### Testes

Execute `python -m pytest -q`. O `pytest.ini` coleta nove arquivos versionados: `test_ternary_verified.py`, `test_ternary_lstm.py`, `test_notebook_workflow.py`, `test_pattern_classifiers.py`, `test_reanalysis_v2.py`, `test_v3_continuation.py`, `test_torres_lstm.py`, `test_aedes_classical.py` e `test_v4.py`. Com `.venv-v3` e os dados locais: 103 aprovados e 1 pulado (a regra de camadas do fine-tuning exige `onnx2torch`, presente só em `.venv-v4-rocm`). `test_v4.py` verifica que o teste reservado da v3 nunca entra na v4, que não há vazamento entre clusters, a separação ajuste/calibração/teste, que o fine-tuning usa só as linhas de ajuste, CORAL sem rótulos, determinismo e os congelamentos. O teste que acessa o corpus completo é pulado quando os caches locais estão ausentes; os testes de treino mínimo e do notebook não precisam do corpus nem dos checkpoints históricos. As pequenas entradas geradas nos testes verificam o software e não compõem os resultados científicos.

Os demais `tests/test_*.py` existem só em cópias locais de trabalho e **não são versionados nem coletados**. Na revisão de 22/10/2026, cada um foi rodado isoladamente com `.venv-v3` na cópia local (com dados e módulos locais) e num clone limpo. Só entram na suíte os que passam sem dados locais. Nenhum deles é pulado por falta de dados: os que dependem de arquivos locais dão erro ou falham.

| Arquivo | Cópia local | Clone limpo | Classificação |
|---|---|---|---|
| `test_torres_lstm.py` | 13 aprovados | 13 aprovados | **Passa sem dados locais. Incluído.** |
| `test_aedes_classical.py` | 3 aprovados | 3 aprovados | **Passa sem dados locais. Incluído.** |
| `test_aedes_binary.py` | 3 aprovados | erro de importação | Depende de `aedes_binary.py`, não versionado. |
| `test_audio_preprocessing.py` | 4 aprovados | erro de importação | Depende de `audio_preprocessing.py`, não versionado. |
| `test_expanded_protocol.py` | 3 aprovados | erro de importação | Depende de `train_expanded.py`, não versionado. |
| `test_large_audio_model.py` | 3 aprovados | erro de importação | Depende de `large_audio_model.py`, não versionado. |
| `test_multisource_protocol.py` | 4 aprovados | erro de importação | Depende de `large_audio_model.py`, não versionado. |
| `test_nested_species.py` | 13 aprovados | erro de importação | Depende de `nested_species_ensemble.py`, não versionado. |
| `test_nested_species_v5.py` | 15 aprovados | erro de importação | Depende de `acoustic_features_v5.py`, `nested_species_ensemble_v5.py` e `research_metrics_v2.py`, não versionados. |
| `test_notebook_transfer.py` | 12 aprovados | erro de importação | Depende de `notebook_transfer_features.py` e `reproduce_notebook_transfer.py`, não versionados. |
| `test_publication_artifacts.py` | 5 aprovados | erro de coleta | Lê artefatos locais de publicação. |
| `test_species_preprocessing.py` | 10 aprovados | erro de importação | Depende de `species_preprocessing.py`, não versionado. |
| `test_ternary_detector.py` | 15 aprovados | 3 falhas, 4 erros, 3 pulados | Lê rodadas locais em `results/`. |
| `test_ternary_synthetic.py` | 9 aprovados | erro de importação | Depende de `evaluate_ternary_synthetic.py`, não versionado. |
| `test_drive_exclusion.py` | 1 falha, 3 aprovados | 3 falhas, 1 aprovado | Quebra: exige uma rodada clássica real (`partitions.csv.gz`, `event_level_metrics.json`). |
| `test_faithful_reproduction.py` | 1 falha, 7 aprovados | 8 falhas | Quebra: exige `firmware/README.md`, ausente. |
| `test_nested_species_v2.py` | 1 falha, 19 aprovados | erro de importação | Quebra: confere um arquivo da toolchain Arduino que mudou. |
| `test_nested_species_v3.py` | 2 falhas, 19 aprovados | erro de importação | Quebra: confere o hash do `.gitignore` de 14/10 e carrega pickle de scikit-learn antigo (`_loss`). |
| `test_nested_species_v4.py` | 2 falhas, 23 aprovados | erro de importação | Quebra: mesmos motivos do v3. |
| `test_reliability.py` | erro de importação | erro de importação | Quebra: pacote `requests` ausente. |
| `test_reproduction.py` | erro de importação | erro de importação | Quebra: importa `aedes_classical.raw_scores`, que não existe mais. |
| `test_research_audit.py` | erro de importação | erro de importação | Quebra: idem. |
| `test_corrected_validation.py` | erro de importação | erro de importação | Quebra: idem. |
| `test_cross_domain_strategies.py` | erro de importação | erro de importação | Quebra: idem. |
| `test_sex_balanced_reliability.py` | erro de importação | erro de importação | Quebra: idem. |

## O que dá e o que não dá para reproduzir

Dados, caches e pesos ficam fora do Git. Conferido em 22/10/2026 num clone limpo de `detector-v3`, com `.venv-v3`:

**Roda só com o clone**

| Etapa | Como | Observação |
|---|---|---|
| Testes | `python -m pytest -q` | 100 aprovados e 4 pulados (conferido em 22/10/2026). Os pulados exigem caches locais (`test_ternary_verified.py:184`), o manifesto local da v3 (`test_v3_continuation.py:23`, `test_v4.py:160`) e o Perch convertido (`test_v4.py:235`). |
| Notebook sem treino | executar as células com as flags de treino desligadas | Mostra os retratos históricos versionados em `data/metadata/`. |
| Leitura e auditoria dos resultados v2/v3/v4 | `reports/`, `reanalysis/outputs/` | As predições por arquivo da v2 (`reanalysis/outputs/predictions/*.csv`, com `p0`, `p1`, `p2`) permitem recalcular as métricas pontuais. As métricas, a figura e os congelamentos da v3 estão em `reanalysis/outputs/v3_continuation/`. |

**Exige dados ou pesos locais** (caminhos relativos à raiz)

| Etapa | Comando | Entradas esperadas |
|---|---|---|
| Inferência da cascata histórica | `python predict_ternary.py audio.wav --run <rodada> --output <saida>` | `<rodada>/models/final.joblib` |
| Inferência LSTM/Bi-LSTM histórica | `predict_ternary_lstm.py` | `results/ternary_lstm_2026_10_15/` |
| Inferência dos classificadores sem recorrência | `predict_pattern.py --model-dir ... --audio ...` | `results/pattern_classifiers_2026_10_15/final/` |
| Inferência de pesquisa da v3 | `python -m reanalysis.v3_predict --audio audio.wav` | `results/v3_continuation/final/` e `data/models/perch_v2_onnx/perch_v2.onnx` |
| Regerar o relatório da v3 | `python -m reanalysis.v3_report` | `results/v3_continuation/final/` |
| Manifesto e frontend da v2 | `reanalysis/build_manifest.py`, `reanalysis/frontend.py` | `data/raw/`, `data/external/humbug/neurips_2021_zenodo_0_0_1.csv`, `data/external/additional/mosquitosong/`; gera `results/v2/manifest_files.csv` e `data/cache/v2/` |
| Treino e avaliação da v2 | `python reanalysis/train_v2.py <etapa>` | `results/v2/manifest_files.csv`, `data/cache/v2/` |
| Continuação da v3 | `reanalysis/v3_*.py` | `results/v3_continuation/manifest.csv`, `data/cache/v3_continuation/`, modelo Perch acima |
| v4: cópia aumentada | `python -m reanalysis.v4_augment` | Manifesto v3, `data/external/openslr28/` (OpenSLR 28, 1,3 GB) |
| v4: fine-tuning do Perch (GPU AMD) | `.venv-v4-rocm/bin/python -m reanalysis.v4_finetune prepare` e `folds` | PyTorch 2.9.1+rocm6.4 (≈ 4,7 GB); `data/models/perch_v2_onnx/perch_v2.onnx` |
| v4: desenvolvimento, seleção, diagnósticos, congelamento, relatório | `python -m reanalysis.v4_train dev`, `select`, `diagnostics`, `final`; `python -m reanalysis.v4_report` | `data/cache/v3_continuation/`, `data/cache/v4_augmented/`, `results/v4/finetune/` |
| Cascata e redes históricas | ver "Entradas locais para treinar" | `data/cache/reliability_2026_10_01/` e rodadas em `results/` |

O repositório não traz scripts de download para todos os corpora. Os protocolos congelados (`reanalysis/protocol_*.json`, `*_freeze.json`) registram os SHA-256 esperados das entradas.

## Resultados históricos

Movidos para [reports/historico_ternario.md](reports/historico_ternario.md), sem alteração de números: cascata (90,03% / 47/60 Aedes), LSTM/Bi-LSTM, classificadores sem recorrência (CNN, MLP, SVM, Extra Trees, HistGB) com os comandos de treino, verificação e inferência, e os limites desses resultados. **São retrospectivos e não têm validação por sessão, ambiente ou aparelho.**

## Conteúdo do repositório

- `notebooks/`: notebook didático (conceitos, métricas históricas, treino opcional, aplicação em WAV).
- Módulos na raiz: cascata ternária, LSTM/Bi-LSTM e classificadores sem recorrência, com divisão por grupos, congelamento e verificação.
- `reanalysis/`: protocolos pré-registrados v2/v3/v4 (congelados por SHA-256, com emendas), código e saídas leves auditáveis.
- `reports/`: relatórios da v2, da v3 e da v4, plano da v4, datasets candidatos, análise de lacunas, resultados históricos, inventário da raiz e plano de limpeza do histórico git.
- `data/metadata/`: metadados de janelas/grupos e retratos compactos dos resultados históricos.
- `tests/`: testes de isolamento, integridade, protocolo e execução do notebook.
