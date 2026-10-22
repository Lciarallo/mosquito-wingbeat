# Detector acústico ternário de mosquitos

Classifica gravações em três classes: ruído, outro mosquito e *Aedes aegypti*.

## Estado atual: não usar operacionalmente

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

Execute `python -m pytest -q`. O `pytest.ini` coleta seis arquivos versionados (70 testes):
`test_ternary_verified.py`, `test_ternary_lstm.py`, `test_notebook_workflow.py`, `test_pattern_classifiers.py`, `test_reanalysis_v2.py` e `test_v3_continuation.py`. O teste que acessa o corpus completo é pulado quando os caches locais estão ausentes; os testes de treino mínimo e do notebook não precisam do corpus nem dos checkpoints históricos. As pequenas entradas geradas nos testes verificam o software e não compõem os resultados científicos.

Outros arquivos `tests/test_*.py` existem apenas em cópias locais de trabalho e **não são versionados nem coletados**. Revisão de 22/10/2026, com `.venv-v3`:

| Situação | Arquivos | Motivo de ficarem fora |
|---|---|---|
| Passam num clone limpo | `test_torres_lstm.py`, `test_aedes_classical.py` | Importam só módulos versionados. Candidatos a entrar na suíte (pendente de decisão). |
| Passam só na cópia local | `test_species_preprocessing`, `test_notebook_transfer`, `test_nested_species`, `test_nested_species_v5`, `test_aedes_binary`, `test_expanded_protocol`, `test_large_audio_model`, `test_audio_preprocessing`, `test_multisource_protocol`, `test_publication_artifacts`, `test_ternary_synthetic`, `test_ternary_detector` | Dependem de módulos ou dados locais não versionados; falham num clone. |
| Falham | `test_nested_species_v2`, `_v3`, `_v4`, `test_faithful_reproduction`, `test_drive_exclusion` | Conferem hashes de artefatos de 14/10 que mudaram depois (`.gitignore`, toolchain Arduino), carregam pickles de uma versão antiga do scikit-learn (`_loss`) ou exigem `firmware/README.md` e uma rodada clássica real. |
| Não importam | `test_reliability`, `test_reproduction`, `test_research_audit`, `test_corrected_validation`, `test_cross_domain_strategies`, `test_sex_balanced_reliability` | Pedem `aedes_classical.raw_scores`, que não existe mais no módulo atual, ou o pacote `requests`. |

## O que dá e o que não dá para reproduzir

Dados, caches e pesos ficam fora do Git. Conferido em 22/10/2026 num clone limpo de `detector-v3`, com `.venv-v3`:

**Roda só com o clone**

| Etapa | Como | Observação |
|---|---|---|
| Testes | `python -m pytest -q` | 68 aprovados e 2 pulados. Os 2 pulados exigem caches locais (`test_ternary_verified.py:184`) e o manifesto local da v3 (`test_v3_continuation.py:23`). Com os dados locais, são 70 aprovados. |
| Notebook sem treino | executar as células com as flags de treino desligadas | Mostra os retratos históricos versionados em `data/metadata/`. |
| Leitura e auditoria dos resultados v2/v3 | `reports/`, `reanalysis/outputs/` | As predições por arquivo da v2 (`reanalysis/outputs/predictions/*.csv`, com `p0`, `p1`, `p2`) permitem recalcular as métricas pontuais. As métricas, a figura e os congelamentos da v3 estão em `reanalysis/outputs/v3_continuation/`. |

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
| Cascata e redes históricas | ver "Entradas locais para treinar" | `data/cache/reliability_2026_10_01/` e rodadas em `results/` |

O repositório não traz scripts de download para todos os corpora. Os protocolos congelados (`reanalysis/protocol_*.json`, `*_freeze.json`) registram os SHA-256 esperados das entradas.

## Resultados históricos

Movidos para [reports/historico_ternario.md](reports/historico_ternario.md), sem alteração de números: cascata (90,03% / 47/60 Aedes), LSTM/Bi-LSTM, classificadores sem recorrência (CNN, MLP, SVM, Extra Trees, HistGB) com os comandos de treino, verificação e inferência, e os limites desses resultados. **São retrospectivos e não têm validação por sessão, ambiente ou aparelho.**

## Conteúdo do repositório

- `notebooks/`: notebook didático (conceitos, métricas históricas, treino opcional, aplicação em WAV).
- Módulos na raiz: cascata ternária, LSTM/Bi-LSTM e classificadores sem recorrência, com divisão por grupos, congelamento e verificação.
- `reanalysis/`: protocolos pré-registrados v2/v3 (congelados por SHA-256), código e saídas leves auditáveis.
- `reports/`: relatórios da v2, da v3, análise de lacunas e resultados históricos.
- `data/metadata/`: metadados de janelas/grupos e retratos compactos dos resultados históricos.
- `tests/`: testes de isolamento, integridade, protocolo e execução do notebook.
