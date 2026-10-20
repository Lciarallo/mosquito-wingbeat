# Detector acústico ternário de mosquitos

> **Continuação v3 concluída (20/10/2026).** Foram comparados seis candidatos, com Perch congelado, MLP/MMD, descritores do batimento e HistGB. Nenhum atingiu os critérios de desenvolvimento. O candidato descritivo final detectou **0/197 Aedes** com o portão de concordância; sem esse portão, **17/197**, com **20/371** outros mosquitos chamados Aedes. A rodada não justifica substituir o modelo anterior para uso operacional. [Relatório e limites](reports/continuacao_v3.md) · [Resultados auditáveis](reanalysis/outputs/v3_continuation/).
>
> Os experimentos anteriores permanecem preservados como resultados retrospectivos dos respectivos corpora e protocolos. As novas rodadas ampliam a investigação de generalização; elas não isolam o efeito de cada mudança de arquitetura, processamento e composição dos dados.

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

O ponto de entrada é o [notebook de treinamento, teste e aplicação](notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb). Ele explica **CNN, MLP, SVM, Extra Trees e HistGradientBoosting** para classificar Aedes aegypti, outro mosquito e ruído. O experimento atual usa 1.794 grupos/exemplos, com 360 Aedes. Cascata, LSTM e Bi-LSTM estão preservadas em um apêndice histórico de outro corpus.

## O que está incluído

- Notebook didático com conceitos, métricas históricas, matrizes de confusão, treino opcional e aplicação em WAV.
- Módulos essenciais da cascata e das duas redes profundas, com divisão por grupos/blocos PCM, normalização, seleção interna, congelamento e verificação.
- Metadados de janelas/grupos e resultados neurais históricos compactos em `data/metadata/`.
- Retrato verificado do experimento atual, com resultados por dataset, curvas, avaliação por fonte e 16 gráficos incorporados no notebook executado.
- Testes de isolamento, arquitetura, integridade e execução do notebook sem arquivos locais ignorados.

## Executar o notebook

1. Crie um ambiente Python e instale `requirements.txt`. PyTorch está incluído; para GPU, use uma distribuição compatível com sua máquina. CPU também é suportada.
2. Execute as células em ordem. Os resultados de referência aparecem sem baixar áudios ou modelos e **sem iniciar treinamento**.
   A parte principal mostra o experimento atual com 360 exemplos Aedes; o apêndice identifica explicitamente o corpus antigo com 60 grupos Aedes.
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

## Resultados históricos

A cascata obteve 90,03% de acurácia geral e detectou 47/60 grupos Aedes (78,33%). Os resultados originais de LSTM/Bi-LSTM estão em `data/metadata/ternary_neural_reference.json`, com as duas regras de decisão e hashes de origem. A maior acurácia observada foi 86,55% para LSTM e 85,27% para Bi-LSTM; as sensibilidades das mesmas configurações foram 33,33% e 28,33%. O notebook mostra as métricas juntas para evitar confundir acurácia geral com detecção de Aedes.

Esses números são históricos, não recalculados ao abrir o caderno. O corpus é real (Dryad/HumBug/ESC-50); os testes sintéticos e os experimentos posteriores com a base de infecção não entram nessa referência.

## Testes

Execute `python -m pytest -q tests/test_ternary_verified.py tests/test_ternary_lstm.py tests/test_notebook_workflow.py`. O teste que acessa o corpus completo é pulado quando os caches locais estão ausentes; os testes de treino mínimo e do notebook não precisam do corpus nem dos checkpoints históricos. As pequenas entradas geradas nos testes verificam o software e não compõem os resultados científicos.

## Novos classificadores sem recorrência

`pattern_classifiers.py` e `train_pattern_classifiers.py` implementam **CNN 2D, MLP, SVM RBF, Extra Trees e HistGradientBoosting** para a tarefa ternária. A CNN aprende padrões no espectrograma; as demais técnicas recebem 80 estatísticas Mel por janela. LSTM também permite classificação, mas não é necessária para essa abordagem.

O experimento local `results/pattern_classifiers_2026_10_15/` usa as cinco dobras por grupo da base unificada: 1.794 grupos/exemplos, incluindo os 300 espectrogramas reais de Torres. Infectados e saudáveis são Aedes; não se prevê infecção. O treino equilibra fonte/classe. Parâmetros, épocas e arquitetura são escolhidos na validação interna, com normalização ajustada apenas no treino. Notebook e apresentação foram atualizados; os pesos e resultados científicos anteriores permanecem preservados.

`data/metadata/pattern_classifier_reference.json` contém o retrato verificado usado pelo notebook; `pattern_reporting.py` produz os gráficos a partir desses números. A entrada de sequência real incluída no retrato serve somente para a figura didática, não substitui o cache para treinar. Uma revisão autorizada do notebook é registrada por recibo próprio para não alterar o protocolo original: `verify_pattern_classifiers.py --document-revision caminho/recibo.json` aceita somente a mudança documental explicitamente registrada e continua rejeitando qualquer alteração de pesos protegidos.

Torres não fornece todos os áudios brutos. Esta rodada usa um frontend comum **aproximado** de 179 instantes × 16 bandas abaixo de 4 kHz, diferente da entrada das redes originais. A acurácia não deve ser comparada diretamente aos 90,03% históricos. O relatório local separa Aedes originais de Torres, falsos alarmes e testes exploratórios deixando uma fonte inteira fora do treino.

Para executar, prepare explicitamente `segments.csv.gz` (com `window_target`, `true_label`, `domain`, `group`, `pcm_block`, `fold`, `row_id`), `classical_features.npy`, `feature_cache.json` e `protocol.json` da entrada comum. O JSON do cache contém a identidade e SHA-256 do `.npy` de sequências; o protocolo contém `protected_sha256` dos artefatos preservados. A rodada local `results/torres_tres_tecnicas_2026_10_15/` contém essas entradas, e seu gerador e dependências estão arquivados junto aos experimentos. Esses dados/caches não são distribuídos no Git.

```bash
python train_pattern_classifiers.py \
  --reference results/torres_tres_tecnicas_2026_10_15 \
  --sequences /tmp/mosquito_three_models_sequences_2026_10_15.npy \
  --out results/pattern_classifiers_2026_10_15
python verify_pattern_classifiers.py \
  --reference results/torres_tres_tecnicas_2026_10_15 \
  --sequences /tmp/mosquito_three_models_sequences_2026_10_15.npy \
  --out results/pattern_classifiers_2026_10_15
python predict_pattern.py \
  --model-dir results/pattern_classifiers_2026_10_15/final \
  --audio caminho/gravacao.wav
python -m pytest -q tests/test_pattern_classifiers.py
```

`--stage ml` e `--stage neural` permitem executar as famílias separadamente. O treino não publica nem substitui modelos históricos. A inferência exige pesos finais e normalizador, além do manifesto de integridade; o comando de treino gera os modelos por dobra, enquanto o ajuste final está documentado no experimento local. Os escores de saída não representam probabilidades de campo calibradas.

## Limites dos resultados

A avaliação é retrospectiva e mede grupos deste corpus. A cascata usa quatro dobras internas; as redes usam um holdout interno por grupos para selecionar épocas e limiares. Os cinco testes externos são compartilhados. Grupos não são necessariamente mosquitos individuais independentes.

O critério amostral de falsos positivos entre outras espécies não foi atingido na cascata. Os resultados não demonstram validação independente em novos locais, indivíduos, microfones ou condições de campo. As redes e seu método foram restaurados sem modificar a apresentação ou substituir os modelos históricos.
