# Resultados históricos do detector ternário

> **Aviso: resultados retrospectivos.** Os números desta página (incluindo 90,03% de acurácia geral e 47/60 Aedes) foram medidos com dobras por arquivo/grupo dentro do mesmo corpus. **Não há validação por sessão, ambiente ou aparelho**, e a [reanálise v2](reanalise_v2.md) mostrou que essas cifras não demonstram reconhecimento em gravações novas. O estado atual do projeto está no [README](../README.md) e em [continuacao_v3.md](continuacao_v3.md): nenhum modelo deste repositório está validado para uso operacional.
>
> Conteúdo movido do README em 22/10/2026 sem alteração de números. Caminhos entre crases são relativos à raiz do repositório.

## Experimento retrospectivo apresentado no notebook

O ponto de entrada é o [notebook de treinamento, teste e aplicação](../notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb). Ele explica **CNN, MLP, SVM, Extra Trees e HistGradientBoosting** para classificar Aedes aegypti, outro mosquito e ruído. O experimento retrospectivo do notebook usa 1.794 grupos/exemplos, com 360 Aedes. Cascata, LSTM e Bi-LSTM estão preservadas em um apêndice histórico de outro corpus.

### O que o notebook e os módulos incluem

- Notebook didático com conceitos, métricas históricas, matrizes de confusão, treino opcional e aplicação em WAV.
- Módulos essenciais da cascata e das duas redes profundas, com divisão por grupos/blocos PCM, normalização, seleção interna, congelamento e verificação.
- Metadados de janelas/grupos e resultados neurais históricos compactos em `data/metadata/`.
- Retrato verificado do experimento retrospectivo, com resultados por dataset, curvas, avaliação por fonte e 16 gráficos incorporados no notebook executado.
- Testes de isolamento, arquitetura, integridade e execução do notebook sem arquivos locais ignorados.

## Resultados históricos

A cascata obteve 90,03% de acurácia geral e detectou 47/60 grupos Aedes (78,33%). Os resultados originais de LSTM/Bi-LSTM estão em `data/metadata/ternary_neural_reference.json`, com as duas regras de decisão e hashes de origem. A maior acurácia observada foi 86,55% para LSTM e 85,27% para Bi-LSTM; as sensibilidades das mesmas configurações foram 33,33% e 28,33%. O notebook mostra as métricas juntas para evitar confundir acurácia geral com detecção de Aedes.

Esses números são históricos, não recalculados ao abrir o caderno. O corpus é real (Dryad/HumBug/ESC-50); os testes sintéticos e os experimentos posteriores com a base de infecção não entram nessa referência.

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
