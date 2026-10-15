# Detector acústico ternário de mosquitos

O ponto de entrada é o [notebook de treinamento, teste e aplicação](notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb). Ele reúne **cascata clássica, LSTM e Bi-LSTM** para classificar Aedes aegypti, outro mosquito e ruído.

## O que está incluído

- Notebook didático com conceitos, métricas históricas, matrizes de confusão, treino opcional e aplicação em WAV.
- Módulos essenciais da cascata e das duas redes profundas, com divisão por grupos/blocos PCM, normalização, seleção interna, congelamento e verificação.
- Metadados de janelas/grupos e resultados neurais históricos compactos em `data/metadata/`.
- Testes de isolamento, arquitetura, integridade e execução do notebook sem arquivos locais ignorados.

## Executar o notebook

1. Crie um ambiente Python e instale `requirements.txt`. PyTorch está incluído; para GPU, use uma distribuição compatível com sua máquina. CPU também é suportada.
2. Execute as células em ordem. Os resultados de referência aparecem sem baixar áudios ou modelos e **sem iniciar treinamento**.
3. Para uma nova cascata, prepare as entradas locais e habilite `RUN_NEW_TRAINING`.
4. Para treinar **LSTM e Bi-LSTM**, habilite `RUN_NEURAL_TRAINING`. As redes usam a rodada clássica criada na sessão ou `NEURAL_REFERENCE_RUN`, que deve apontar para uma rodada concluída com cinco dobras externas. A mesma população e as mesmas partições são usadas na comparação.
5. Para aplicar as redes existentes, configure `EXISTING_NEURAL_RUN` com uma rodada neural concluída e `AUDIO_PATH` com um WAV. O notebook aplica as duas arquiteturas, sem retreinar nem ajustar limiares.
6. `VERIFY_NEURAL_RUN=True` reproduz as previsões OOF da rodada local. Essa verificação é opcional e pode demorar; não sobrescreve resultados existentes.

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

## Limites dos resultados

A avaliação é retrospectiva e mede grupos deste corpus. A cascata usa quatro dobras internas; as redes usam um holdout interno por grupos para selecionar épocas e limiares. Os cinco testes externos são compartilhados. Grupos não são necessariamente mosquitos individuais independentes.

O critério amostral de falsos positivos entre outras espécies não foi atingido na cascata. Os resultados não demonstram validação independente em novos locais, indivíduos, microfones ou condições de campo. As redes e seu método foram restaurados sem modificar a apresentação ou substituir os modelos históricos.
