# Treino para identificar Aedes com os exemplos de Torres — 15/10/2026

## O que foi implementado

Um novo detector de **Aedes aegypti / outro mosquito / ruído**, usando árvores em boosting (HistGradientBoostingClassifier). Todos os exemplos de Torres foram rotulados como Aedes; infectado e saudável não são classes deste modelo. Trata-se de um estudo separado: a cascata e as redes anteriores, seus resultados e a apresentação foram preservados.

## Treino, validação e teste

Dos 300 tensores de Torres: **192 treino, 48 validação, 60 teste**. As mesmas divisões por hash exato já arquivadas foram reutilizadas. Os dados de Torres foram adicionados somente ao treino/validação do candidato correspondente; o teste não foi usado para ajustar pesos ou escolher hiperparâmetros. Não há IDs de indivíduos ou gravações no JSON: a separação de hashes não demonstra separação biológica. A fonte e esses exemplos já foram estudados em outro experimento; isto é uma comparação retrospectiva.

Nas bases originais: dobras 2, 3 e 4 para treino, dobra 1 para validação e dobra 0 para teste (293 grupos: 86 ruídos, 195 outros mosquitos, 12 Aedes). Grupos e blocos PCM ficam em uma única partição. Não se trata da validação cruzada de cinco testes que produziu os 90,03% da cascata anterior.

Comparação entre dois NOVOS modelos, com o mesmo teste e escolha apenas pela validação:

| Medida de teste | Sem Torres | Com Torres |
|---|---:|---:|
| Aedes reconhecidos nos 60 exemplos Torres reservados | 0/60 (0,00%) | **56/60 (93,33%)** |
| Aedes reconhecidos nas bases originais | 3/12 (25,00%) | 3/12 (25,00%) |
| Acertos gerais nos 293 grupos das bases originais | 237/293 (80,89%) | 242/293 (82,59%) |
| Ruídos que geraram alerta Aedes | 0/86 (0,00%) | 0/86 (0,00%) |
| Outros mosquitos que geraram alerta Aedes | 5/195 (2,56%) | 6/195 (3,08%) |

Os 93,33% são a **sensibilidade entre positivos Torres**, não uma acurácia ternária de 93,33% ou uma medição de falso positivo nessa fonte, que só tem Aedes. A inclusão ajudou a reconhecer os exemplos reservados de Torres, mas não aumentou a detecção dos Aedes das outras bases neste recorte. Os sinais de origem das bases podem contribuir para a decisão; este teste não separa completamente reconhecimento biológico de reconhecimento de fonte.

## Teste ilustrativo em WAV

Com os modelos já congelados, foram testados os dois WAVs exemplificativos:

| Áudio | Novo modelo sem Torres | Novo modelo com Torres |
|---|---|---|
| infectado | Ruído | **Aedes aegypti**, probabilidade de 99,79% |
| saudável | Ruído | Ruído |

A infecção não é usada na decisão: a classe correta dos dois é Aedes. O resultado foi 1/2 acertos do novo modelo com Torres nos exemplos WAV. Não se sabe se esses WAVs correspondem a tensores usados no treino; portanto não são anunciados como teste independente. Os 99,79% são um escore de modelo, não uma certeza calibrada.

## Compatibilização dos dados

O artigo publica 48 kHz, FFT1024, avanço256, janela Hann e 32 bandas mel, mas não disponibiliza a receita completa de extração dos 300 sinais. Foram avaliadas duas aproximações documentadas, HTK e Slaney, com faixa0–24kHz, e seleção somente pela validação. Ambas as variantes escolheram **HTK, 15 folhas, 180 iterações, taxa0,06, regularização L2=1**.

Os áudios do corpus foram reamostrados de8 para48kHz com filtro polifásico (isso não cria conteúdo acima de4kHz). Foram usadas apenas bandas cujos limites superiores são no máximo4kHz. Por gravação, até três janelas de0,992s foram selecionadas pelo RMS antes do treino. Os atributos são média, desvio padrão e percentis10/50/90 da energia mel relativa em dB, retirando dois quadros de cada borda. Os tensores do artigo passaram pelo mesmo resumo e escala relativa, sem interpolação de bandas ou duração. A compatibilidade é uma aproximação, não reprodução comprovada da extração original.

As classes receberam massa de peso equilibrada. No candidato com Torres, metade da massa de Aedes veio do corpus original e metade de Torres. O estado de infecção, IDs e fonte não entraram nos atributos do modelo. A regra final é a maior probabilidade entre três classes; não houve ajuste de limiar no teste.

## Verificação e reprodução

Todas as probabilidades de teste salvas foram reproduzidas a partir dos modelos finais e dos atributos; partições por identidade e rótulos dos grupos foram conferidos. Durante a revisão, uma gravação com rótulos de janela mistos na validação teve o rótulo de evento corrigido para o máximo sobre todas as janelas, conforme o protocolo original. A execução anterior foi preservada em `results/torres_aedes_transfer_2026_10_15_pre_validation_label_correction/`; os candidatos foram reexecutados sem acrescentar configurações ou usar o teste na seleção. As escolhas e os resultados de teste permaneceram iguais.

- Treino: `.venv/bin/python train_torres_aedes_transfer.py` (recusa sobrescrever a pasta existente).
- Inferência: `.venv/bin/python predict_torres_aedes_transfer.py caminho/arquivo.wav`.
- Verificação: `.venv/bin/python verify_torres_aedes_transfer.py`.
- `protocol.json`, `selection_frozen.json`: configuração e seleção anterior à inferência de teste.
- `partitions.csv`, `selected_windows.csv`: grupos e janelas usados.
- `test_predictions.csv`, `metrics.json`: previsões e métricas, com denominadores separados.
- `wav_predictions.json`: inferência ilustrativa nos dois WAVs.
- `verification.json`: reprodução, hashes e preservação dos modelos/PDF anteriores.

Os checkpoints e PDF anteriores mantiveram os hashes. Nenhuma chamada de edição dos slides ou atualização de seus arquivos no Drive foi feita. O novo modelo não substituiu o detector anterior.

## Fontes locais

- Artigo fornecido: Torres et al., *Dengue-Infected Mosquito Detection with Uncertainty Evaluation of the Wingbeats using Monte Carlo Dropout*, IEEE Latin America Transactions, 2025, Tabela I e descrição da coleta.
- `data/external/torres_2025/manifest.json`, `Dataset/Melspec.json` e repositório dos autores fixado no commit6c96ccd5347cf5bafc9b424579700af7b5ffb7ce.
- `results/ternary_lstm_2026_10_15/segments.csv.gz` e partições arquivadas do estudo de infecção.
