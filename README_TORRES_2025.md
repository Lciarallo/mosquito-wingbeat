# Detector ternário e estudos de redes recorrentes

## Objetivo do projeto

O detector precisa emitir somente uma destas três classes:

| Código | Saída | Conteúdo |
|---|---|---|
| 0 | `no_mosquito` | Ruído/ambiente, sem mosquito detectado |
| 1 | `other_mosquito` | Mosquito de espécie diferente de Aedes aegypti |
| 2 | `aedes_aegypti` | Aedes aegypti |

Não é requisito identificar individualmente as outras 19 espécies nem inferir
infecção por dengue. Esses rótulos pertencem aos estudos auxiliares abaixo.

O áudio privado do **Drive (enxame)** está excluído do treino, normalização,
seleção, calibração, teste e das comparações acústicas correntes. Ele não entrou
na população que produziu os 90,03% nem nos resultados da LSTM. As análises
antigas desse áudio são somente arquivo histórico, fora da avaliação atual.
[Escopo e auditoria da exclusão](reports/exclusao_audio_drive_2026_10_15.md).

A referência principal é o [detector ternário verificado](reports/correcoes_ternarias_2026_10_14.md),
com inferência em `predict_ternary.py`. Seus resultados OOF são 90,03% de acurácia
geral e 78,33% de sensibilidade de Aedes (47/60); o ruído foi chamado de Aedes em
0/426 grupos e outros mosquitos foram chamados de Aedes em 23/1008 grupos.
Houve também 102/426 ruídos chamados de outro mosquito. As métricas de presença
e de identificação de Aedes precisam permanecer explícitas.

[Notebook de treinamento, avaliação OOF e aplicação em WAV](notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb).

## Comparação ternária com LSTM e Bi-LSTM

As duas redes foram treinadas e avaliadas nas mesmas 42.029 janelas, 1.494 grupos
e cinco dobras externas da referência. Na decisão direta, a LSTM atingiu 85,74%
de acurácia geral e a Bi-LSTM, 82,33%. Com limiares escolhidos na validação interna,
os valores foram 86,55% e 85,27%, acompanhados de queda na sensibilidade de Aedes.
A cascata clássica mantém a maior acurácia geral desta comparação (90,03%).

- [Relatório completo, sensibilidade e falsos positivos](reports/lstm_ternaria_2026_10_15.md).
- [Protocolo e análise de lacunas antes do treino](reports/analise_lacunas_lstm_ternaria_apresentacao_2026_10_15.md).
- [Métricas e evidências verificadas](results/ternary_lstm_2026_10_15/metrics.json).
- [Critérios acadêmicos e correspondência com os slides](output/ifsc_ternario_lstm_2026_10_15/criterios_apresentacao.md).

```bash
.venv/bin/python verify_ternary_lstm.py results/ternary_lstm_2026_10_15
.venv/bin/python predict_ternary_lstm.py caminho/gravacao.wav --output resultado.json
```

A inferência recebe WAV real com pelo menos 0,992 s, usa a cascata neural congelada
e inclui 100 passagens de Monte Carlo dropout. As métricas OOF usam inferência
determinística. MosquitoSong permanece fora de todo ajuste.

Os 62,11% de acerto e 72,00% de F1 macro do estudo de 20 espécies abaixo pertencem
a outra tarefa e não medem o desempenho do detector de três classes.

## Estudos auxiliares

As técnicas de Torres et al. foram aplicadas ao dataset de infecção DENV-2 dos
autores e à classificação Dryad de 20 espécies. Os novos treinos usam validação
separada e Monte Carlo dropout com 100 passagens.

- [Aplicação do artigo: resultados, gráficos e comandos](reports/aplicacao_torres_2025_2026_10_15.md).
- [Pré-processamento das 20 espécies](reports/preprocessamento_especies_2026_10_15.md).
- [Protocolo da nova comparação](reports/analise_lacunas_preprocessamento_especies_2026_10_15.md).

```bash
.venv/bin/python fetch_torres_2025.py
.venv/bin/python predict_torres_2025.py results/torres_2025_2026_10_15/species --audio caminho/gravacao.wav
.venv/bin/python predict_torres_2025.py results/torres_2025_2026_10_15/infection --author-row 0
```

O modelo de infecção recebe os tensores mel originais. A identificação de espécie
recebe áudio. Os relatórios mantêm as duas tarefas e seus denominadores próprios.
O `README.md` histórico está protegido pelos manifests das avaliações anteriores;
este arquivo concentra as novas entradas de documentação.
