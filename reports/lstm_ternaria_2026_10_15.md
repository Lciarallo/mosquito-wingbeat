# Comparação LSTM/Bi-LSTM para o detector ternário — 15/10/2026

Autor: Luiz Eduardo do Nascimento Ciarallo.

## Resultado

A cascata LR + HistGB mantém 90,03% de acurácia geral. Nas configurações avaliadas, LSTM e Bi-LSTM não mantiveram esse valor. A decisão direta das redes aumentou a sensibilidade de Aedes, acompanhada de mais falsos positivos. Os limiares conservadores reduziram falsos positivos e também a sensibilidade. Ambos os modos foram preservados, sem selecionar retrospectivamente apenas o melhor número.

| Modelo / decisão | Acurácia geral | Sensibilidade Aedes | Ruído → Aedes | Outro → Aedes |
|---|---:|---:|---:|---:|
| Cascata clássica | 90,03% | 78,33% | 0,00% | 2,28% |
| LSTM · direta | 85,74% | 86,67% | 3,76% | 3,97% |
| LSTM · calibrada | 86,55% | 33,33% | 0,47% | 0,60% |
| Bi-LSTM · direta | 82,33% | 88,33% | 4,93% | 6,25% |
| Bi-LSTM · calibrada | 85,27% | 28,33% | 0,00% | 0,40% |

Os denominadores são os mesmos: 1.494 grupos, sendo 426 de ruído, 1.008 de outros mosquitos e 60 de Aedes. Os 90,03% equivalem a 1.345/1.494 acertos. Zero ruído → Aedes (0/426) não é zero ruído → qualquer mosquito: na cascata, há 102/426 ruídos classificados como outro mosquito.

## Protocolo executado

Foram treinadas redes ternárias novas, sem importar pesos ou rótulos de infecção: duas camadas LSTM de 128 unidades, unidirecionais ou bidirecionais, dropout 0,20, camada linear de três saídas e softmax na inferência. A arquitetura é uma adaptação do artigo de Torres et al. (2025), que utiliza 256 unidades por camada em outro problema.

As 42.029 janelas de 0,992 s são exatamente as retidas no estudo clássico auditado. Os cinco testes externos usam as mesmas linhas e as mesmas partições; 1.441 blocos de PCM não atravessam treino e teste. Grupos com rótulos de janela mistos recebem o maior rótulo, conforme o contrato anterior. MosquitoSong permanece fora de todo desenvolvimento.

O frontend usa áudio de 8 kHz, FFT 1.024, Hann de 128 ms, avanço de 16 ms, 64 filtros mel de 80 a 4.000 Hz e 55 passos temporais. A separação entre bins é 7,8125 Hz. Não há recuperação de informação acima do Nyquist do áudio de origem. DC/RMS por janela, potência relativa em dB limitada a [-80,0], média e desvio por banda calculados somente no treino de cada ajuste.

O treino usa Adam, taxa 0,0006, lotes de 256, entropia cruzada, clipping da norma do gradiente em 3 e amostragem composta com gamma 0,50, teto de 60 janelas e massa equilibrada entre as três classes. Limite de 30 épocas e paciência de 5.

Dentro de cada treino externo, uma partição interna agrupada (primeira divisão de StratifiedGroupKFold com 4 partes) escolhe a época pela perda de grupos equilibrada por classe. Os limiares são ajustados exclusivamente nessa validação interna, com tetos empíricos de 0,75% ruído → Aedes e 1,75% outro → Aedes. Depois o modelo e o normalizador são reajustados em todo o treino externo pelo número de épocas escolhido. Os testes externos são aplicados após congelar os arquivos.

A seleção interna das redes é um holdout agrupado, enquanto a referência clássica usa calibração interna OOF mais ampla. A população e as cinco dobras externas são idênticas, mas os processos internos de seleção não são idênticos. Os limites internos de FPR não garantem limites populacionais.

A agregação por grupo pondera cada janela por (1-p_ruído)^1,5. Foram salvas as probabilidades OOF, a decisão argmax e a decisão calibrada. As métricas principais são determinísticas. A inferência final também implementa 100 passagens de Monte Carlo dropout; a discordância de decisões é um indicador de estabilidade da rede, sem garantia de correção. O exemplo de áudio salvo é apenas uma verificação de execução, não um teste independente.

O modelo neural final foi escolhido pela validação interna: LSTM, época 4; Bi-LSTM selecionou época 2. Os dois candidatos e todas as dobras foram preservados. A LSTM final tem 231.811 parâmetros e a Bi-LSTM 594.691.

## Verificações

O verificador reproduziu todas as 42.029 probabilidades de cada arquitetura nas cinco dobras, com diferença máxima zero; confirmou 24 normalizadores calculados apenas no treino, hashes dos pesos/fontes, partições e a seleção sem uso do teste externo. A suíte completa passou: 313 testes. A inferência de um WAV real e as 100 passagens de dropout também foram executadas.

Comandos de reprodução:

```bash
.venv/bin/python train_ternary_lstm.py
.venv/bin/python verify_ternary_lstm.py results/ternary_lstm_2026_10_15
.venv/bin/python report_ternary_lstm.py
.venv/bin/python -m pytest -q
```

O treinador recusa sobrescrever uma execução existente. Para refazer, escolha um diretório novo com --output. Evidências em results/ternary_lstm_2026_10_15: protocol.json, metrics.json, comparison.csv, verification.json, freeze.json, pesos, históricos, normalizadores e previsões de cada dobra.

## Discussão final

O resultado positivo da cascata é mantido e atribuído corretamente. O aumento de sensibilidade das redes diretas revela um compromisso com os falsos positivos; esta rodada não demonstrou vantagem de acurácia geral das redes. A avaliação é retrospectiva em um corpus conhecido, sem ablação causal de cada técnica. Novos indivíduos, sessões, microfones e ambientes continuam sendo o próximo teste de generalização. O objetivo apresentado é detecção ternária, sem inferência de dengue e sem afirmar execução embarcada.

Referência: Torres et al. (2025), https://latamt.ieeer9.org/index.php/transactions/article/view/9221 ; implementação pública https://github.com/NakanoMariko/Dengue-Infected-Mosquito-Detection .
