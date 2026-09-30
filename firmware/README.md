# Classificação acústica para Arduino

Para **identificar espécies no Nano 33 BLE Sense**, use a versão
[MosquitoSpecies](MosquitoSpecies/README.md). Ela inclui o modelo treinado para
20 espécies, captura PDM, confirmação, saída incerta e instruções de instalação.
Baixe a [pasta completa em ZIP](../output/arduino/MosquitoSpecies.zip).
Extraia e siga o [guia rápido](LEIA_PRIMEIRO.md): no Linux/macOS com Python,
`bash flash_arduino.sh --monitor` prepara as ferramentas e grava. No Windows,
use o atalho incluído ou a Arduino IDE. O pacote dispensa o dataset e o notebook.

O modelo exportado acertou 42,2% quando forçado a escolher em todos os trechos de
teste. Com rejeição e confirmação, acertou 87,9% entre 231 identificações emitidas,
mas cobriu só 3,9% dos 5.983 trechos positivos contíguos. Não houve teste físico.
Veja o manual dessa versão para interpretar acurácia, cobertura e limites.

## Versão anterior: detector de presença

O protótipo detecta **candidatos acústicos de presença de mosquito**. Não conta
indivíduos nem reconhece a espécie. A versão foi compilada para **Arduino Nano 33
BLE Sense / Sense Rev2**, usando o microfone PDM integrado. O Uno R3/Nano clássico
não comporta este firmware: o resultado da compilação foi 100.832 bytes de programa
e 59.584 bytes de variáveis globais, contra 32 KB de flash e 2 KB de SRAM do Uno R3.

Documentação oficial: [Nano 33 BLE Sense Rev2](https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2),
[microfone PDM](https://docs.arduino.cc/tutorials/nano-33-ble-sense-rev2/microphone-sensor/),
[Uno R3](https://docs.arduino.cc/hardware/uno-rev3).

> 📘 **Guia Completo para Montagem e Instalação:**
> Consulte o [Manual de Montagem e Instalação](MANUAL_DE_MONTAGEM_E_INSTALACAO.md) para detalhes de seleção de componentes, diagrama do microfone PDM, alimentação, permissões no Linux/Windows, pinout e resolução de problemas.

## Instalar

1. Instale **Arduino Mbed OS Nano Boards** no Gerenciador de Placas da Arduino IDE.
2. Abra `MosquitoPresence/MosquitoPresence.ino`. Mantenha os arquivos `.h` nessa pasta.
3. Selecione **Arduino Nano 33 BLE**, a porta da placa e faça o upload.
4. Abra o monitor serial em **115200 baud**. O LED acende após confirmação em duas
   de três janelas completas. O aparelho funciona sem o monitor conectado.

A primeira confirmação precisa de **2,976 s de áudio** (três janelas completas).
Esse é o tempo de observação escolhido, sem incluir o tempo computacional ainda
não medido na placa. Passagens muito curtas podem ser perdidas; não foi validada
a detecção de cada voo isolado.

Por linha de comando:

```bash
arduino-cli core update-index
arduino-cli core install arduino:mbed_nano@4.6.0
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble firmware/MosquitoPresence
arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/ttyACM0 firmware/MosquitoPresence
```

A porta acima é um exemplo; use a porta da sua placa. Compilação verificada com
CLI 1.5.1/core 4.6.0. **Não houve upload em uma placa física, teste do microfone,
medição de latência, autonomia ou pico de RAM**. O resultado de RAM do compilador
é estático e não inclui o pico de pilha/heap. Atingir os prazos de áudio deve ser
verificado no monitor: perdas de amostras geram aviso e reiniciam a confirmação.

## O que o código processa

- Áudio mono a 16 kHz, PCM de 16 bits.
- Frames Hann de 512 amostras sem sobreposição. Cada decisão usa 31 frames, **0,992 s**.
- 68 características: média/desvio da energia relativa em 31 bandas, flatness
  espectral e pico de 219-875 Hz. Remove DC por frame; razões de energia dispensam
  normalização usando um segundo completo. Não é um algoritmo de remoção de ruído.
- Rede neural com 16 unidades ReLU e uma saída, **1.121 parâmetros**. Modelo e
  constantes de calibração/limiares: aproximadamente **4.504 bytes numéricos**.
- Frontend sem alocação dinâmica: objeto de **6.420 bytes** no teste nativo, mais
  fila PCM de 8.192 bytes e outros componentes incluídos na compilação completa.
- Confirmação causal de duas em três janelas. Reinicia em perda de amostras,
  silêncio quase digital ou clipping excessivo. As regras de qualidade são
  salvaguardas de captura, ainda sem avaliação de campo.

## Resultados: o aparelho ainda erraria bastante

As três dobras reservam grupos inteiros e mantêm treino, seleção interna,
calibração e teste separados. Os rótulos de presença vêm **do arquivo**, sem
anotação de cada voo. São 23.877 trechos de arquivos com mosquito e 528 de ruído,
em 569 e 44 fontes, respectivamente. Estes números não medem eventos individuais.

| Modo | Recall de trechos positivos | Falsos positivos em trechos de ruído | Recall médio por fonte | Falsos positivos médios por fonte |
|---|---:|---:|---:|---:|
| Balanceado, limiar 0,5 | 82,7% | 35,2% | 81,0% | 21,8% |
| Meta de 5% na calibração | 48,9% | 20,3% | 48,3% | 9,2% |
| Meta de 10% na calibração, padrão | 72,6% | 29,2% | 70,9% | 15,9% |

Metas de calibração **não foram garantidas no teste**. A distribuição de fontes
de ruído muda; poucas fontes longas afetam bastante as taxas por janela. O modo
padrão deixa passar cerca de **27,4%** dos trechos positivos e marca **29,2%** dos
trechos de ruído como candidatos, antes da confirmação. A taxa por fonte dá peso
igual a cada gravação e responde a outra média.

Para trocar o modo, edite `detectionThreshold` no sketch para `kBalancedThreshold`,
`kConservativeThreshold` ou `kModerateThreshold`. O escore usa calibração com peso
igual a fontes positivas/negativas; **não é a probabilidade de haver mosquito em
um ambiente real**, cuja prevalência é desconhecida.

A confirmação 2/3 foi verificada em C++ e Python. O teste no subconjunto contíguo
teve somente **duas fontes de ruído**, e não demonstrou supressão confiável de
sons persistentes. Não converta essas taxas em alarmes/hora. Faltam gravações
contínuas anotadas, mosquito ativo/inativo e ruídos locais para medir esse valor.

Com ruído real reservado adicionado a 0 dB, o recall do modo padrão foi **40,2%**
na sonda de 480 trechos balanceados por espécie, contra 70,2% sem ruído adicionado.

## Arquivos e validação

`PresenceModel.h` contém o modelo da **dobra 0**; ele não usa os grupos de teste
dessa dobra. `PresenceModelFold1.h` e `PresenceModelFold2.h` são alternativas para
reproduzir as outras avaliações. O benchmark agregado usa o modelo correspondente
a cada dobra; não é uma avaliação das três dobras com o mesmo modelo da dobra 0.
O modelo exportado é um protótipo, sem revalidação para produção.

O processamento completo PCM -> características -> escore -> limiar coincidiu
em **532/532 janelas de teste da dobra 0** entre C++ e Python. Maior erro de
característica: 3,49e-5; de escore: 8,72e-6. Confirmação, reset por lacuna e
invariância a ganho/DC também passaram. Veja `results/arduino/native_audit.json`
e `results/arduino/compile_audit.json`.

Para reproduzir a partir do corpus preparado pelo notebook:

```bash
python train_arduino.py
python verify_arduino.py
python probe_arduino_noise.py
```

## Como melhorar na placa escolhida

Grave áudio com o próprio Arduino no local de uso. Reserve dias/locais inteiros
antes de ajustar modelos. Anote atividade real de mosquito, distância e condições
de captura; inclua horas sem mosquito com ventilador, fala, chuva, outros insetos
e máquinas. Treine com ruídos locais, calibre novamente e compare **sensibilidade
por evento, falsos alarmes/hora e latência**. Essa coleta é necessária para saber
quanto um aparelho real errará. Não foi substituída pelos arquivos antigos de celular.

A análise de importância por permutação das famílias de características está em
`results/arduino/permutation_importance.csv`: mede sensibilidade preditiva em teste,
sem atribuição causal ou prova de que o modelo reconhece exclusivamente as asas.
