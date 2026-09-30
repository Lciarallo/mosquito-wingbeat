# Identificar espécies no Arduino Nano 33 BLE Sense

Este sketch usa o **microfone PDM integrado** do Nano 33 BLE Sense ou Sense Rev2
para calcular uma espécie candidata entre as 20 classes treinadas. Inclui captura,
FFT incremental, duas redes neurais e confirmação. O monitor serial distingue
uma candidata de uma identificação provisória. É um protótipo experimental.

## Instalar na placa

1. Baixe [MosquitoSpecies.zip](../../output/arduino/MosquitoSpecies.zip) e extraia.
2. Na Arduino IDE, instale **Arduino Mbed OS Nano Boards** pelo Gerenciador de Placas.
3. Abra `MosquitoSpecies/MosquitoSpecies.ino`. Mantenha **todos os `.h` na mesma pasta**.
4. Selecione a placa **Arduino Nano 33 BLE**, escolha a porta e faça o upload.
5. Abra o monitor serial em **115200 baud**. A placa também processa sem o monitor aberto.

O core fornece `PDM.h`; as redes e o processamento estão nos headers incluídos.
Não precisa de Edge Impulse, TensorFlow, microfone externo ou conexão Bluetooth.

Com Arduino CLI:

```bash
arduino-cli core update-index
arduino-cli core install arduino:mbed_nano@4.6.0
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble firmware/MosquitoSpecies
arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/ttyACM0 firmware/MosquitoSpecies
```

A porta é um exemplo. Ao usar o ZIP fora do repositório, substitua o caminho por
`MosquitoSpecies`. Foi compilado com CLI 1.5.1 e core 4.6.0 para esse alvo.

Referências oficiais: [placa Sense Rev2](https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2),
[microfone e PDM](https://docs.arduino.cc/tutorials/nano-33-ble-sense-rev2/microphone-sensor/).

## Ler o resultado

Cada linha CSV contém tempo, estado, candidata, identificada, escore da classe,
limiar da classe, escore de presença, qualidade de áudio, amostras perdidas e
tempo de inferência em microssegundos. Esse último mede o intervalo das redes e
da decisão, podendo incluir interrupções; exclui a FFT e o tempo de observação.

| Estado | Significado |
|---|---|
| `IDENTIFICACAO_PROVISORIA` | Presença e classe passam pelos limiares, e a classe atual concorda em 2/3 janelas. O campo `identificada` contém o nome. |
| `INCERTO` | Há candidata, mas o limiar da classe ou a concordância temporal não foi atingido. `identificada` é `-`. |
| `SEM_EVIDENCIA` | O escore de presença não passou. Isso pode incluir mosquitos que o detector perdeu. |
| `AUDIO_INVALIDO` | Silêncio quase digital ou clipping excessivo; histórico reiniciado. |

O LED acende quando o código emite identificação provisória. Avisos iniciados
por `# AUDIO_INTERROMPIDO` mostram perda de amostras e reset do processamento.
Uma janela usa 31 frames de 512 amostras a 16 kHz: **0,992 s**. A primeira
confirmação exige três janelas completas, **2,976 s de observação**, além do tempo
computacional. Sons curtos podem ser perdidos. A janela atual deve ser elegível;
uma janela incerta não emite novamente um nome antigo.

`escore_classe` e `escore_presenca` são saídas calibradas sob distribuições do
corpus. **Não representam uma garantia de acerto nem a prevalência do ambiente.**
Uma espécie fora das classes treinadas pode receber um nome conhecido.

## Quais espécies

| Gênero | Espécies presentes no treinamento |
|---|---|
| Aedes | aegypti, albopictus, mediovittatus, sierrensis |
| Anopheles | albimanus, arabiensis, atroparvus, dirus, farauti, freeborni, gambiae, merus, minimus, quadriannulatus, quadrimaculatus, stephensi |
| Culex | pipiens, quinquefasciatus, tarsalis |
| Culiseta | incidens |

Os arquivos possuem rótulos por gravação. Não há anotação de cada voo nem
identificação independente de cada mosquito individual dentro de um trecho.

## Acurácia medida e rejeição

O padrão exportado é a **dobra 2**, primeira que atingiu a meta de calibração;
a escolha não usa acurácia do teste. É a rede 68 -> 64 -> 20, com **5.716
parâmetros**, junto do detector 68 -> 16 -> 1. Temperatura de espécies: 2,475964;
limiar de classe: aproximadamente **0,48**. O limiar de presença é o moderado do
modelo da mesma dobra. Não houve retreino usando todos os dados.

| Avaliação do modelo exportado | Acerto | Cobertura positiva |
|---|---:|---:|
| Escolha forçada em todas as 7.845 janelas positivas | 42,2% geral; 54,0% balanceada | 100% |
| Presença + confiança, uma janela | 76,9% entre 585 identificações | 7,5% das 7.845 janelas |
| Uma janela nos endpoints contíguos | 77,5% entre 453 identificações | 7,6% de 5.983 endpoints |
| Rejeição + mesma espécie em 2/3, mesmos endpoints | 87,9% entre 231 identificações | 3,9% de 5.983 endpoints |

O acerto de 87,9% é **condicional aos poucos trechos aceitos**: a maioria fica
incerta ou sem evidência. Esses valores usam fontes de teste reservadas do corpus
de celulares, não gravações feitas na placa. Nos 56 trechos de ruído reservados
dessa dobra não houve identificação por uma janela; nenhum trecho negativo dessa
dobra foi elegível para confirmação contígua. Isso não demonstra zero alarmes
em campo ou estima alarmes por hora. Guardas de silêncio/clipping ainda não foram
avaliados com o microfone físico.

Entre as emissões confirmadas dessa dobra, o nome **Aedes aegypti** apareceu 22
vezes, com 21 corretas (95,5%); **Aedes albopictus**, cinco vezes, todas corretas.
São poucos trechos correlacionados, não mosquitos independentes ou garantia para
um novo som. **Culex quinquefasciatus não teve nenhuma emissão aceita**. Não há
acurácia condicional estimável para uma classe que nunca foi emitida. A tabela
com suporte e cobertura por espécie está em `selective_per_class.csv`.

Nas três dobras, a seleção interna teve **39,5% geral / 52,1% balanceada por
janela** e **69,8% macro recall por fonte agregada**. Esse último agrega vários
trechos de uma fonte e não descreve um som isolado. Dois modelos não atingiram a
meta de calibração e rejeitam todas as identificações. O benchmark agregado usa
o modelo de cada dobra, diferente da avaliação da dobra exportada acima.

A meta de calibração é acerto ponderado por fonte/espécie de 80%, cobertura
ponderada mínima de 10% e 15 fontes. Metas de calibração não se garantem no teste.
Uma espécie possui só três fontes e não permite validação interna independente.
O modelo compacto ficou abaixo dos classificadores maiores de computador.

## Verificações disponíveis

- Compilação: **125.120 bytes de programa** e **59.664 bytes globais**. RAM estática
  não inclui pico de pilha/heap. Placa alvo: `arduino:mbed_nano:nano33ble`.
- Todos os **7.901 trechos** reservados da dobra exportada: classe e decisão por
  limiar iguais entre C++ e Python, erro máximo de escore de classe 1,11e-6.
- Caminho completo PCM -> FFT -> duas redes -> limiares: **456/456 decisões iguais**,
  400 positivos e todos os 56 negativos. Erro máximo de escore de classe 3,04e-6.
- Confirmação: warmup, discordância entre espécies, janela atual incerta, reset
  e decisões em sequência do corpus verificados; checkpoint numérico restaurado.
- **Sem upload ou teste em placa física**. Faltam prazo de processamento, microfone,
  pico de RAM, distância, consumo, sensibilidade por voo e alarmes por hora.

Os arquivos `SpeciesModelFold0/1/2.h` e `PresenceModelFold0/1/2.h` permitem reproduzir
outras dobras: sempre substitua **ambos** os modelos pelo par correspondente. Os
headers padrão já contêm o par da dobra 2. A versão do ZIP contém apenas esse par.

## Reproduzir e melhorar

Prepare o corpus completo no notebook. Depois:

```bash
python train_arduino.py
python train_arduino_species.py
python verify_arduino_species.py
```

Resultados, matriz de confusão, recalls por espécie, calibração, grupos,
probabilidades e auditorias: [results/arduino_species](../../results/arduino_species).
O checkpoint `.npz` inclui todos os pesos, temperatura, limiares e calibração do
detector. Os arquivos `.joblib` ficam locais e são regenerados pelo treinamento.

Para aumentar utilidade no aparelho, são necessários áudios anotados do próprio
Arduino, mais fontes das espécies raras, ruídos locais e teste em dias/locais
reservados. Reduzir `speciesThreshold` ou `presenceThreshold` aumenta emissões e
pode aumentar erros; os resultados acima valem para os limiares fornecidos.
