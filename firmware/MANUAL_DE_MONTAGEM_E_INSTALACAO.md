# Montagem e instalação no Arduino Nano 33 BLE Sense

O sketch [MosquitoSpecies](https://github.com/Lciarallo/mosquito-wingbeat/blob/main/firmware/MosquitoSpecies/README.md) captura o som pelo microfone
integrado e calcula uma candidata entre 20 espécies. Inclui detecção de presença,
identificação, rejeição e confirmação temporal. O detector anterior, somente de
presença, continua disponível em [MosquitoPresence](https://github.com/Lciarallo/mosquito-wingbeat/tree/main/firmware/MosquitoPresence).

**Instalação rápida:** baixe o ZIP, extraia e siga
[LEIA_PRIMEIRO.md](LEIA_PRIMEIRO.md). No Linux/macOS, basta
`bash flash_arduino.sh --monitor` na pasta extraída. No Windows, use o atalho
incluído ou a Arduino IDE. Não precisa baixar os áudios ou treinar os modelos.

## 1. Funcionamento e resultado esperado

- Áudio mono a 16 kHz, frames Hann/FFT de 512 amostras e 31 frames por janela:
  **0,992 s** por decisão, 68 características espectrais.
- Duas redes **float32**, sem quantização INT8: presença 68 -> 16 -> 1;
  espécies 68 -> 64 -> 20. Os pesos estão incluídos nos headers.
- Uma identificação exige presença, confiança da classe e concordância da mesma
  espécie em duas de três janelas; a janela atual também deve ser elegível.
- O monitor mostra candidata, identificação provisória, `INCERTO`, `SEM_EVIDENCIA`
  ou `AUDIO_INVALIDO`. O LED indica uma identificação emitida.
- A primeira emissão requer **2,976 s de observação**, além do processamento.
  Isso não garante identificação de cada voo ou supressão de ruídos persistentes.

No teste do modelo exportado: **42,2%** de acerto ao forçar uma escolha em todas
as janelas positivas. Com rejeição/confirmação: **87,9% entre 231 identificações**,
mas apenas **3,9% de cobertura** dos 5.983 endpoints positivos contíguos.
A maioria dos trechos não recebeu identificação. **Não houve teste em placa física.**
Detalhes por espécie, ruído e validação: [manual do sketch](https://github.com/Lciarallo/mosquito-wingbeat/blob/main/firmware/MosquitoSpecies/README.md).

## 2. Materiais

1. **Arduino Nano 33 BLE Sense ou Sense Rev2**, com microfone PDM integrado.
2. Cabo **Micro-B USB com dados**. Ambas as revisões oficiais usam Micro-B;
   confira o conector da sua placa. [Sense](https://docs.arduino.cc/resources/datasheets/ABX00031-datasheet.pdf),
   [Sense Rev2](https://docs.arduino.cc/resources/datasheets/ABX00069-datasheet.pdf).
3. Computador com Arduino IDE ou Arduino CLI. A conexão USB fornece alimentação.

Para essa montagem, o microfone e o LED já estão na placa: não são necessárias
conexões externas. Um power bank USB pode alimentar o protótipo; autonomia e
funcionamento contínuo precisam de medição.

O firmware de espécies compilou com **125.120 bytes de programa** e **59.664 bytes
globais**, excluindo pico de pilha/heap. Ele excede a memória do Uno R3/Nano
clássico; o alvo compilado é `arduino:mbed_nano:nano33ble`.

## 3. Montagem física

Posicione a placa com a abertura acústica desobstruída. Confira a localização
do microfone no desenho da revisão da placa; não cole fita ou cola no orifício.
Uma caixa, espuma de proteção ou base antivibração altera a captura e deve ser
avaliada depois da montagem, especialmente perto de motores e ventiladores.

```text
Computador ou alimentação USB
            |
     Nano 33 BLE Sense
     microfone PDM integrado -> FFT -> modelos -> LED / serial
```

As GPIOs são de **3,3 V**, sem tolerância a sinais de 5 V. Para a montagem inicial,
use alimentação USB. Para alimentação por pinos, confira o pinout da revisão:
o pinout da Rev2 especifica VIN de 5 a 21 V.
[Datasheet Rev2](https://docs.arduino.cc/resources/datasheets/ABX00069-datasheet.pdf),
[pinout Rev2](https://docs.arduino.cc/resources/pinouts/ABX00069-full-pinout.pdf).

O código usa o LED integrado. Buzzer ou relé exigem circuito de acionamento
compatível, proteção da carga e alteração do firmware; essa montagem não foi testada.

## 4. Preparar o computador

Para o instalador automático, use **Python 3.9 ou mais recente**, sem pacotes pip.
Na primeira preparação, tenha internet e cerca de **1 GB livre**. Ele instala
Arduino CLI **1.5.1** e core **4.6.0** em `.arduino-tools`, dentro da pasta do
projeto/ZIP. Reutiliza versões compatíveis e cache nas próximas execuções.
Não substitui o core da IDE. Não execute o instalador com `sudo`.

Se preferir a [Arduino IDE](https://www.arduino.cc/en/software), no Gerenciador de
Placas instale **Arduino Mbed OS Nano Boards**, versão **4.6.0**.

No Linux, se houver erro de permissão na porta, confira seu proprietário/grupo
com `ls -l /dev/ttyACM*`. Use o grupo indicado pelo sistema; em distribuições que
usam `dialout`, o comando é `sudo usermod -aG dialout "$USER"`, seguido de novo
login. No Windows, selecione a porta COM reconhecida pela IDE.

## 5. Gravar o firmware

### Instalador automático: caminho recomendado no Linux

1. Baixe [MosquitoSpecies.zip](https://github.com/Lciarallo/mosquito-wingbeat/raw/refs/heads/main/output/arduino/MosquitoSpecies.zip), extraia tudo
   e conecte a placa por USB. Feche outros monitores seriais.
2. Abra o terminal na pasta `MosquitoSpecies` extraída e execute:

```bash
bash flash_arduino.sh --monitor
```

O mesmo comando funciona na raiz de um clone deste repositório. O script reutiliza
ou baixa a CLI oficial, confere seu SHA-256, prepara o core fixado, detecta a placa,
compila, faz upload e abre o monitor serial. Com várias Nanos conectadas, pede a
escolha; uma porta desconhecida também exige seleção explícita. Uma porta reconhecida
como outra placa é recusada. O alvo Nano 33 BLE é compartilhado pela variante sem
microfone: confirme que o seu hardware é **Sense/Sense Rev2**.

Saia do monitor com **Ctrl+C**. Também pode escolher a porta:

```bash
bash flash_arduino.sh --port /dev/ttyACM0 --monitor
```

No macOS, use uma porta da listagem, como `/dev/cu.usbmodem...`. No Windows x86/x64,
instale Python e abra **Instalar_no_Windows.cmd** dentro da pasta extraída; o atalho
sem argumentos prepara, grava e abre o monitor. Num terminal Windows:

```powershell
py -3 install_arduino.py --port COM3 --monitor
```

O caminho real de preparação/compilação foi verificado no Linux. Windows/macOS
não foram executados em sistemas nativos. Para Windows ARM, use a Arduino IDE.

Comandos auxiliares, sem upload:

```bash
bash flash_arduino.sh --compile-only
bash flash_arduino.sh --list-ports
bash flash_arduino.sh --monitor-only --port /dev/ttyACM0
bash flash_arduino.sh --help
```

O detector anterior continua acessível no repositório:

```bash
bash flash_arduino.sh /dev/ttyACM0 firmware/MosquitoPresence
```

Não é necessário preparar o ambiente científico Python. O instalador só usa a
biblioteca padrão. Guarde `.arduino-tools` para evitar novos downloads; os logs
`last_setup.log`, `last_core_install.log`, `last_compile.log` e `last_upload.log`
ajudam a localizar uma falha. O sucesso de upload só é exibido se a CLI terminar
sem erro. Este projeto verificou compilação e falhas simuladas, sem gravar placa física.

### Arduino IDE: alternativa sem Python

1. Baixe [MosquitoSpecies.zip](https://github.com/Lciarallo/mosquito-wingbeat/raw/refs/heads/main/output/arduino/MosquitoSpecies.zip) e extraia.
2. Abra `MosquitoSpecies/MosquitoSpecies.ino` mantendo juntos os headers
   `StreamingFeatures.h`, `PresenceModel.h`, `SpeciesModel.h` e `SpeciesDecision.h`.
   No repositório, a pasta é `firmware/MosquitoSpecies/`.
3. Instale **Arduino Mbed OS Nano Boards 4.6.0**, selecione a placa **Arduino Nano 33 BLE**
   e a porta da sua placa.
4. Faça upload e abra o monitor serial a **115200 baud**.

### Arduino CLI: uso manual/avançado

Na raiz do repositório, substitua a porta pelo caminho real:

```bash
arduino-cli core update-index
arduino-cli core install arduino:mbed_nano@4.6.0
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble firmware/MosquitoSpecies
arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/ttyACM0 firmware/MosquitoSpecies
arduino-cli monitor -p /dev/ttyACM0 -c baudrate=115200
```

Ao usar somente o ZIP, o caminho do sketch é a pasta extraída `MosquitoSpecies`.
As opções `--cli`, `--config-file` e `--tools-dir` do instalador permitem reutilizar
uma instalação existente no uso avançado. O padrão usa configuração própria;
não precisa dessas opções na instalação normal.

## 6. Conferir a captura e interpretar o serial

O sketch anuncia o modelo e os limiares, seguido deste cabeçalho CSV:

```text
tempo_ms,estado,candidata,identificada,escore_classe,limiar_classe,escore_presenca,audio_ok,amostras_perdidas,inferencia_us
```

`IDENTIFICACAO_PROVISORIA` preenche a espécie identificada. `INCERTO` pode mostrar
uma candidata, mas não emite identificação. `SEM_EVIDENCIA` também pode incluir
mosquitos perdidos pelo detector. Os escores são saídas dos modelos, **não
probabilidades garantidas de acerto num ambiente real**.

`AUDIO_INVALIDO` sinaliza silêncio quase digital ou clipping excessivo.
`# AUDIO_INTERROMPIDO` sinaliza amostras perdidas e reset do histórico.
`inferencia_us` mede o intervalo das redes/decisão, excluindo a FFT e o tempo de
observação; pode incluir interrupções. Essas medidas precisam ser coletadas na placa.

Uma gravação de referência reproduzida perto do microfone permite experimentar
a captura, mas não valida detecção de mosquito vivo, alcance ou acurácia em campo.
Ela pode resultar em `INCERTO` ou `SEM_EVIDENCIA`. Registre o resultado observado.

## 7. Limiares e calibração

No sketch de espécies, `presenceThreshold` usa `kModerateThreshold` do modelo da
dobra 2 (aproximadamente 0,671274); `speciesThreshold` usa o limiar de classe
calibrado (aproximadamente 0,48). A temperatura de softmax é 2,475964.

Alterar limiares modifica cobertura e erros. As taxas publicadas valem para os
valores fornecidos. Grave áudio anotado do Arduino no local de uso, inclua ruídos
locais e reserve dias/locais antes de ajustar e medir novos resultados.

Para o detector antigo, consulte [firmware/README.md](https://github.com/Lciarallo/mosquito-wingbeat/blob/main/firmware/README.md). Seus limiares
variam por dobra: no header padrão da dobra 0, o conservador é 1,0001 e rejeita
tudo; o moderado é aproximadamente 0,781769. As taxas agregadas de três dobras
não são taxas de campo nem usam um único limiar numérico para todas as dobras.

## 8. Resolver problemas

- **Porta desapareceu:** tente entrar no bootloader com dois toques rápidos em
  RESET, execute `bash flash_arduino.sh --list-ports` e informe a nova porta com
  `--port`. [Recuperação oficial](https://docs.arduino.cc/resources/datasheets/ABX00069-datasheet.pdf).
- **Porta ocupada:** feche o monitor da IDE e outros programas que usam a serial.
- **Erro de download:** confira internet/espaço livre e execute novamente. A
  instalação reutiliza o que já foi baixado e não executa uma CLI com SHA divergente.
- **Compilou, mas upload falhou:** consulte `last_upload.log`, confira porta,
  permissão e cabo de dados. O script não informa sucesso nem abre o monitor após falha.
- **Erro de microfone PDM:** confirme que é uma placa Sense com microfone integrado.
  O sketch mostra `ERRO: microfone PDM indisponivel.` e pisca o LED.
- **Amostras perdidas:** registre o contador e o estado da captura. O processamento
  reinicia para não confirmar usando áudio com lacunas.
- **Muitos resultados incertos:** é uma limitação esperada desse protótipo; não
  há garantia de identificação das 20 espécies. Confira suporte/cobertura por
  classe antes de interpretar uma taxa de acerto.
- **Identificações erradas ou ruído persistente:** grave os casos e faça avaliação
  com fontes novas. A confirmação temporal sozinha não demonstrou controle de
  alarmes contínuos de campo.
