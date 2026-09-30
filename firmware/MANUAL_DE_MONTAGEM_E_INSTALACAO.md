# Manual de Montagem e Instalação: Detector Acústico de Mosquitos no Arduino

Este manual fornece o guia passo a passo completo para selecionar os componentes, realizar a montagem física, configurar o ambiente de desenvolvimento e gravar o firmware de detecção de presença de mosquitos em tempo real (**TinyML**) no microcontrolador.

---

## 1. Visão Geral do Sistema

O firmware contido na pasta [`firmware/MosquitoPresence/`](MosquitoPresence/) transforma a placa em um detector acústico inteligente autônomo:
- **Taxa de amostragem:** 16.000 Hz (mono, PCM 16-bit).
- **Processamento:** Janelas de 0,992 segundos (31 quadros de 512 amostras com janela Hann).
- **Extração de características:** 68 variáveis espectrais e energéticas (picos de 219 a 875 Hz onde ocorrem as frequências fundamentais de batimento de asa, *flatness* espectral e distribuição de energia em 31 bandas).
- **Classificador embarcado:** Rede neural densa quantizada (16 neurônios com ativação ReLU, ~1.121 parâmetros).
- **Filtro de confirmação temporal (2/3):** Exige confirmação em pelo menos 2 de 3 janelas consecutivas (~2,97 segundos) para acionar o alerta, mitigando ruídos impulsivos (palmas, estalos, batidas de porta).
- **Aviso:** Acionamento do LED onboard (`LED_BUILTIN`) e transmissão de telemetria via porta serial (115200 baud).

---

## 2. Lista de Materiais (Hardware BOM)

### Componentes Obrigatórios
1. **Placa de Desenvolvimento:** **Arduino Nano 33 BLE Sense** ou **Arduino Nano 33 BLE Sense Rev2**.
   - *Microcontrolador:* Nordic nRF52840 (ARM Cortex-M4F @ 64 MHz, 1 MB Flash, 256 KB SRAM).
   - *Microfone integrado:* Sensor digital PDM omnidirecional onboard (ST MP34DT05 ou MP34DT06J).
2. **Cabo USB:** Cabo Micro-USB (Rev1) ou Type-C (Rev2/compatíveis) de boa qualidade com **linhas de dados ativas** (cabos que servem apenas para carregamento de celular não serão reconhecidos pelo computador).
3. **Fonte de Alimentação para Campo:**
   - Para uso conectado ao computador: a própria porta USB.
   - Para operação autônoma no ambiente: *Power bank* USB (5V) comum ou carregador de celular de 5V conectado à porta USB da placa.

> [!CAUTION]
> **Compatibilidade de Hardware:**
> O código **NÃO funciona** em placas tradicionais de 8 bits como **Arduino Uno R3**, **Arduino Nano clássico (ATmega328P)** ou **Arduino Mega 2560**.
> - Este firmware exige cerca de **100 KB de Flash** e **60 KB de memória RAM estática**.
> - O Arduino Uno possui apenas 32 KB de Flash e 2 KB de RAM, sendo insuficiente para o buffer circular de áudio, FFT e inferência da rede neural.

---

## 3. Montagem Física e Cuidados com o Microfone

```
                      +-------------------+
                      |   [Micro-USB]     |
                      |                   |
                      |  [RESET] [LED]    |  <-- LED_BUILTIN (Alerta)
                      |                   |
                      |   (o) MICROFONE   |  <-- Orifício acústico PDM
                      |       PDM         |      (NÃO OBSTRUIR!)
                      |                   |
                      |   [nRF52840]      |
                      |   ARM Cortex-M4   |
                      |                   |
                      +-------------------+
```

### 3.1. Localização e Proteção do Microfone PDM
Na face superior da placa existe um pequeno componente metálico retangular prateado/dourado com um **minúsculo orifício circular**. Este é o microfone digital PDM (frequência de resposta de até 80 kHz).

- **Abertura Acústica Livre:** Nunca cole fitas adesivas, etiquetas ou cola quente diretamente sobre esse orifício.
- **Instalação em Caixas / Cases:**
  - Se for utilizar um case plástico comercial ou impresso em 3D, faça uma perfuração de pelo menos **2 mm a 3 mm de diâmetro** na carcaça, alinhada exatamente à posição do orifício do microfone.
  - Para proteção contra poeira e vento, utilize uma pequena camada de espuma acústica de baixa densidade (como a de microfones comuns) sobre a abertura externa.
- **Isolamento de Vibração Mecânica:**
  - O bater de asas do mosquito produz vibrações de baixa a média frequência (300 Hz a 900 Hz).
  - Se a placa for montada perto de ventiladores, motores ou armadilhas com ventoinha, monte o Arduino sobre uma base de borracha ou espuma para evitar que o ruído mecânico da estrutura se propague para o sensor.

### 3.2. Níveis de Tensão e Cuidados Elétricos
- **Tensão de Operação:** O nRF52840 opera internamente em **3.3V**.
- Todas as portas GPIO digitais e analógicas aceitam no máximo **3.3V**. **Nunca conecte sensores ou sinais de 5V diretamente aos pinos**, sob risco de queimar o microcontrolador.
- Para alimentar a placa sem USB:
  - Pino **VIN**: Aceita tensões reguladas de **5V a 18V** (passa pelo regulador interno da placa).
  - Pino **3.3V**: Saída regulada de 3.3V (para sensores externos de baixa corrente).

### 3.3. Conexão de Alertas Externos (Buzzer / Relé) — Opcional
Caso queira acionar um alarme sonoro alto ou um relé de armadilha além do LED interno:

```
 Pino D2 (Arduino) --------[ Resistor 1k ]------ Base (Transistor NPN ex: BC547)
                                                  |
 Pino GND (Arduino) --------------------------- Emissor
                                                  |
 +5V / +3.3V ------------[ Buzzer Ativo (+) ]-- Coletor
```
*(Se for utilizar apenas o LED integrado onboard, nenhuma ligação externa é necessária)*.

---

## 4. Configuração do Ambiente no Computador

### No Linux (Ubuntu, Debian, Fedora, Arch)
Por padrão, portas seriais USB exigem permissão de grupo:
1. Abra o terminal e adicione o seu usuário ao grupo `dialout` (ou `uucp`):
   ```bash
   sudo usermod -aG dialout $USER
   ```
2. Faça logout e login novamente (ou reinicie o computador) para aplicar a permissão.

### No Windows
Ao instalar a Arduino IDE, os drivers de porta serial virtual (CDC) para a família Nano 33 BLE são instalados automaticamente.

---

## 5. Gravação do Firmware na Placa

### Método A: Utilizando a Arduino IDE 2.x (Interface Gráfica)

1. **Instalar a Arduino IDE:** Baixe e instale a versão oficial 2.x em [arduino.cc/en/software](https://www.arduino.cc/en/software).
2. **Instalar o Pacote de Placas Nano 33 BLE:**
   - No menu lateral esquerdo da IDE, clique no ícone do **Boards Manager** (segundo ícone).
   - Digite `mbed nano` na barra de busca.
   - Localize o pacote **Arduino Mbed OS Nano Boards** e clique em **Install**.
   - *(Aguarde o download e a instalação dos compiladores ARM)*.
3. **Abrir o Projeto:**
   - Na Arduino IDE, clique em `Arquivo` > `Abrir...`.
   - Navegue até a pasta do repositório:
     `firmware/MosquitoPresence/MosquitoPresence.ino`
   - Observe que a IDE abrirá automaticamente as abas com os arquivos de cabeçalho:
     - `MosquitoPresence.ino` (código principal)
     - `StreamingFeatures.h` (frontend extrator FFT)
     - `PresenceModel.h` (pesos e arquitetura da rede neural)
4. **Conectar e Selecionar a Placa:**
   - Plugue o Arduino Nano 33 BLE Sense na porta USB do computador.
   - Na barra superior da IDE, clique no menu suspenso de placas e escolha:
     **Arduino Nano 33 BLE** (ou *Arduino Nano 33 BLE Sense*).
   - Selecione a porta serial identificada (ex: `/dev/ttyACM0` no Linux, `COM3` no Windows).
5. **Gravar (Upload):**
   - Clique no botão de **Upload** (ícone de seta redonda para a direita `➔`).
   - A IDE compilará o código e fará o upload para a memória Flash.
   - Na janela de console inferior deverá surgir:
     `CPU: ... Uploading ... Done.`

---

### Método B: Utilizando a Linha de Comando (`arduino-cli`)

Para ambientes automatizados ou servidores Linux, você pode usar o `arduino-cli`:

1. Instale o pacote de placas:
   ```bash
   arduino-cli core update-index
   arduino-cli core install arduino:mbed_nano
   ```
2. Compile o sketch a partir da raiz do repositório:
   ```bash
   arduino-cli compile --fqbn arduino:mbed_nano:nano33ble firmware/MosquitoPresence
   ```
3. Grave na placa conectada (substitua `/dev/ttyACM0` pela sua porta):
   ```bash
   arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/ttyACM0 firmware/MosquitoPresence
   ```

---

## 6. Validação e Teste em Bancada

1. **Abrir o Monitor Serial:**
   - Na Arduino IDE, abra o **Serial Monitor** (`Ctrl + Shift + M`).
   - Configure a velocidade para **115200 baud**.
2. **Mensagem de Inicialização:**
   Ao ligar, a placa emitirá:
   ```text
   Detector experimental: 16 kHz; 0,992 s; confirmacao 2/3.
   ```
3. **Leitura dos Dados em Tempo Real:**
   A cada ~1 segundo, uma nova linha de telemetria é impressa:
   ```text
   escore=0.1245 limiar=0.4500 audio_ok=1 candidato_persistente=0
   escore=0.6830 limiar=0.4500 audio_ok=1 candidato_persistente=0
   escore=0.7412 limiar=0.4500 audio_ok=1 candidato_persistente=1
   ```
   - **`escore`:** Probabilidade calculada pela rede neural de que o trecho contenha um mosquito (entre 0.0 e 1.0).
   - **`limiar`:** Valor mínimo de corte para considerar o trecho positivo.
   - **`audio_ok`:** Indica se o áudio não sofreu clipping excessivo ou silêncio digital absoluto.
   - **`candidato_persistente`:** `1` quando confirmado em 2 de 3 janelas sucessivas. Quando igual a 1, o **LED onboard laranja/amarelo acende**.

4. **Teste de Verificação Acústica:**
   - Reproduza em um smartphone ou computador um áudio de referência de zumbido de mosquito (por exemplo, buscando por *"mosquito wingbeat sound 400Hz 600Hz"* ou usando gravações de teste).
   - Aproxime o alto-falante a cerca de 10 a 20 cm do microfone da placa em volume moderado.
   - Observe no Serial Monitor o aumento do `escore` e o acendimento do LED da placa.

---

## 7. Ajuste de Sensibilidade (Calibração)

No arquivo [`firmware/MosquitoPresence/MosquitoPresence.ino`](MosquitoPresence/MosquitoPresence.ino#L16), a linha:

```cpp
const float detectionThreshold = mosquito::kModerateThreshold;
```

permite escolher entre três perfis de detecção de acordo com o ambiente:

| Perfil | Limiar | Recall de Mosquitos | Falso Alarme em Ruído | Recomendação de Uso |
| :--- | :---: | :---: | :---: | :--- |
| **`kModerateThreshold`** *(Padrão)* | `~0,45` | **72,6%** | **15,9%** por fonte | Ambientes residenciais internos comuns. |
| **`kConservativeThreshold`** | `~0,65` | **48,9%** | **9,2%** por fonte | Ambientes ruidosos com ventiladores ou vozes frequentes. |
| **`kBalancedThreshold`** | `0,50` | **82,7%** | **21,8%** por fonte | Armadilhas fechadas com isolamento sonoro ou alta sensibilidade necessária. |

---

## 8. Resolução de Problemas (Troubleshooting)

- **A porta serial da placa desapareceu ou o upload falhou:**
  - *Solução do Duplo Clique:* Pressione o botão físico **RESET** da placa duas vezes rapidamente (em menos de 1 segundo). O LED da placa começará a pulsar suavemente em cor laranja/verde, indicando que ela entrou no **Modo Bootloader de Recuperação**. Em seguida, selecione novamente a porta na IDE e faça o upload.
- **Mensagem: "Falha ao iniciar microfone PDM":**
  - O LED piscará em intervalos de 500 ms de forma contínua.
  - Verifique se a placa é de fato o modelo **Nano 33 BLE Sense** (a versão simples "Nano 33 BLE" comum não possui microfone PDM integrado).
- **Aviso no monitor: "Audio interrompido; amostras perdidas":**
  - Ocorre caso haja travamentos no barramento serial ou atrasos de processamento. O algoritmo limpa o buffer e reinicia o contador para evitar decisões com dados corrompidos.
- **Falsos positivos frequentes:**
  - Verifique se a placa está recebendo zumbido elétrico de fontes de alimentação de baixa qualidade ou vibração de mesa. Troque para o perfil `kConservativeThreshold`.
