# Instalar o detector de espécies

Use **Arduino Nano 33 BLE Sense ou Sense Rev2** e cabo Micro-B USB com dados.
O microfone já está na placa. O modelo é experimental e frequentemente retorna
`INCERTO`; instalação bem-sucedida não comprova acurácia em campo.

**Windows: [baixe o aplicativo em português](https://github.com/Lciarallo/mosquito-wingbeat/releases/latest/download/MosquitoWingbeat-Windows.exe).**
Abra com dois cliques, conecte a placa, clique em **Buscar minha placa** e depois
em **Instalar no Arduino**. Os resultados aparecem na janela. Não precisa de
Python, Arduino IDE, terminal ou conta no GitHub.
[Passo a passo para Windows](https://github.com/Lciarallo/mosquito-wingbeat/blob/main/WINDOWS.md).

Para Linux/macOS ou Arduino IDE:
Baixe o [ZIP pronto para instalar](https://github.com/Lciarallo/mosquito-wingbeat/raw/refs/heads/main/output/arduino/MosquitoSpecies.zip).
**Extraia tudo** e entre na pasta `MosquitoSpecies`.
Não é necessário baixar o dataset, executar o notebook ou treinar modelos.

## Linux ou macOS: um comando

Com Python 3.9 ou mais recente instalado, conecte a placa, feche outros monitores
seriais, abra um terminal dentro da pasta extraída e execute:

```bash
bash flash_arduino.sh --monitor
```

O instalador baixa a CLI 1.5.1 com SHA-256 conferido, prepara o core 4.6.0, detecta
a placa, compila, grava e abre o serial a **115200 baud**. Se detectar várias
placas ou uma porta desconhecida, pede a escolha. Saia do monitor com **Ctrl+C**.
A primeira preparação precisa de internet, pode demorar e requer cerca de 1 GB
livre. As ferramentas ficam em `.arduino-tools`; as próximas execuções reutilizam
essa pasta. Não execute com `sudo`.

Para informar a porta, se necessário:

```bash
bash flash_arduino.sh --port /dev/ttyACM0 --monitor
```

No macOS, use a porta mostrada na listagem, por exemplo `/dev/cu.usbmodem...`.

## Windows

O **MosquitoWingbeat-Windows.exe** é o caminho recomendado para Windows 10/11
Intel/AMD de 64 bits. Ele prepara as ferramentas na pasta do seu usuário,
`%LOCALAPPDATA%\MosquitoWingbeat`, sem administrador. No primeiro uso, precisa de
internet e cerca de 1 GB livre. Há instruções para avisos do Windows e problemas
de cabo/porta no [guia ilustrado](https://github.com/Lciarallo/mosquito-wingbeat/blob/main/WINDOWS.md).

O `.exe` foi executado em Windows no GitHub Actions e compilou o firmware incluído.
Isso não substitui um teste de gravação/microfone na placa física.

O atalho **Instalar_no_Windows.cmd** no ZIP abre o aplicativo se ele estiver na
mesma pasta, ou abre seu download no navegador. Com argumentos, é uma alternativa
avançada que exige Python 3.9+ e launcher `py` ou Python no PATH. Por terminal:

```powershell
py -3 install_arduino.py --port COM3 --monitor
```

O instalador por terminal tem downloads para Windows x86/x64; para Windows ARM,
use a IDE. O caminho macOS não foi executado em sistema nativo neste projeto.

## Pela Arduino IDE: alternativa sem Python

1. Instale a [Arduino IDE](https://www.arduino.cc/en/software).
2. No Gerenciador de Placas, instale **Arduino Mbed OS Nano Boards**, versão **4.6.0**.
3. Abra `MosquitoSpecies.ino`, mantendo os quatro headers `.h` juntos.
4. Escolha **Arduino Nano 33 BLE** e a porta da sua Sense/Sense Rev2.
5. Clique em **Upload** e abra o monitor serial em **115200 baud**.

Não precisa instalar bibliotecas de machine learning; `PDM.h` vem no core.

## Conferir ou resolver problemas

| Comando na pasta extraída | Ação |
|---|---|
| `bash flash_arduino.sh --compile-only` | Prepara e compila sem placa, sem upload. |
| `bash flash_arduino.sh --list-ports` | Prepara ferramentas e mostra as portas, sem upload. |
| `bash flash_arduino.sh --monitor-only` | Abre o serial sem gravar novamente. |
| `bash flash_arduino.sh --help` | Mostra as opções. |

No uso avançado por terminal no Windows, substitua `bash flash_arduino.sh` por
`py -3 install_arduino.py`. No aplicativo, use os botões e a aba **Detalhes**.
Nenhuma placa detectada: confira o cabo de dados, feche o monitor da IDE e, se
precisar, toque duas vezes rapidamente em **RESET**, liste e selecione a nova porta.
Permissão no Linux: o erro informa o grupo da porta e o comando aplicável; depois
de entrar no grupo, saia e entre na sessão. Upload interrompido: conecte novamente
e repita. Logs ficam em `.arduino-tools/last_*.log`.

Após gravar, surgem linhas CSV a cada aproximadamente **0,992 s**. O LED indica
`IDENTIFICACAO_PROVISORIA`; `INCERTO`, `SEM_EVIDENCIA` e `AUDIO_INVALIDO` também são
resultados possíveis. Escores não são garantia de acerto. Não houve teste físico
do microfone ou upload em placa neste estudo. A explicação dos resultados está
no [manual do modelo](https://github.com/Lciarallo/mosquito-wingbeat/blob/main/firmware/MosquitoSpecies/README.md).

Manual completo de montagem e instalação:
[MANUAL_DE_MONTAGEM_E_INSTALACAO.md](https://github.com/Lciarallo/mosquito-wingbeat/blob/main/firmware/MANUAL_DE_MONTAGEM_E_INSTALACAO.md).
