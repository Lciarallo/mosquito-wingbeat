# Instalar no Arduino pelo Windows

**[⬇ Baixar o instalador para Windows](https://github.com/Lciarallo/mosquito-wingbeat/releases/latest/download/MosquitoWingbeat-Windows.exe)**

O download é público. Não precisa de conta no GitHub, Python, Arduino IDE,
terminal, dataset ou treinamento. O programa abre uma janela em português e
prepara as ferramentas de gravação automaticamente.

## O que você precisa

- Windows 10 ou 11 de **64 bits**, em computador Intel/AMD.
- **Arduino Nano 33 BLE Sense ou Sense Rev2**. O microfone já está na placa.
- Cabo Micro-B USB que transmita **dados**; um cabo que só carrega não serve.
- Internet no primeiro uso e cerca de **1 GB livre** no computador.

## Instalar em cinco passos

1. Clique no botão de download acima. Abra **MosquitoWingbeat-Windows.exe** com
   dois cliques. Aguarde a janela abrir.
2. Conecte o Arduino ao computador pelo cabo USB.
3. Clique em **Buscar minha placa**. Na primeira vez, aguarde alguns minutos
   enquanto o programa prepara o computador. Quando encontrar uma única placa
   reconhecida, ele a seleciona. Se aparecer uma lista, escolha a sua placa.
4. Clique em **Instalar no Arduino**. Mantenha o cabo conectado até aparecer
   **Programa gravado**. Essa ação substitui o programa que estava no Arduino.
5. Os resultados abrem automaticamente na aba **Resultado**. Nas próximas vezes,
   basta buscar a placa e clicar em **Ver resultados**, sem gravar de novo.

O instalador é portátil: não pede administrador nem cria atalhos no sistema.
Guarde o `.exe` onde preferir. As ferramentas baixadas ficam na pasta do seu
usuário, em `%LOCALAPPDATA%\MosquitoWingbeat`, e são reutilizadas.

## Entender o resultado

| Texto na janela | Significado |
|---|---|
| **Possível Aedes aegypti**, ou outro nome | Identificação provisória. O modelo pode errar. |
| **Espécie incerta** | Não houve evidência suficiente para informar uma espécie. |
| **Sem evidência suficiente** | O detector não indicou presença nesse trecho; pode ter perdido um mosquito. |
| **Áudio insuficiente** | O áudio ficou baixo demais ou saturado. Confira a posição do microfone. |

A primeira leitura confirmada precisa de aproximadamente **3 segundos de som**,
além do processamento. Não há distância de detecção garantida. O modelo é
experimental e costuma deixar a espécie incerta. A acurácia no Arduino em uso
real ainda não foi medida. Consulte os
[resultados e limites](firmware/MosquitoSpecies/README.md#acurácia-medida-e-rejeição).

## Se algo não funcionar

- **A placa não apareceu:** confira o cabo de dados e outra porta USB. Feche o
  monitor serial da Arduino IDE. Clique novamente em **Buscar minha placa**.
- **A gravação falhou:** toque duas vezes rapidamente em **RESET** na placa,
  clique em **Buscar minha placa** e repita **Instalar no Arduino**.
- **Há várias portas:** desconecte as outras placas e busque novamente. A versão
  Nano 33 BLE sem “Sense” não tem o microfone necessário.
- **Sem resultados:** feche outros programas que estejam usando a placa e clique
  em **Ver resultados**. Confira o estado na aba **Detalhes**.
- **Falha no download:** confira internet e espaço livre; repita a busca. O
  programa aproveita os arquivos já baixados.
- **Aviso do Windows:** este `.exe` ainda não tem assinatura digital. Confira
  que você baixou da página **Lciarallo/mosquito-wingbeat** no GitHub. Se o Windows
  mostrar “Windows protegeu o computador”, use **Mais informações → Executar
  assim mesmo** somente se confiar nessa origem. Não desative o antivírus. Em
  computadores de empresa, peça ao responsável para liberar o aplicativo.

## Download e verificação

A [página de downloads](https://github.com/Lciarallo/mosquito-wingbeat/releases/latest)
também oferece o guia em `.txt`, hashes SHA-256 e auditorias da versão. O `.exe`
foi compilado e executado em Windows no GitHub Actions; o diagnóstico abriu a
janela e compilou o firmware incluído, **sem gravar uma placa física**.

Para Linux/macOS, uso avançado ou instalação pela Arduino IDE, consulte o
[guia alternativo](firmware/LEIA_PRIMEIRO.md).
