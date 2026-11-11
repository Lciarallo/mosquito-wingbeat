# Apresentação: "Zzzz… é Aedes?"

Versão **para leigos** (23 slides): conta a história do projeto sem entrar em modelos, bibliotecas ou código. Pouco texto por slide, uma ideia por vez, e animações: o mosquito voando, o zumbido como onda (com botão para ouvir um zumbido sintético), o som virando imagem, os exemplos entrando na "máquina", números que contam e gráficos que crescem.

- `Apresentacao_Mosquito_Interpretabilidade.html`: versão animada e autocontida. Abra no navegador; setas ou clique para avançar, `f` para tela cheia. Nos slides 6, 11, 18 e 19 o conteúdo aparece **em passos**, um a cada clique (há um aviso "clique ▸" no canto).
- `Apresentacao_Mosquito_Interpretabilidade.pdf`: versão estática, com tudo já revelado.
- `videos/`: os vídeos usados nos slides (ver abaixo).
- `fotos/`: as fotos do laboratório usadas nos slides 14 a 16, já recortadas e comprimidas.

A versão anterior, técnica (30 slides, com modelos, métricas e código), continua no histórico do Git (commit `4cab181`).

**Todos os números dos slides saem do caderno** `notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb`, que é a única fonte da verdade do projeto. Se os dois divergirem, vale o caderno. Dois desenhos são ilustrações da ideia, e os slides dizem isso no rodapé: a "foto do som" pintada à esquerda no slide 8 e as barras do slide 18.

## Vídeos

Os quatro vídeos (`videos/`) vêm de Mukundarajan H., Hol F. J. H., Castillo E. A., Newby C., Prakash M. *Using mobile phones as acoustic sensors for high-throughput mosquito surveillance*. **eLife** 6:e27854 (2017), [doi:10.7554/eLife.27854](https://doi.org/10.7554/eLife.27854), licença **CC BY 4.0**. Foram recortados e comprimidos (vídeos 2, 3, 4 e 5 do artigo). Estão embutidos no HTML (WebM e MP4), que continua funcionando offline como arquivo único; nos slides, cada vídeo toca sozinho, sem som, quando o slide abre.

| Arquivo | Slide | Conteúdo |
|---|---|---|
| `asa_camera_lenta` | 4 | *Culex tarsalis* filmado a 10 kHz, mostrado 100× mais devagar |
| `gravando_garrafa`, `gravando_decolagem` | 7 | Gravação com celular: mosquito na garrafa e decolando |
| `espectrograma_ao_vivo` | 8 | Espectrograma de *Anopheles gambiae* sendo montado |


## Fotos e vídeo do laboratório

Os slides 14 a 16 usam dez fotos e um vídeo curto feitos no laboratório durante as gravações: a bancada, os potes da criação, o aspirador, os mosquitos adormecidos no frio, a armadilha (uma garrafa cortada com o Arduino dentro) e o computador gravando. Duas fotos mostram a data na tela do computador (05/11/2026 e 21/10/2026); o vídeo é de 05/11/2026. As fotos foram **recortadas para não mostrar o rosto de ninguém**, porque este repositório é público. O vídeo `videos/armadilha_arduino.mp4` está sem som.

## Como a apresentação se liga ao caderno

| Slides | Assunto | Seção do caderno |
|---|---|---|
| 5 e 6 | o zumbido é uma nota (cerca de 400 batidas por segundo); os tons das espécies se misturam | §3 |
| 8 | o som vira imagem (espectrograma) | §4 |
| 9 | aprender com exemplos: 2.167 Aedes, 3.050 outros mosquitos, 426 ruídos | §2 e §5 |
| 10 | prova com aparelho nunca ouvido | §7 |
| 11 | duas perguntas separadas: de cada 100 avisos, 76 eram Aedes; de cada 100 Aedes, ela avisou em 52 | §11 e §12 |
| 12 | em que faixas do som a máquina presta atenção | §15 |
| 14 a 16 | fotos e vídeo do laboratório: a criação, a preparação da armadilha e a gravação | §19 (de onde vêm as gravações) |
| 17 | Arduino do laboratório: 81% de detecção, 3% do ruído e 0% do outro mosquito em alarme; 54% → 62% → 69% → 81% | §19.16, §19.17 e §19.19 |
| 18 | descontar o som da armadilha vazia (54% → 62%) | §19.16 |
| 19 | 30/09, segundo a segundo: o rótulo automático e o de quem ouviu (69% → 81%) | §19.18 e §19.19 |
| 21 | o que falta | §20 |

**Limites** (detalhados em §17 e §20 do caderno, e ditos nos slides 11, 17, 19 e 21): os resultados são retrospectivos, não é teste cego. Na base pública a máquina acerta 3 em cada 4 alarmes, mas só avisa em metade dos Aedes. No Arduino do laboratório, os 81% vêm de 4 dias de gravação, com uma única gravação de outra espécie, e usam o rótulo de quem ouviu o áudio de 30/09, corrigido depois da primeira prova (com o rótulo automático são 69%). Falta confirmar num dia novo, com outra espécie na mesma armadilha.
