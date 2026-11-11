# Apresentação: "Zzzz… é Aedes?"

Versão **para leigos** (20 slides): conta a história do projeto sem entrar em modelos, bibliotecas ou código. Pouco texto por slide, uma ideia por vez, e animações: o mosquito voando, o zumbido como onda (com botão para ouvir um zumbido sintético), o som virando imagem, os exemplos entrando na "máquina", números que contam e gráficos que crescem.

- `Apresentacao_Mosquito_Interpretabilidade.html`: versão animada e autocontida. Abra no navegador; setas ou clique para avançar, `f` para tela cheia. Nos slides 6, 15 e 16 o conteúdo aparece **em passos**, um a cada clique (há um aviso "clique ▸" no canto).
- `Apresentacao_Mosquito_Interpretabilidade.pdf`: versão estática, com tudo já revelado.
- `videos/`: os quatro vídeos curtos usados nos slides (ver abaixo).

A versão anterior, técnica (30 slides, com modelos, métricas e código), continua no histórico do Git (commit `4cab181`).

**Todos os números dos slides saem do caderno** `notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb`, que é a única fonte da verdade do projeto. Se os dois divergirem, vale o caderno. Dois desenhos são ilustrações da ideia, e os slides dizem isso no rodapé: a "foto do som" pintada à esquerda no slide 8 e as barras do slide 15.

## Vídeos

Os quatro vídeos (`videos/`) vêm de Mukundarajan H., Hol F. J. H., Castillo E. A., Newby C., Prakash M. *Using mobile phones as acoustic sensors for high-throughput mosquito surveillance*. **eLife** 6:e27854 (2017), [doi:10.7554/eLife.27854](https://doi.org/10.7554/eLife.27854), licença **CC BY 4.0**. Foram recortados e comprimidos (vídeos 2, 3, 4 e 5 do artigo). Estão embutidos no HTML (WebM e MP4), que continua funcionando offline como arquivo único; nos slides, cada vídeo toca sozinho, sem som, quando o slide abre.

| Arquivo | Slide | Conteúdo |
|---|---|---|
| `asa_camera_lenta` | 4 | *Culex tarsalis* filmado a 10 kHz, mostrado 100× mais devagar |
| `gravando_garrafa`, `gravando_decolagem` | 7 | Gravação com celular: mosquito na garrafa e decolando |
| `espectrograma_ao_vivo` | 8 | Espectrograma de *Anopheles gambiae* sendo montado |


## Como a apresentação se liga ao caderno

| Slides | Assunto | Seção do caderno |
|---|---|---|
| 5 e 6 | o zumbido é uma nota (cerca de 400 batidas por segundo); os tons das espécies se misturam | §3 |
| 8 | o som vira imagem (espectrograma) | §4 |
| 9 | aprender com exemplos: 2.167 Aedes, 3.050 outros mosquitos, 426 ruídos | §2 e §5 |
| 10 | prova com aparelho nunca ouvido | §7 |
| 11 | 76% dos alarmes são Aedes; avisa em metade dos Aedes (52%) | §11 e §12 |
| 12 | em que faixas do som a máquina presta atenção | §15 |
| 14 | Arduino do laboratório: 81% de detecção, 3% do ruído e 0% do outro mosquito em alarme; 54% → 62% → 69% → 81% | §19.16, §19.17 e §19.19 |
| 15 | descontar o som da armadilha vazia (54% → 62%) | §19.16 |
| 16 | 30/09, segundo a segundo: o rótulo automático e o de quem ouviu (69% → 81%) | §19.18 e §19.19 |
| 18 | o que falta | §20 |

**Limites** (detalhados em §17 e §20 do caderno, e ditos nos slides 11, 14, 16 e 18): os resultados são retrospectivos, não é teste cego. Na base pública a máquina acerta 3 em cada 4 alarmes, mas só avisa em metade dos Aedes. No Arduino do laboratório, os 81% vêm de 4 dias de gravação, com uma única gravação de outra espécie, e usam o rótulo de quem ouviu o áudio de 30/09, corrigido depois da primeira prova (com o rótulo automático são 69%). Falta confirmar num dia novo, com outra espécie na mesma armadilha.
