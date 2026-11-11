# Apresentação: "Zzzz… é Aedes?"

Apresentação para alunos da Eletrônica sobre o trabalho de um desenvolvedor de ML (30 slides, com referências no penúltimo): o processo, as ferramentas, os dados, os atributos, os modelos (árvores, redes neurais MLP e CNN, e a SVM, que é o modelo final), o ranking de 9 modelos na mesma prova, a validação por grupo, as métricas (matriz de confusão, curva ROC, limiar), o resultado, a interpretabilidade, cada método de ML em poucas linhas de código, o uso de assistentes de IA no código (Claude Code, Codex, Hermes, DeepSeek), a ponte com o hardware (Arduino) e o primeiro teste com gravações do Arduino do laboratório.

- `Apresentacao_Mosquito_Interpretabilidade.html`: versão animada e autocontida. Abra no navegador; setas ou clique para navegar, `f` para tela cheia.
- `Apresentacao_Mosquito_Interpretabilidade.pdf`: versão estática.
- `videos/`: os quatro vídeos curtos usados nos slides (ver abaixo).

**Todos os números e gráficos dos slides saem do caderno** `notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb`, que é a única fonte da verdade do projeto: ele vai do áudio bruto ao resultado, sem scripts externos. Se os dois divergirem, vale o caderno.

## Vídeos

Os quatro vídeos (`videos/`) vêm de Mukundarajan H., Hol F. J. H., Castillo E. A., Newby C., Prakash M. *Using mobile phones as acoustic sensors for high-throughput mosquito surveillance*. **eLife** 6:e27854 (2017), [doi:10.7554/eLife.27854](https://doi.org/10.7554/eLife.27854), licença **CC BY 4.0**. Foram recortados e comprimidos (vídeos 2, 3, 4 e 5 do artigo). Estão embutidos no HTML (WebM e MP4), que continua funcionando offline como arquivo único; nos slides, cada vídeo toca sozinho, sem som, quando o slide abre.

| Arquivo | Slide | Conteúdo |
|---|---|---|
| `asa_camera_lenta` | 3 | *Culex tarsalis* filmado a 10 kHz, mostrado 100× mais devagar |
| `gravando_garrafa`, `gravando_decolagem` | 7 | Gravação com celular: mosquito na garrafa e decolando |
| `espectrograma_ao_vivo` | 9 | Espectrograma de *Anopheles gambiae* sendo montado |


## Como a apresentação se liga ao caderno

| Slides | Seção do caderno |
|---|---|
| 4 (física do zumbido) | §3 |
| 8 (dados) | §2 e §5 |
| 9 (do zumbido aos números) | §4 |
| 10–12 e 16 (modelos e métodos em código) | §9 |
| 13 (não colar na prova) | §7 |
| 14 (três ajustes) | §6 e §10 (ablação) |
| 15 (como medir) | §8 |
| 17 (ranking de 9 modelos) | §10 |
| 18–21 (matriz, ROC, limiar, resultado) | §11–§13 |
| 22 (prova mais justa) | §14 |
| 24 (o que o modelo ouve) | §15 e §16 |
| 27 (teste no Arduino do laboratório) | §19.16, §19.17 e §19.19 |

**Limites** (detalhados em §17 e §20 do caderno): os resultados são retrospectivos, não é teste cego; o modelo só viu Aedes de dois datasets (Torres e Funchal) e **não** reconhece Aedes de aparelhos novos (AUROC 0,51, igual ao acaso); o resultado depende de como os poucos aparelhos do Funchal são agrupados (§11, nota). No Arduino do laboratório (slide 27, §19), a melhor receita (CNN treinada com a base e os outros dias do Arduino, calibração pelo ruído da armadilha e enxames sintéticos) acha 81% dos segundos de Aedes de um dia nunca visto, com 3% do ruído e 0% do outro mosquito em alarme. Esse número usa o rótulo de quem ouviu o áudio de 30/09, corrigido depois da primeira prova (com o rótulo automático são 69%); falta confirmar num dia novo, com outra espécie na mesma armadilha.
