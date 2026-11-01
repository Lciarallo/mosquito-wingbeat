# Mais datasets ajudariam? Como ter resultados melhores

Análise de 01/11/2026. **Tudo aqui é retrospectivo:** usa dados que o projeto já viu e não toca no teste reservado da v3 (os dados da v4 vêm de `v4_train.load_development`; o UFRGS sai do pool da apresentação). Código: `reanalysis/data_scaling.py`. Saídas: `reanalysis/outputs/data_scaling/`.

## Resposta curta

**Depende do tipo de dado.**

| Tipo de dado a mais | Ajuda? | Evidência (abaixo) |
|---|---|---|
| Mais arquivos dos mesmos clusters (mesmo aparelho, sessão, indivíduo) | **Não** | Com todos os clusters, usar 5% ou 100% dos arquivos dá o mesmo AUROC no HumBug e no MosquitoSong. |
| Corpus com uma classe só (Torres = só Aedes, ESC-50 = só ruído) | **Atrapalha** | O modelo aprende a fonte: o Torres é reconhecido com AUROC 0,98, e Aedes de outros aparelhos ficam abaixo do acaso (0,41). |
| Mais corpora públicos de laboratório, somados ao treino | **Pouco e de forma instável** | De 1 para 3 corpora, o AUROC dentro do corpus novo sobe 0,01 a 0,24, mas só com normalização por fonte. Sem ela, às vezes cai. O MosquitoSong fica perto do acaso. Com 11 clusters de Aedes de fora, o HumBug fica em 0,51 a 0,54. |
| Mais clusters independentes (aparelho × sessão × gaiola) em vez de mais horas | **Sim, mais que volume** | Com o mesmo número de arquivos, ter mais clusters dá AUROC igual ou maior (até +0,16 no Abuzz). |
| **Dados rotulados do domínio de uso** (mesmo aparelho e ambiente), com Aedes e outros mosquitos nos mesmos blocos | **Sim, é o que mais ajuda** | Com logística sobre o Perch, 1 a 3 clusters do próprio domínio sobem o AUROC em +0,04 a +0,20 nos clusters restantes, com ganho em 15 a 20 de 20 sorteios. Dez clusters de fora não fazem isso. |

Conclusão: **juntar mais datasets públicos não leva o detector às metas.** O ganho vem de clusters do aparelho e do ambiente onde ele vai rodar, e qualquer afirmação de melhora precisa de um teste novo (coleta prospectiva, `plano_v4.md` §7). O BioDCASE 2026 tem o mesmo limite: o baseline cai de 0,88 de acurácia balanceada em domínios vistos para 0,17 em domínios novos.

## 1. O que o AUROC 0,86 da apresentação mede

O SVM final da apresentação foi refeito (`decompose`) com a mesma receita: Aedes só do Torres e do Funchal, normalização e pesos por fonte e dois estágios. O AUROC deu **0,858** (0,856 nos slides; a diferença vem de o UFRGS ter saído).

| Recorte | AUROC |
|---|---|
| Agregado (o número dos slides) | 0,858 |
| Aedes do Torres x todos os negativos | **0,982** |
| Aedes do Funchal x todos os negativos | 0,838 |
| Aedes x *Culex* dentro do Funchal | 0,869 |
| Aedes de aparelhos fora do Torres e do Funchal (`so_torres_funchal.json`) | **0,41** |

- **O Torres é reconhecido pela fonte, não pela espécie.** O Torres tem 100% de Aedes e só espectrogramas prontos, e a prova de fonte separa o Torres de todo o resto com AUROC 0,998 (`probe_torres.json`). Por isso 286 dos 300 são detectados.
- **Alarmes falsos com limiar 0,5, por fonte:** MosquitoSong 148/960 (15%), Funchal 92/1.077 (8,5%), ESC-50 14/289 (4,8%), Dryad 7/593, HumBug 5/552.
- **Erro no agrupamento do Funchal.** Em `svm_final.py` e `aedes_torres_funchal.py`, `'8khz' in g` também é verdadeiro para `48khz`. As pastas de 48 kHz caem no grupo de 8 kHz, e o Funchal fica com **2 grupos, não 3**: 8 kHz + 48 kHz (1.778 Aedes, 1.060 outros) e 44,1 kHz (89 Aedes, 17 outros). Isso não cria vazamento, porque juntar grupos é conservador, mas quase toda a validação "por aparelho" do Funchal é uma dobra só. O `data_scaling.py` corrige o agrupamento nas análises 2 a 4 e só repete o erro na análise 1, para reproduzir os slides.
- **A normalização por fonte depende da proporção de Aedes no lote.** O teste fixou um modelo treinado sem o Funchal e trocou só o lote usado para normalizar o Funchal:

| Aedes no lote | Aedes detectados | *Culex* chamados de Aedes | AUROC Aedes x *Culex* |
|---|---|---|---|
| 10% | 52% | 10% | 0,88 |
| 30% | 39% | 3,8% | 0,87 |
| 63% (o real) | 18% | 1,0% | 0,78 |
| 90% | 12% | 0,8% | 0,62 |

Em campo, a proporção de Aedes no lote é justamente o que se quer descobrir. Com esse tipo de normalização, a mesma gravação recebe outra nota conforme o que veio junto. Além disso, a normalização usa as estatísticas do próprio conjunto de teste.

## 2. Treinar com 1, 2 ou 3 corpora

Pool da apresentação: 80 atributos HTK por arquivo, sem UFRGS e sem Torres (o Torres não tem negativos, então não permite medir AUROC dentro da fonte). Um corpus fica fora; o treino usa k dos outros três. O AUROC é de Aedes x outros mosquitos **dentro** do corpus deixado de fora, onde a fonte não pode servir de atalho. Média sobre os subconjuntos de tamanho k.

| Modelo | Normalização por fonte | Dryad k=1→2→3 | HumBug | MosquitoSong | Funchal |
|---|---|---|---|---|---|
| SVM RBF | sim | 0,48 → 0,59 → 0,70 | 0,68 → 0,72 → 0,74 | 0,55 → 0,57 → 0,60 | 0,65 → 0,78 → 0,81 |
| SVM RBF | não | 0,47 → 0,55 → 0,68 | 0,58 → 0,53 → 0,50 | 0,54 → 0,55 → 0,56 | 0,55 → 0,59 → 0,55 |
| HistGB | sim | 0,54 → 0,67 → 0,78 | 0,64 → 0,70 → 0,72 | 0,50 → 0,49 → 0,51 | 0,51 → 0,59 → 0,63 |
| HistGB | não | 0,55 → 0,56 → 0,67 | 0,58 → 0,48 → **0,38** | 0,55 → 0,52 → 0,50 | 0,44 → 0,45 → **0,32** |
| Logística | sim | 0,45 → 0,51 → 0,58 | 0,64 → 0,63 → 0,66 | 0,56 → 0,57 → 0,58 | 0,72 → 0,74 → 0,76 |

Leitura:
- Com normalização por fonte, mais corpora tendem a melhorar o **ranking**. O ganho varia muito entre alvos, e o MosquitoSong não sai de perto de 0,5.
- Sem normalização, somar fontes pode **piorar**: o HistGB cai abaixo do acaso no HumBug e no Funchal. Cada fonte nova traz outro atalho de aparelho.
- O que funciona no Funchal não funciona no MosquitoSong. Treinar só com o Funchal dá 0,43 no MosquitoSong; treinar com o MosquitoSong dá 0,50 no Funchal.
- Limites: a normalização do alvo é transdutiva e herda o problema da seção 1. No pool, Dryad e HumBug estão agrupados por bloco ou arquivo, não pelos clusters da v4, e o Funchal tem 2 grupos de Aedes. Os IC por grupo gravados no JSON são otimistas e não foram usados na leitura.

## 3. Clusters x arquivos (dados da v4)

Desenvolvimento da v4 com os clusters reais: Abuzz 6 clusters de Aedes, HumBug 5, MosquitoSong 4 (um mosquito por cluster) e BDC 1. Binário Aedes x outros mosquitos, deixando um corpus fora. Na curva de **clusters**, sorteiam-se n clusters de Aedes do treino e entram todos os arquivos deles. Na curva de **arquivos**, entram todos os clusters, mas só uma fração dos arquivos de cada. São 12 sorteios por nível.

| Alvo | Modelo | 1 cluster | Todos os clusters de fora | 5% dos arquivos de todos | 100% |
|---|---|---|---|---|---|
| Abuzz (20 Aedes) | HistGB bandas ≥ 300 Hz | 0,49 | 0,84 (10 clusters) | 0,65 | 0,84 |
| Abuzz | Logística sobre o Perch | 0,62 | 0,68 | 0,75 | 0,68 |
| HumBug | HistGB | 0,44 | 0,51 (11) | 0,47 | 0,51 |
| HumBug | Logística sobre o Perch | 0,55 | 0,54 | 0,56 | 0,54 |
| MosquitoSong | HistGB | 0,50 | 0,54 (12) | 0,50 | 0,54 |
| MosquitoSong | Logística sobre o Perch | 0,56 | 0,59 | 0,60 | 0,59 |

Com o mesmo número de arquivos de Aedes, mais clusters ganham de poucos clusters. Exemplos: Abuzz, HistGB, cerca de 120 arquivos: 0,59 com até 5 clusters e 0,75 com todos. Logística: 0,55 e 0,71. No HumBug e no MosquitoSong, a diferença é de no máximo 0,04. **Diversidade vale mais que volume**, mas só onde já existe alguma transferência. Para o HumBug e o MosquitoSong, 10 a 12 clusters de outros corpora não tiram o modelo do acaso. Não há tendência que, extrapolada, chegue às metas.

## 4. Dados do próprio domínio

O mesmo LODO da v4, agora com m clusters de Aedes do corpus-alvo no treino. Entra também a mesma fração dos clusters de outros mosquitos do alvo. O teste usa os clusters restantes do alvo, comparados no mesmo sorteio com e sem os dados do alvo (20 sorteios).

| Alvo | Modelo | m = 1 | m = 2 | m = 3 |
|---|---|---|---|---|
| Abuzz | Logística sobre o Perch | 0,65 → 0,74 (16/20) | 0,67 → 0,81 (19/20) | 0,65 → **0,85** (20/20) |
| HumBug | Logística sobre o Perch | 0,53 → 0,60 (18/20) | 0,53 → 0,63 (20/20) | 0,53 → **0,68** (20/20) |
| MosquitoSong | Logística sobre o Perch | 0,59 → 0,63 (15/20) | 0,60 → **0,70** (16/20) | — |
| HumBug | HistGB bandas | 0,50 → 0,56 (19/20) | 0,51 → 0,56 (20/20) | 0,52 → 0,62 (20/20) |
| Abuzz | HistGB bandas | 0,85 → 0,81 | 0,84 → 0,79 | 0,84 → 0,81 |
| MosquitoSong | HistGB bandas | 0,53 → 0,45 | 0,56 → **0,38** | — |

(sem → com dados do alvo; entre parênteses, os sorteios em que houve ganho)

- **Um cluster do HumBug vale mais que onze clusters de fora** na logística sobre o Perch: +0,07, contra uma curva plana na seção 3.
- O HistGB sobre bandas **piora** com dados do alvo no Abuzz e no MosquitoSong. Com poucos indivíduos, a árvore decora o cluster. No MosquitoSong, cada cluster é um mosquito, de um sexo e uma idade. Numa checagem rápida (só com Aedes do alvo no treino, sem os outros mosquitos dele), ao entrar só Aedes fêmeas, o AUROC nos machos caiu de 0,54 para 0,38. É um indício de confusão por sexo, não uma prova.
- A v4 testou só **recalibrar** o mapa de probabilidade e o limiar com k arquivos locais, e não chegou às metas. Aqui é a **cabeça** que é retreinada com clusters locais. São coisas diferentes, e a segunda é o que funciona.

## 5. Contexto externo: BioDCASE 2026 (CD-MSC)

- Baseline (MTRCNN, log-mel, [arXiv 2603.20118](https://arxiv.org/abs/2603.20118)): acurácia balanceada de **0,88 em domínios vistos e 0,17 a 0,18 em não vistos**. Os organizadores atribuem a queda a artefatos de gravação próprios de cada domínio.
- Resultados do desafio ([página](https://biodcase.github.io/challenge2026/task5-results)): 13 equipes; a melhor fez **0,31** em domínios não vistos. A página não diz qual partição foi usada.
- O conjunto de avaliação ([Zenodo 20478577](https://zenodo.org/records/20478577)) **não tem rótulos publicados** e só usa os domínios D1–D5. "Não visto" significa "não visto para aquela espécie", não aparelho novo. **Não serve como teste novo.**

O problema do projeto é o problema da área: reconhecer em condições parecidas com as do treino é fácil; transferir para um aparelho novo, não.

## 6. Quais datasets valem a pena

| Dataset | Para quê | Prioridade |
|---|---|---|
| **Coleta própria com o aparelho de uso** (Arduino/microfone do projeto), no desenho de `plano_v4.md` §7 | Treino do domínio e **teste novo** (reservar uma parte antes de treinar) | **1** |
| MosquitoSong+ não publicado e Gana (KNUST), sob pedido | Teste novo, se tiver *Ae. aegypti* com sessão identificada | 2 |
| UFRGS RD6 (345 machos) | Treino com machos. Mesmo aparelho do teste da v3, então **nunca** como teste | 3 |
| Wingbeats óptico (BioDCASE D5) | No máximo pré-treino; é outro sensor | baixa |
| MUSAN, DEMAND, OpenSLR 28 | Aumento com ruído e reverberação (o C1 da v4 não melhorou) | baixa |
| Restante do HumBugDB (mesmos aparelhos) | Mais arquivos dos mesmos clusters: não ajuda | não |
| Torres 2025, ESC-50 como fonte de uma classe só | Ensinam a fonte | não, para treino |
| BioDCASE avaliação, AudioSet "Mosquito" | Sem rótulo de espécie | não |

## 7. Como ter resultados melhores, por prioridade

1. **Coletar no domínio de uso.** É a única alavanca com efeito grande nas análises acima. Priorize **número de sessões, gaiolas e dias** (clusters) em vez de horas por sessão. Em cada bloco, grave *Ae. aegypti*, *Ae. albopictus* e *Culex* no mesmo aparelho e ambiente, mais ruído de fundo, anotando sexo e temperatura. Congele uma parte como teste antes de qualquer treino.
2. **Pré-registrar a adaptação com clusters locais** como hipótese da próxima rodada: retreinar a cabeça logística sobre o Perch com 1 a 3 clusters do aparelho novo, além de recalibrar o limiar. Meça quantos clusters locais bastam para chegar às metas.
3. **Trocar a normalização por fonte** por uma que não dependa de rótulo nem da proporção de Aedes. Por exemplo, estatísticas do ruído de fundo do aparelho (trechos sem inseto) ou normalização por gravação, como o `perch_white` já faz. Ao falar da apresentação, diga que o 0,86 depende de normalizar com o lote do próprio teste.
4. **Tirar o Torres do treino**, ou deixá-lo só como descrição, e **relatar AUROC por fonte e dentro da fonte**, nunca só o agregado.
5. **Preferir modelos simples e regularizados sobre embeddings** (logística sobre o Perch) quando há poucos indivíduos. O HistGB sobre bandas decorou clusters.
6. **Tratar o sexo:** treinar só fêmeas, se o uso for armadilha de fêmeas, ou separar sexo antes de espécie. O MosquitoSong mistura um indivíduo de cada sexo por espécie, e há indício de confusão.
7. **Registrar a temperatura** na coleta. A frequência do batimento varia com ela, e hoje só o MosquitoSong (25 °C) tem esse dado.
8. Corrigir o agrupamento de 48 kHz do Funchal nos scripts da apresentação, se forem rodados de novo.

## Limites

- Nada aqui é teste cego: são os mesmos 16 clusters com Aedes que a v3 e a v4 já exploraram, e o pool da apresentação já foi analisado várias vezes.
- Poucos clusters por alvo (2 a 5 de Aedes no teste da seção 4). As médias entre sorteios mostram tendência, não um IC de generalização.
- Os modelos e hiperparâmetros foram fixados sem busca. Não são os candidatos congelados da v4, então os números não se comparam um a um com `resultados_v4.md`.
- AUROC mede ranking. As metas da v4 (recall ≥ 20% com FPR ≤ 2%) exigem também um limiar que se transfira, e nenhuma análise aqui mostra isso.

## Reprodução

```bash
.venv-v3/bin/python -m reanalysis.data_scaling decompose   # precisa de apresentacao/experimento_mistura/pool.npz (local)
.venv-v3/bin/python -m reanalysis.data_scaling sources     # idem
.venv-v3/bin/python -m reanalysis.data_scaling clusters    # cache v3_continuation (local)
.venv-v3/bin/python -m reanalysis.data_scaling indomain    # idem
```

Tempo total: cerca de 30 minutos em 14 núcleos de CPU.
