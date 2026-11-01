# Plano da v5 (exploratória): dados e modelos massivos para reconhecer *Ae. aegypti*

Escrito em 01/11/2026, **antes de qualquer treino ou avaliação da v5**. Pedido do autor: enquanto não há gravações com o Arduino, procurar datasets maiores e testar hipóteses para reconhecer *Ae. aegypti* como espécie.

Esta rodada é **exploratória**. Não há teste novo (`datasets_candidatos_v4.md`), então nenhum resultado aqui sustenta afirmação de generalização. O objetivo é escolher o que vale levar para a coleta com o Arduino.

## 1. Levantamento: onde há dados em massa

| Fonte | Volume | *Ae. aegypti* | Sensor | Situação |
|---|---|---|---|---|
| **Wingbeats** (Potamitis/Fanioudakis 2018; BioDCASE 2026 D5) | 265.652 clipes de 0,625 s, 6 espécies | 81.250 | **Óptico** | Local. Nunca usado em treino até a v4. |
| HumBugDB (Kiskin et al. 2021) | ~20 h, 9.295 registros | 123 no desenvolvimento | Microfone (vários) | Local. Os 3.500 restantes vêm dos mesmos aparelhos. |
| BioDCASE 2026 D1–D4 | 5.728 clipes | 337 (192 no D3) | Microfone | São reempacotamentos de HumBug, Abuzz, UFRGS e MosquitoSong; o D3 é o teste reservado da v3. |
| UFRGS completo (Paim et al. 2024) | ~16.700 clipes | sim | Microfone (LG K10) | Mesmo aparelho e insetário do teste reservado da v3. **Fora desta rodada.** |
| Zenodo 17287037 (DR-BioL, 2025; 1,4 GB, CC BY 4.0) | 60.124 clipes, 33,4 h | sim | Microfone | Reúne HumBug (Tanzânia), UFRGS, Abuzz e 655 clipes da Tailândia (só *Ae. albopictus*). Fontes que o projeto já tem; não traz aparelho novo com *Ae. aegypti*. **Não baixado.** |
| Modelos pré-treinados em áudio massivo: Perch 2.0, AST (AudioSet, ~2 milhões de clipes), CLAP (LAION-Audio-630K), BioLingual (sons de animais) | — | — | — | Pesos públicos |

Não existe corpus **acústico** massivo de *Ae. aegypti* fora do que o projeto já usa. Na literatura recente (arXiv 2610.09737, sobre o BioDCASE 2026), o Perch 2.0 congelado foi o encoder que mais generalizou para domínios novos (acurácia balanceada 0,307, contra 0,259 do BEATs, 0,188 do BirdAVES e 0,165 do HuBERT), e o alinhamento por MMD ajudou mais os encoders mais fortes. No baseline oficial, *Ae. aegypti* teve acurácia balanceada de 0,046 em domínio não visto, uma das piores espécies. Há duas rotas "massivas": (a) o sensor óptico, com 81 mil *Ae. aegypti*; (b) encoders pré-treinados em milhões de sons.

## 2. Hipóteses

| # | Hipótese | Como testar |
|---|---|---|
| H1a | Uma CNN treinada só no Wingbeats óptico reconhece *Ae. aegypti* em microfone sem ver áudio acústico (sem treino acústico) | Pontuação direta p(*Ae. aegypti*) em cada corpus acústico |
| H1b | O embedding dessa CNN é melhor que o Perch como representação | Mesma cabeça logística do A0, LODO |
| H1c | Pré-treinar no óptico e afinar no acústico supera a mesma CNN treinada do zero no acústico | CNN pré-treinada e afinada x controle C0 (mesma arquitetura, do zero), LODO |
| H2 | Encoders treinados em áudio massivo (AST, CLAP, BioLingual) generalizam melhor que o Perch | Embedding congelado + cabeça logística, LODO |
| H3 | A fusão das melhores representações supera cada uma | Média das pontuações padronizadas |
| H5 | Perch + cabeça MLP com alinhamento MMD entre corpora (receita reduzida de arXiv 2610.09737) supera a logística sobre o Perch | MLP 1536→256→1, perda BCE + MMD entre corpora por classe, λ = 1, 30 épocas, sem busca |
| H4 | Com a melhor representação, 1 ou 2 clusters do domínio-alvo bastam para um salto | Mesmo desenho da análise de 01/11 (`analise_dados_e_melhorias_2026_11_01.md`, §4) |

Aumento de dados no óptico: equalização aleatória suave (curva aleatória de ganho na frequência), ruído do OpenSLR 28 com SNR de 0 a 30 dB e variação de ganho. O objetivo é que a rede não use a resposta do sensor.

## 3. Dados e regras

- **Avaliação só em microfone:** Abuzz, HumBug, MosquitoSong (linhas de mosquito do desenvolvimento da v4, via `v4_train.load_development`) e **Funchal** (figshare, 3 aparelhos). O BDC restante fica sempre no treino.
- **Fora:** o teste reservado da v3 (D3/UFRGS) e todo o UFRGS. O Wingbeats óptico **só entra em treino**, nunca em avaliação.
- **Tarefa:** binária, *Ae. aegypti* x outros mosquitos, dado que há mosquito.
- **LODO:** cada corpus acústico fica fora do treino por vez. O pré-processamento é igual para todos: 8 kHz, 100–3.400 Hz, sem rótulos.
- **Cabeça comum** para todas as representações: padronização, PCA de 128 componentes quando a dimensão passa de 128, e logística com C = 0,1 e pesos iguais por classe e por fonte. Sem busca de hiperparâmetros em nenhum candidato.
- **Referência A0:** Perch normalizado com essa mesma cabeça.
- Seeds fixas. Pesos e caches ficam fora do Git; o código e as saídas leves vão para `reanalysis/outputs/v5/`.

## 4. Métricas e critério

- **Primária:** AUROC *Ae. aegypti* x outros mosquitos **dentro** de cada corpus deixado de fora, com IC95% por bootstrap de clusters, e a média nos 4 corpora.
- **Secundária:** limiar escolhido no treino (validação por cluster) para FPR ≤ 2% em outros mosquitos, e o recall e a FPR que ele dá no corpus novo. São as metas da v4.
- **Promissor:** AUROC médio ≥ A0 + 0,05 **e** melhor que o A0 em pelo menos 3 dos 4 corpora.
- **Não promissor:** qualquer outro resultado. Nada aqui é "confirmado": isso exige o teste novo.
- Os quatro corpora têm poucos clusters de *Ae. aegypti* (6, 5, 4 e 3). Diferenças menores que cerca de 0,05 são ruído.
