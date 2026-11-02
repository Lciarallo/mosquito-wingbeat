# v5 (exploratória): datasets massivos e modelos para reconhecer *Ae. aegypti*

Executado em 01/11/2026 conforme `reports/plano_v5_exploratorio.md`, que foi commitado antes de qualquer treino. O teste reservado da v3 e o UFRGS ficaram fora. O óptico entrou só no treino. Saídas: `reanalysis/outputs/v5/`.

## Resposta direta

- **Não existe dataset acústico massivo novo de *Ae. aegypti*.** Os grandes conjuntos públicos (BioDCASE D1–D4, DR-BioL no Zenodo, HumBugDB completo) reempacotam fontes que o projeto já usa. O iNaturalist não tem nenhuma observação de *Ae. aegypti* com som.
- **O dado massivo que existe é óptico:** o Wingbeats tem 265.652 clipes, 81.250 deles de *Ae. aegypti*, e já estava no disco como BioDCASE D5. **Usar esse dado ajudou.** O embedding de uma CNN treinada só no óptico, somado ao Perch, superou o Perch sozinho **nos 4 corpora acústicos**: AUROC médio de 0,746 contra 0,675 (+0,07). A fusão escolhida de forma aninhada passou no critério de "promissor" do plano (+0,063, melhor em 3 de 4).
- **Encoders genéricos treinados em áudio massivo (AST, CLAP, BioLingual) não superaram o Perch**, como na literatura recente. O alinhamento por MMD piorou.
- **Limite:** o ranking melhorou, mas um limiar escolhido no treino ainda não se transfere. O recall fica em 0–12% com FPR em torno de 2%, longe da meta de 20%. Isso continua exigindo dados do aparelho de uso.

## 1. Datasets pesquisados

| Fonte | O que é | Serve? |
|---|---|---|
| **Wingbeats** ([Kaggle](https://www.kaggle.com/datasets/potamitis/wingbeats); BioDCASE D5) | 265.652 clipes ópticos de 0,625 s, 6 espécies, 81.250 de *Ae. aegypti* | **Sim, para treino.** Usado em H1. |
| BioDCASE 2026 D1–D4 ([Zenodo 20478577](https://zenodo.org/records/20478577)) | 5.728 clipes de microfone; só 337 de *Ae. aegypti* (192 no D3) | Já no projeto; o D3 tem o teste reservado da v3. |
| BioDCASE 2026, avaliação | Sem rótulos; mesmos domínios D1–D5 | Não. |
| DR-BioL ([Zenodo 17287037](https://zenodo.org/records/17287037), 1,4 GB) | 60.124 clipes, 33,4 h: HumBug (Tanzânia), UFRGS, Abuzz e Tailândia (só *Ae. albopictus*) | Não traz aparelho novo com *Ae. aegypti*. Não baixado. |
| HumBugDB completo | ~20 h; os 3.500 registros fora do desenvolvimento vêm dos mesmos aparelhos | Mais arquivos dos mesmos clusters não ajudam (`analise_dados_e_melhorias_2026_11_01.md`). |
| iNaturalist (API, 01/11/2026) | 0 observações de *Ae. aegypti* com som; 47 de Culicidae, quase todas *Toxorhynchites* | Não. |
| Encoders pré-treinados: Perch 2.0, AST (AudioSet), CLAP (LAION-Audio-630K), BioLingual | Pesos públicos | Testados em H2. |

## 2. Resultados LODO

Cada corpus fica fora do treino por vez. A métrica é o AUROC de *Ae. aegypti* x outros mosquitos dentro do corpus deixado de fora, com IC95% por bootstrap de clusters. Clusters de *Ae. aegypti*: Abuzz 6, HumBug 5, MosquitoSong 4, Funchal 3. **Com só 3 clusters, o IC do Funchal é degenerado** (às vezes não contém o ponto) e deve ser lido como indicativo.

| Candidato | Abuzz | HumBug | MosquitoSong | Funchal | Média | vs. A0 |
|---|---|---|---|---|---|---|
| **A0 Perch + logística (referência)** | 0,706 | 0,638 | 0,582 | 0,774 | 0,675 | — |
| **A0 + H1b, média dos logits** | **0,788** [0,69–0,89] | **0,721** [0,64–0,84] | **0,659** [0,48–0,81] | **0,816** | **0,746** | +0,071, 4/4 |
| A0 + H1b + H1c, média dos logits | 0,876 | 0,735 | 0,679 | 0,757 | 0,762 | +0,087, 3/4 |
| H3 fusão aninhada (top 3 escolhido sem o corpus-alvo) | 0,863 | 0,742 | 0,628 | 0,718 | 0,738 | **+0,063, 3/4: promissor** |
| H1b embedding da CNN óptica + logística | 0,746 | 0,723 | 0,649 | 0,714 | 0,708 | +0,033, 3/4 |
| H1c CNN óptica afinada no acústico | 0,863 | 0,713 | 0,606 | 0,613 | 0,699 | +0,024, 3/4 |
| C0 mesma CNN, do zero, só acústico (controle) | 0,824 | 0,679 | 0,524 | 0,672 | 0,675 | 0, 2/4 |
| H2 AST (AudioSet) | 0,801 | 0,645 | 0,545 | 0,517 | 0,627 | −0,048 |
| H2 BioLingual, sem treino (texto "Aedes aegypti") | 0,716 | 0,596 | 0,644 | 0,600 | 0,639 | −0,036 |
| H2 BioLingual + logística | 0,749 | 0,625 | 0,467 | 0,619 | 0,615 | −0,060 |
| H2 CLAP + logística | 0,679 | 0,597 | 0,490 | 0,528 | 0,574 | −0,101 |
| H1a CNN óptica aplicada direto, sem treino acústico | 0,427 | 0,557 | 0,497 | 0,816 | 0,574 | −0,101 |
| H5 Perch + MLP com MMD | 0,503 | 0,491 | 0,454 | 0,698 | 0,536 | −0,138 |

O pré-registro previa uma fusão "das melhores representações". A versão honesta é a **aninhada** (H3): para cada corpus, as 3 componentes são escolhidas pelo AUROC nos outros 3 corpora. Nos 4 corpora ela escolheu o embedding óptico, a CNN óptica afinada e o Perch ou a CNN do zero. As médias simples de logits (A0 + H1b etc.) foram calculadas depois, para conferir que o ganho não vem da padronização por postos no lote de teste. Não vem: a média dos logits não usa o lote e dá o mesmo resultado.

### O que cada hipótese mostrou

- **H1, dado óptico massivo.**
  - Sozinha, a CNN óptica não serve para o microfone. Aplicada direto, acerta o Funchal (0,82) e falha no Abuzz (0,43). Restringir os negativos às espécies que existem no óptico não muda isso (Abuzz 0,35, MosquitoSong 0,44). A barreira é o sensor, não as espécies.
  - Como **representação**, ela traz informação que o Perch não tem. H1b + A0 melhora todos os corpora.
  - Pré-treinar no óptico e afinar no acústico (H1c) supera a mesma CNN treinada do zero (C0) em 3 de 4 corpora: 0,699 contra 0,675.
- **O aumento de dados no óptico foi decisivo.** Sem equalização aleatória e ruído, a mesma CNN dá 0,34 no Funchal na aplicação direta e 0,31 como embedding. No próprio óptico, as duas versões acertam igual (acurácia balanceada de 0,88 com aumento e 0,91 sem; recall de *Ae. aegypti* de 0,85 e 0,88). Ou seja, o aumento não melhora o óptico; ele torna a rede robusta à troca de sensor.
- **H2, encoders genéricos.** AST, CLAP e BioLingual ficaram abaixo do Perch, como em arXiv 2610.09737 (Perch 0,307 > BEATs 0,259 > BirdAVES 0,188 no BioDCASE). O BioLingual sem treino nenhum, só com o texto "Aedes aegypti", chegou a 0,64 no MosquitoSong. É curioso, mas fraco.
- **H5, MMD.** Piorou (0,536). A mesma MLP sem o termo de MMD fez 0,630. Nesta implementação, o alinhamento tira informação de espécie. Na v4, o CORAL também piorou.
- **Mais um corpus no treino ajuda.** Com o Funchal no treino, o A0 sobe no HumBug (0,552 → 0,638) e no Abuzz (0,652 → 0,706); o MosquitoSong fica igual (0,587 → 0,582). O Funchal traz 1.876 *Ae. aegypti* de 3 aparelhos novos. Isso confirma a análise anterior: o que ajuda é aparelho novo, não mais arquivos do mesmo aparelho.

### Limiar (metas da v4)

| Candidato | Abuzz recall / FPR | HumBug | MosquitoSong | Funchal |
|---|---|---|---|---|
| A0 | 0% / 0,9% | 0% / 0,3% | 0% / 0% | 0,6% / 0,2% |
| H1b | 0% / 0% | 4,1% / 1,7% | 1,9% / 0,5% | 11,8% / 1,3% |

O limiar escolhido no treino para FPR ≤ 2% continua quase sem alertas no corpus novo. **Nenhum candidato atinge recall ≥ 20%.** O ranking melhorou; a calibração entre aparelhos, não.

## 3. H4: dados locais, com o melhor candidato

Mesmo desenho da análise de 01/11: m clusters do corpus-alvo entram no treino e o teste usa os clusters restantes, com 20 sorteios. AUROC médio sem → com dados locais, para A0 + H1b:

| Alvo | m = 1 | m = 2 | m = 3 | m = 4 |
|---|---|---|---|---|
| Abuzz | 0,79 → 0,84 | 0,79 → 0,89 | 0,80 → 0,91 | 0,78 → **0,93** |
| HumBug | 0,72 → 0,78 | 0,72 → 0,80 | 0,70 → **0,82** | — |
| MosquitoSong (1 mosquito por cluster) | 0,65 → 0,62 | 0,62 → 0,61 | — | — |
| Funchal (1 aparelho por cluster) | 0,79 → 0,64 | — | — | — |

Clusters do **mesmo arranjo de gravação** (Abuzz, HumBug) levam o AUROC a 0,82–0,93. Clusters que são **outro aparelho** (Funchal) ou **um único mosquito** (MosquitoSong) pioram. Para o Arduino, isso significa: gravar várias sessões **no próprio aparelho**, com vários mosquitos por gaiola.

## 4. Candidato congelado

`reanalysis/v5_final.py freeze` ajustou as duas cabeças (Perch e embedding óptico) em todo o desenvolvimento acústico. Os SHA-256 das cabeças, da CNN óptica e do Perch estão em `reanalysis/outputs/v5/final_freeze.json`. Para pontuar gravações novas (por exemplo, do Arduino) **sem retreinar**:

```bash
.venv-v3/bin/python -m reanalysis.v5_final predict gravacao.wav
```

A saída é o logit médio. O limiar gravado (FPR ≤ 2% na validação por cluster) é **só de pesquisa**: o LODO mostra que ele não se transfere. O plano para o Arduino:
1. Pontuar as primeiras sessões com o modelo congelado. Esse é o teste novo.
2. Só depois recalibrar e retreinar as cabeças com clusters locais.

## 5. Recomendações

1. **Levar o candidato A0 + H1b para o teste com o Arduino**, congelado como está.
2. **Na coleta**, priorizar sessões e gaiolas no aparelho de uso (H4), com vários mosquitos por gaiola e os dois sexos anotados.
3. **Se for retreinar a CNN óptica**, manter o aumento de dados. Vale testar uma equalização mais forte ou a resposta medida do microfone do Arduino. Gravar uma varredura de frequência com o próprio microfone permite simular o sensor no óptico.
4. **Não investir em** encoders genéricos (AST, CLAP), MMD ou CORAL: três rodadas seguidas sem ganho.

## Limites

- **Exploratório e retrospectivo.** Não há teste novo. Os 4 corpora já foram vistos em rodadas anteriores, e a fusão foi a última escolha da rodada.
- **Poucos clusters:** 3 a 6 de *Ae. aegypti* por corpus. Diferenças de 0,03 a 0,05 cabem no ruído. O IC do Funchal é degenerado.
- Os encoders da H2 receberam a mesma entrada de 8 kHz (até 3,4 kHz) do Perch. Isso favorece o Perch, que foi avaliado assim na v3, e desfavorece modelos treinados em áudio de banda larga.
- Um único seed por CNN. A variabilidade de treino não entra nos IC.
- O cache acústico guarda a forma de onda em float16. O `predict` usa float32; a diferença no embedding do Perch é de no máximo 0,08 por componente.

## Reprodução

```bash
.venv-v3/bin/python -m reanalysis.v5_data optical && .venv-v3/bin/python -m reanalysis.v5_data acoustic
.venv-v3/bin/python -m reanalysis.v5_embed perch
.venv-v4-rocm/bin/python -m reanalysis.v5_embed ast   # e clap, biolingual (GPU; transformers e torchaudio)
.venv-v4-rocm/bin/python -m reanalysis.v5_optical pretrain
.venv-v4-rocm/bin/python -m reanalysis.v5_optical apply
.venv-v4-rocm/bin/python -m reanalysis.v5_optical lodo
.venv-v4-rocm/bin/python -m reanalysis.v5_eval main
.venv-v4-rocm/bin/python -m reanalysis.v5_fusion_check
.venv-v4-rocm/bin/python -m reanalysis.v5_eval fewshot --reps A0_perch H1b_optico_embedding_aug A0_perch+H1b_optico_embedding_aug
.venv-v3/bin/python -m reanalysis.v5_final freeze
```

Tempo: cerca de 2 h numa RX 9070 XT com 28 núcleos. A CNN óptica leva cerca de 40 min por versão, e cada encoder da H2 cerca de 30 min.
