# Plano da v4

Escrito em 22/10/2026, **antes de qualquer treino da v4**. O protocolo executável está em `reanalysis/protocol_v4.json` e é congelado por SHA-256 em `reanalysis/protocol_v4_freeze.json`.

## 1. Diagnóstico

O que a v3 mostrou e o que os dados permitem:

| Fato | Evidência |
|---|---|
| Nenhum candidato passou nos critérios de desenvolvimento. O descritivo final detectou 0/197 Aedes (17/197 sem o portão, com 20/371 falsos alertas). AUROC de 0,729 no teste reservado. | `reports/continuacao_v3.md` |
| O ranking existe, mas os limiares não se transferem: limiares calibrados em corpora de treino produzem zero alertas no corpus novo. | LODO v3: recall ≤ 0,83% em Perch, AUROC 0,48–0,67 |
| A origem da gravação é decodificável (acurácia balanceada de 0,96 entre Dryad e HumBug), e os modelos dependem de 130–280 Hz, abaixo do batimento do Aedes. | `reports/reanalise_v2.md` |
| **Há poucos clusters com Aedes no desenvolvimento: 16 no total.** | Abuzz 6, HumBug 5, MosquitoSong 4, BDC 1 (`results/v3_continuation/manifest.csv`, papéis `train` e `noise_train`) |
| O ruído aparece separado só no HumBug; ESC-50 e Pete são 100% ruído. | Mesmo manifesto |
| O teste reservado da v3 (BioDCASE D3, insetário UFRGS) já foi visto. | Regra da v4 |
| **Não existe hoje um teste novo** com Aedes de aparelhos ou corpora nunca usados. | `reports/datasets_candidatos_v4.md` |

Leitura: a arquitetura não é o gargalo principal. Com 16 clusters positivos, qualquer seleção entre muitos candidatos tem alta variância, e a transferência de limiares entre corpora falha mesmo quando o ranking é razoável. As hipóteses abaixo atacam o atalho de fonte e a calibração, mas **nenhuma delas substitui dados novos**.

## 2. Dados e regras

- **Desenvolvimento:** linhas do manifesto v3 com papel `train` ou `noise_train`. Ficam excluídos `locked_test`, `locked_noise` e `quarantine_holdout_related`. Testes automatizados verificam a exclusão.
- **O teste reservado da v3 não é lido** pela v4: nem para escolher modelo, hiperparâmetro ou limiar, nem para relatório.
- **Aumento de dados** usa só recursos externos nunca vistos em nenhuma rodada: RIRs e ruídos do OpenSLR 28 (Apache 2.0, 1,3 GB), mais transformações paramétricas (equalização simulando microfone e variação de ganho). Nenhum ruído de ESC-50, Pete ou HumBug entra no aumento, para não contaminar as avaliações de fonte de ruído.
- **Seeds fixas.** Os hashes de dados, código e modelos são registrados em `reanalysis/outputs/v4/`.

## 3. Hipóteses, por custo e benefício

| # | Hipótese | Custo | Benefício esperado | Como testar |
|---|---|---|---|---|
| H1 | **Diversidade de dados** (mais aparelhos, ambientes, sessões e clusters com Aedes) | Alto, fora do código | O maior | Coleta prospectiva ou pedido de dados (§7). Sem isso, não há afirmação de melhora. |
| H2 | **Calibração local por corpus** com 5, 10 ou 20 arquivos por classe resolve a falta de transferência de limiares | Baixo | Alto se o ranking transferir | Recalibrar só os mapas de probabilidade e o limiar com arquivos locais e avaliar nos clusters restantes. |
| H3 | **Restringir a banda** ao batimento (≥ 300 Hz) e usar descritores físicos reduz o atalho de fonte | Baixo | Médio | Candidatos `B_*`, com prova de fonte e oclusão de bandas antes e depois. |
| H4 | **Alinhamento de domínio** em embeddings (CORAL por corpus; DANN) reduz a decodificação da origem | Baixo a médio | Médio | Candidatos `A1` e `A2`; teste de alinhamento por grupo (corpus × classe). |
| H5 | **Aumento de dados** com RIR, ruído externo, equalização e ganho força invariância ao aparelho | Médio (cerca de 1–2 h de CPU para uma cópia aumentada) | Médio | Candidatos `C_*`, treinados com originais e uma cópia aumentada e testados só em originais. |
| H6 | **Fine-tuning leve do Perch** (adaptadores ou LoRA) | Alto: sem GPU nesta máquina | Incerto | **Pré-registrado como não executado nesta rodada.** A base de comparação é a cabeça sobre o Perch congelado (`A0`). |
| H7 | **Temperatura e sexo como metadados** | Baixo | Baixo com os dados atuais | Esses campos só existem no MosquitoSong (sexo, 25 °C) e parcialmente no HumBug. Fica como estratificação descritiva, não como entrada do modelo. |

## 4. Candidatos

Todos seguem o esquema de duas etapas da v3: presença de mosquito, depois Aedes entre mosquitos. Há calibração logística em clusters separados do ajuste, e o limiar é escolhido só na calibração, para FPR ≤ 2% em outros mosquitos e ≤ 1% em ruído.

| Candidato | Representação | Cabeça |
|---|---|---|
| `A0_perch_white` | Perch normalizado (cache v3) | Logística. **Referência:** reproduz o candidato congelado da v3. |
| `A1_perch_coral` | Perch normalizado com CORAL por corpus. No teste, as estatísticas vêm do lote não rotulado do corpus avaliado (transdutivo, declarado). | Logística |
| `A2_perch_dann` | Perch normalizado | MLP de 256 unidades com reversão de gradiente contra a origem (DANN) |
| `B1_band_tabular` | 70 estatísticas mel das bandas ≥ 300 Hz e 9 descritores do batimento | HistGB |
| `B2_wingbeat` | 9 descritores do batimento (igual ao `C_wingbeat` da v3) | Logística. **Também é o portão de concordância.** |
| `C1_aug_band_tabular` | `B1` com uma cópia aumentada por arquivo de treino | HistGB |
| `C2_aug_perch_coral` | `A1` com uma cópia aumentada por arquivo de treino | Logística |

**Ablação predefinida:** cada candidato é avaliado com e sem o portão de concordância. Com o portão, uma discordância com `B2` vira `incerto`, que permanece nos denominadores. A escolha entre com e sem portão é feita no desenvolvimento, não pelo teste da v3.

Grades e parâmetros fixos estão no protocolo. A seleção de hiperparâmetros usa 3 dobras internas por cluster, só nos clusters de ajuste.

## 5. Avaliação no desenvolvimento

- **LODO de mosquitos:** Abuzz, HumBug e MosquitoSong, cada um fora do treino por vez (o BDC tem só 1 cluster com Aedes e fica sempre no treino).
- **LODO de fontes de ruído:** ESC-50 fora e Pete fora, medindo a FPR em ruído de uma fonte nunca vista.
- **Por corpus**, não só agregado: recall, FPR em outros mosquitos, FPR em ruído, precisão, cobertura e AUROC, com IC95% por bootstrap de clusters estratificado. Quando há zero eventos, o intervalo degenerado é sinalizado e o IC binomial por arquivo aparece só como descrição.
- **Custo de calibração:** k = 5, 10 e 20 arquivos por classe do corpus fora do treino, 5 seeds. Todo cluster tocado pela calibração sai da avaliação. Recalibram-se apenas os mapas e o limiar.
- **Prova de fonte** antes e depois: classificador linear que tenta prever o corpus de origem entre mosquitos, com dobras por cluster. O resultado é a acurácia balanceada comparada ao acaso (1/número de corpora).
- **Alinhamento por grupo:** distância MMD entre corpora dentro de cada classe, antes e depois.
- **Oclusão de bandas** nos candidatos tabulares: substituir cada grupo de bandas pela média do treino e medir a variação de p(Aedes).
- **Sexo:** métricas estratificadas no MosquitoSong, só descritivas.

## 6. Seleção

1. **Elegível sem calibração local (k = 0):** recall ≥ 20%, FPR em outros ≤ 2% e FPR em ruído ≤ 1% **em cada corpus onde a métrica é mensurável** (LODO de mosquitos para recall e FPR em outros; LODO de HumBug e de fontes de ruído para FPR em ruído).
2. Se nenhum for elegível com k = 0, vale a elegibilidade com o **menor k** em que todas as metas são atingidas na mediana das 5 seeds, em todos os corpora.
3. Entre os elegíveis, vence a maior precisão média; em caso de empate, o maior AUROC.
4. **Se nenhum for elegível**, o maior AUROC médio no LODO é congelado apenas para avaliação descritiva, como na v3, e o relatório diz que não houve elegível.

As metas não são relaxadas.

## 7. Teste novo

**Critérios** (congelados no protocolo):
- Gravações de aparelhos (modelos) e locais nunca usados em nenhuma rodada.
- Pelo menos **10 clusters independentes com *Ae. aegypti*** (gaiola × sessão × aparelho) e pelo menos 10 clusters de outros mosquitos, com prioridade para *Ae. albopictus* e *Culex quinquefasciatus*, gravados nos mesmos blocos de aparelho e dia.
- Ruído ambiente gravado nos mesmos blocos.
- Sexo e temperatura registrados.
- O manifesto é congelado por SHA-256 **antes** de qualquer pontuação, e o modelo final congelado da v4 é avaliado **uma única vez**.

**Situação:** não há dataset que cumpra esses critérios (`reports/datasets_candidatos_v4.md`). O candidato final da v4 é congelado, com o hash dos pesos, antes de os dados existirem. Assim o teste novo não pode influenciar escolhas.

**Consequência:** enquanto o teste novo não existir, a resposta à pergunta "a v4 melhorou?" é **inconclusiva**, seja qual for o desempenho no desenvolvimento. O desenvolvimento usa os mesmos 16 clusters com Aedes que a v3 já explorou e não sustenta afirmação de generalização.

Desenho mínimo de coleta: 2 ou mais modelos de celular novos; 3 ou mais dias; em cada dia, gaiolas separadas de *Ae. aegypti*, *Ae. albopictus* e *Culex*, alternadas em ordem aleatória no mesmo aparelho e no mesmo ambiente, mais 5 minutos de ruído ambiente; fêmeas e machos separados; temperatura anotada.

## 8. Saídas

- `reanalysis/outputs/v4/`: partições, métricas por corpus, seleção, provas de fonte, oclusão, custo de calibração, hashes e o congelamento do candidato final.
- `reports/resultados_v4.md`: relatório.
- Novos testes: manifesto sem vazamento entre clusters, separação entre ajuste, calibração e teste, exclusão do teste reservado da v3 e determinismo.
