# Generalização entre fontes (leave-one-source-out) — 03/10/2026

Script: `loso_generalization.py` · métricas: `results/loso_2026_10_03/loso_metrics.csv` · protocolo: `results/loso_2026_10_03/protocol.json`.

**Correção de 07/10/2026:** o código atual escreve em outra pasta. A tabela abaixo preserva a execução histórica. O Dryad foi classificado erroneamente como óptico no filtro `hgb_acoustic_only`; os resultados dessa linha não avaliam corretamente uma restrição a fontes acústicas. Veja a [auditoria e a repetição corrigida](auditoria_pesquisa_2026_10_07.md).

## Método

- Alvo: *Aedes aegypti* contra não-*Aedes*. Mosquitos de espécie desconhecida ficam fora, como nas rodadas anteriores.
- Para cada fonte com as duas classes (Dryad, HumBug, MosquitoSong), o treino usa as outras fontes. A fonte retirada não entra no ajuste nem na seleção de variáveis.
- As 138 características acústicas já em cache foram usadas. Os embeddings PCEN foram excluídos: o codificador `pcen_large` viu essas fontes no treino, o que causaria vazamento.
- O áudio privado (UFRGS/Drive) foi excluído do treino e da avaliação.
- Dez candidatos foram declarados antes da execução. O IC 95% do AUC reamostra gravações inteiras, não janelas.
- As variantes `*_source_norm` usam a média e o desvio, sem rótulos, de todas as janelas da fonte avaliada, incluindo o teste. São avaliações transdutivas: precisam de acesso ao lote de destino. Um conjunto de ruído ambiente separado não equivale à distribuição desse lote. As variantes também retiram Figshare, portanto a comparação com o baseline não isola o efeito da normalização.

## AUC por fonte retirada

| Modelo | Dryad | HumBug | MosquitoSong | Média | Pior |
| --- | --- | --- | --- | --- | --- |
| lr_source_norm | 0.790 | 0.669 | 0.640 | 0.700 | 0.640 |
| lr_baseline | 0.832 | 0.671 | 0.628 | 0.710 | 0.628 |
| et_baseline | 0.805 | 0.576 | 0.569 | 0.650 | 0.569 |
| hgb_baseline | 0.790 | 0.601 | 0.568 | 0.653 | 0.568 |
| hgb_source_norm | 0.568 | 0.709 | 0.568 | 0.615 | 0.568 |
| et_no_positive_only_source | 0.682 | 0.556 | 0.572 | 0.603 | 0.556 |
| et_source_norm | 0.419 | 0.599 | 0.500 | 0.506 | 0.419 |
| hgb_acoustic_only | 0.694 | 0.412 | 0.458 | 0.521 | 0.412 |
| et_consistent40 | 0.353 | 0.568 | 0.538 | 0.486 | 0.353 |
| lr_consistent40 | 0.323 | 0.564 | 0.541 | 0.476 | 0.323 |

## Leitura

- Com as características atuais, a separação entre fontes é fraca: o AUC fica perto de 0,5–0,7 em fontes acústicas inéditas. O IC de MosquitoSong (10 gravações) é muito largo.
- O limiar 0,5 não transfere: a sensibilidade e a especificidade ficam em extremos opostos conforme o modelo e a fonte. Isso explica parte do colapso visto nas rodadas anteriores.
- Os modelos lineares tiveram AUC maior que as árvores baseline nas três fontes observadas. Isso é compatível com sobreajuste às fontes, mas não demonstra sua causa ou superioridade fora dessas fontes.
- A seleção por consistência teve resultados piores nesta execução. Isso não descarta a estratégia em outras representações. A conclusão sobre treinar só com áudio acústico foi retirada por causa do erro de classificação do Dryad.
- A normalização transdutiva e a retirada de Figshare mudaram os resultados em direções diferentes conforme o modelo e a fonte. Não foi demonstrada uma melhora geral, nem o efeito isolado de cada mudança.
- A tabela histórica calculava AUC, sensibilidade e especificidade, mas não precisão/F1 ou seus intervalos. A auditoria de 07/10 recalculou essas métricas: nenhum candidato histórico satisfaz simultaneamente os mínimos de 0,60 nas três fontes.

## Limitações

- A melhor variante foi escolhida olhando as mesmas três fontes, então há viés de seleção. Uma melhora real precisa ser confirmada em dados novos, congelados antes de olhar.
- O [Dryad 10.5061/dryad.98d7s](https://datadryad.org/dataset/doi:10.5061/dryad.98d7s) contém gravações acústicas de celulares. O corpus óptico de Fanioudakis é outro conjunto. Mudanças de aparelho, ambiente, composição das classes e protocolo continuam confundidas na comparação entre fontes.
- Nenhum modelo do projeto foi alterado, e o firmware e o instalador também não.
