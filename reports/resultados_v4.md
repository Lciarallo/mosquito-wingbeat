# v4: resultados do desenvolvimento e estado do teste novo

Executado em 22/10/2026 conforme `reanalysis/protocol_v4.json` (congelado em `protocol_v4_freeze.json`, com as emendas A1 — fine-tuning do Perch na GPU — e A2 — correção numérica —, todas congeladas antes de qualquer treino). Usa só os papéis de desenvolvimento da v3; o teste reservado da v3 não foi lido. Saídas auditáveis em `reanalysis/outputs/v4/`.

## Resposta direta

**Os resultados melhoraram no teste novo, com IC por cluster? Inconclusivo.** Não existe teste novo que cumpra os critérios pré-registrados (`reports/datasets_candidatos_v4.md`), e o desenvolvimento reutiliza os mesmos 16 clusters com Aedes já explorados na v3. No desenvolvimento, **nenhum candidato atingiu** as metas por corpus (recall ≥ 20%, FPR ≤ 2% em outros mosquitos, ≤ 1% em ruído). Congelado para o teste futuro: **B1 bandas ≥ 300 Hz**, sem portão, apenas como referência descritiva (maior AUROC médio no LODO).

## LODO de mosquitos: cada corpus fora do treino

Recall / FPR em outros mosquitos, por corpus, sem calibração local (k = 0). IC95% por bootstrap de clusters entre colchetes no AUROC.

| Candidato | Portão | Abuzz recall / FPR | HumBug recall / FPR | MosquitoSong recall / FPR | AUROC Abuzz | AUROC HumBug | AUROC MosquitoSong |
|---|---|---|---|---|---|---|---|
| A0 Perch normalizado (referência v3) | não | 0.0% / 0.2% | 4.1% / 2.7% | 5.5% / 5.8% | 0.623 [0.33–0.86] | 0.585 [0.46–0.68] | 0.562 [0.45–0.66] |
| A0 Perch normalizado (referência v3) | sim | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.1% | 0.623 [0.33–0.86] | 0.585 [0.46–0.68] | 0.562 [0.45–0.66] |
| A1 Perch + CORAL | não | 0.0% / 1.8% | 11.4% / 5.9% | 1.7% / 1.5% | 0.525 [0.40–0.73] | 0.551 [0.48–0.64] | 0.530 [0.45–0.62] |
| A1 Perch + CORAL | sim | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.525 [0.40–0.73] | 0.551 [0.48–0.64] | 0.530 [0.45–0.62] |
| A2 Perch + DANN | não | 20.0% / 1.3% | 4.1% / 3.4% | 6.6% / 6.6% | 0.627 [0.26–0.91] | 0.577 [0.43–0.70] | 0.555 [0.41–0.69] |
| A2 Perch + DANN | sim | 0.0% / 0.0% | 0.0% / 0.0% | 0.3% / 0.1% | 0.627 [0.26–0.91] | 0.577 [0.43–0.70] | 0.555 [0.41–0.69] |
| B1 bandas ≥ 300 Hz | não | 5.0% / 0.2% | 2.4% / 1.1% | 3.9% / 1.4% | 0.691 [0.31–0.96] | 0.632 [0.49–0.73] | 0.506 [0.36–0.65] |
| B1 bandas ≥ 300 Hz | sim | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.691 [0.31–0.96] | 0.632 [0.49–0.73] | 0.506 [0.36–0.65] |
| B2 batimento | não | 0.0% / 0.0% | 0.0% / 0.0% | 1.7% / 5.3% | 0.459 [0.26–0.66] | 0.459 [0.34–0.62] | 0.413 [0.20–0.69] |
| B2 batimento | sim | 0.0% / 0.0% | 0.0% / 0.0% | 1.7% / 5.3% | 0.459 [0.26–0.66] | 0.459 [0.34–0.62] | 0.413 [0.20–0.69] |
| C1 bandas ≥ 300 Hz + aumento | não | 0.0% / 0.2% | 3.3% / 1.9% | 5.0% / 1.8% | 0.671 [0.29–0.94] | 0.633 [0.51–0.73] | 0.509 [0.38–0.64] |
| C1 bandas ≥ 300 Hz + aumento | sim | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.671 [0.29–0.94] | 0.633 [0.51–0.73] | 0.509 [0.38–0.64] |
| C2 Perch + CORAL + aumento | não | 0.0% / 2.4% | 13.8% / 5.9% | 1.1% / 1.7% | 0.578 [0.50–0.66] | 0.523 [0.41–0.62] | 0.522 [0.44–0.61] |
| C2 Perch + CORAL + aumento | sim | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.578 [0.50–0.66] | 0.523 [0.41–0.62] | 0.522 [0.44–0.61] |
| D1 Perch afinado na GPU (emenda A1) | não | 0.0% / 0.2% | 4.1% / 3.8% | 5.2% / 6.3% | 0.627 [0.33–0.87] | 0.569 [0.45–0.67] | 0.555 [0.45–0.66] |
| D1 Perch afinado na GPU (emenda A1) | sim | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.627 [0.33–0.87] | 0.569 [0.45–0.67] | 0.555 [0.45–0.66] |

Clusters com Aedes avaliados: abuzz 6, humbug 5, mosquitosong 4. Com tão poucos clusters, diferenças de AUROC entre candidatos têm IC largo e se sobrepõem.

## FPR em ruído de fonte nunca vista

| Candidato | Portão | HumBug (ruído do próprio corpus) | ESC-50 fora | Pete fora |
|---|---|---|---|---|
| A0 Perch normalizado (referência v3) | não | 2/1895 = 0.1% [0.0–0.7] | 5/235 = 2.1% [0.4–4.3] | 0/200 = 0.0% (bootstrap degenerado) |
| A0 Perch normalizado (referência v3) | sim | 0/1895 = 0.0% (bootstrap degenerado) | 0/235 = 0.0% (bootstrap degenerado) | 0/200 = 0.0% (bootstrap degenerado) |
| A1 Perch + CORAL | não | 40/1895 = 2.1% [0.9–4.9] | 1/235 = 0.4% [0.0–1.3] | 4/200 = 2.0% [0.5–3.6] |
| A1 Perch + CORAL | sim | 0/1895 = 0.0% (bootstrap degenerado) | 0/235 = 0.0% (bootstrap degenerado) | 0/200 = 0.0% (bootstrap degenerado) |
| A2 Perch + DANN | não | 2/1895 = 0.1% [0.0–0.7] | 7/235 = 3.0% [0.9–5.1] | 1/200 = 0.5% [0.0–1.6] |
| A2 Perch + DANN | sim | 0/1895 = 0.0% (bootstrap degenerado) | 0/235 = 0.0% (bootstrap degenerado) | 0/200 = 0.0% (bootstrap degenerado) |
| B1 bandas ≥ 300 Hz | não | 1/1895 = 0.1% [0.0–0.3] | 1/235 = 0.4% [0.0–1.3] | 0/200 = 0.0% (bootstrap degenerado) |
| B1 bandas ≥ 300 Hz | sim | 0/1895 = 0.0% (bootstrap degenerado) | 0/235 = 0.0% (bootstrap degenerado) | 0/200 = 0.0% (bootstrap degenerado) |
| B2 batimento | não | 0/1895 = 0.0% (bootstrap degenerado) | 12/235 = 5.1% [2.6–8.1] | 0/200 = 0.0% (bootstrap degenerado) |
| B2 batimento | sim | 0/1895 = 0.0% (bootstrap degenerado) | 12/235 = 5.1% [2.6–8.1] | 0/200 = 0.0% (bootstrap degenerado) |
| C1 bandas ≥ 300 Hz + aumento | não | 0/1895 = 0.0% (bootstrap degenerado) | 1/235 = 0.4% [0.0–1.3] | 1/200 = 0.5% [0.0–1.8] |
| C1 bandas ≥ 300 Hz + aumento | sim | 0/1895 = 0.0% (bootstrap degenerado) | 0/235 = 0.0% (bootstrap degenerado) | 0/200 = 0.0% (bootstrap degenerado) |
| C2 Perch + CORAL + aumento | não | 25/1895 = 1.3% [0.2–1.8] | 1/235 = 0.4% [0.0–1.3] | 1/200 = 0.5% [0.0–1.7] |
| C2 Perch + CORAL + aumento | sim | 0/1895 = 0.0% (bootstrap degenerado) | 0/235 = 0.0% (bootstrap degenerado) | 0/200 = 0.0% (bootstrap degenerado) |
| D1 Perch afinado na GPU (emenda A1) | não | 2/1895 = 0.1% [0.0–0.7] | 4/235 = 1.7% [0.4–3.4] | 0/200 = 0.0% (bootstrap degenerado) |
| D1 Perch afinado na GPU (emenda A1) | sim | 0/1895 = 0.0% (bootstrap degenerado) | 0/235 = 0.0% (bootstrap degenerado) | 0/200 = 0.0% (bootstrap degenerado) |

## Custo de calibração local

Mediana de 5 repetições; só os mapas de probabilidade e o limiar são recalibrados com k arquivos por classe do corpus fora do treino; clusters tocados saem da avaliação. Recall / FPR em outros.

| Candidato | Portão | k | Abuzz | HumBug | MosquitoSong |
|---|---|---|---|---|---|
| A0 Perch normalizado (referência v3) | não | 5 | 46.7% / 17.6% (n=5) | 0.0% / 0.0% (n=5) | 12.8% / 10.3% (n=5) |
| A0 Perch normalizado (referência v3) | não | 10 | 44.4% / 17.5% (n=3) | 13.7% / 17.2% (n=5) | 11.1% / 19.6% (n=5) |
| A0 Perch normalizado (referência v3) | não | 20 | indisponível | 18.0% / 16.9% (n=5) | 0.0% / 0.2% (n=5) |
| A0 Perch normalizado (referência v3) | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) | 0.3% / 0.0% (n=5) |
| A0 Perch normalizado (referência v3) | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 3.5% / 3.8% (n=5) |
| A0 Perch normalizado (referência v3) | sim | 20 | indisponível | 0.0% / 0.2% (n=5) | 0.0% / 0.0% (n=5) |
| A1 Perch + CORAL | não | 5 | 20.0% / 25.6% (n=5) | 4.3% / 3.7% (n=5) | 3.1% / 2.4% (n=5) |
| A1 Perch + CORAL | não | 10 | 0.0% / 8.0% (n=3) | 11.8% / 14.4% (n=5) | 8.1% / 16.2% (n=5) |
| A1 Perch + CORAL | não | 20 | indisponível | 2.2% / 3.8% (n=5) | 3.1% / 2.5% (n=5) |
| A1 Perch + CORAL | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) | 0.4% / 0.4% (n=5) |
| A1 Perch + CORAL | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 0.7% / 4.0% (n=5) |
| A1 Perch + CORAL | sim | 20 | indisponível | 0.0% / 1.6% (n=5) | 0.0% / 0.0% (n=5) |
| A2 Perch + DANN | não | 5 | 0.0% / 0.4% (n=5) | 9.2% / 5.9% (n=5) | 18.2% / 15.1% (n=5) |
| A2 Perch + DANN | não | 10 | 0.0% / 14.9% (n=3) | 2.2% / 1.1% (n=5) | 6.3% / 15.3% (n=5) |
| A2 Perch + DANN | não | 20 | indisponível | 3.0% / 0.9% (n=5) | 5.9% / 5.2% (n=5) |
| A2 Perch + DANN | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) | 1.0% / 0.2% (n=5) |
| A2 Perch + DANN | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 2.8% / 4.0% (n=5) |
| A2 Perch + DANN | sim | 20 | indisponível | 0.0% / 0.0% (n=5) | 0.3% / 0.5% (n=5) |
| B1 bandas ≥ 300 Hz | não | 5 | 53.3% / 9.3% (n=5) | 20.2% / 19.2% (n=5) | 17.0% / 11.6% (n=5) |
| B1 bandas ≥ 300 Hz | não | 10 | 40.0% / 9.3% (n=3) | 0.0% / 0.9% (n=5) | 16.1% / 12.0% (n=5) |
| B1 bandas ≥ 300 Hz | não | 20 | indisponível | 0.0% / 1.7% (n=5) | 11.5% / 5.5% (n=5) |
| B1 bandas ≥ 300 Hz | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) | 14.2% / 8.1% (n=5) |
| B1 bandas ≥ 300 Hz | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 8.0% / 7.3% (n=5) |
| B1 bandas ≥ 300 Hz | sim | 20 | indisponível | 0.0% / 0.0% (n=5) | 1.4% / 0.1% (n=5) |
| B2 batimento | não | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.4% (n=5) | 24.7% / 30.0% (n=5) |
| B2 batimento | não | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 27.6% / 29.4% (n=5) |
| B2 batimento | não | 20 | indisponível | 0.0% / 7.1% (n=5) | 1.7% / 2.6% (n=5) |
| B2 batimento | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.4% (n=5) | 24.7% / 30.0% (n=5) |
| B2 batimento | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 27.6% / 29.4% (n=5) |
| B2 batimento | sim | 20 | indisponível | 0.0% / 7.1% (n=5) | 1.7% / 2.6% (n=5) |
| C1 bandas ≥ 300 Hz + aumento | não | 5 | 38.5% / 17.1% (n=5) | 29.3% / 28.6% (n=5) | 26.3% / 30.9% (n=5) |
| C1 bandas ≥ 300 Hz + aumento | não | 10 | 0.0% / 11.2% (n=3) | 9.0% / 8.2% (n=5) | 8.3% / 6.1% (n=5) |
| C1 bandas ≥ 300 Hz + aumento | não | 20 | indisponível | 3.6% / 4.2% (n=5) | 5.5% / 1.2% (n=5) |
| C1 bandas ≥ 300 Hz + aumento | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) | 6.2% / 5.5% (n=5) |
| C1 bandas ≥ 300 Hz + aumento | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 6.6% / 2.3% (n=5) |
| C1 bandas ≥ 300 Hz + aumento | sim | 20 | indisponível | 0.0% / 0.3% (n=5) | 0.7% / 0.0% (n=5) |
| C2 Perch + CORAL + aumento | não | 5 | 0.0% / 14.8% (n=5) | 2.2% / 1.7% (n=5) | 5.6% / 6.0% (n=5) |
| C2 Perch + CORAL + aumento | não | 10 | 0.0% / 0.0% (n=3) | 3.9% / 4.4% (n=5) | 1.1% / 2.5% (n=5) |
| C2 Perch + CORAL + aumento | não | 20 | indisponível | 3.4% / 4.7% (n=5) | 0.0% / 0.0% (n=5) |
| C2 Perch + CORAL + aumento | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) |
| C2 Perch + CORAL + aumento | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 0.4% / 0.8% (n=5) |
| C2 Perch + CORAL + aumento | sim | 20 | indisponível | 0.0% / 0.9% (n=5) | 0.0% / 0.0% (n=5) |
| D1 Perch afinado na GPU (emenda A1) | não | 5 | 26.7% / 27.9% (n=5) | 4.6% / 3.4% (n=5) | 15.3% / 13.8% (n=5) |
| D1 Perch afinado na GPU (emenda A1) | não | 10 | 20.0% / 16.4% (n=3) | 13.7% / 11.9% (n=5) | 8.9% / 17.5% (n=5) |
| D1 Perch afinado na GPU (emenda A1) | não | 20 | indisponível | 8.9% / 11.7% (n=5) | 2.9% / 4.1% (n=5) |
| D1 Perch afinado na GPU (emenda A1) | sim | 5 | 0.0% / 0.0% (n=5) | 0.0% / 0.0% (n=5) | 1.1% / 2.2% (n=5) |
| D1 Perch afinado na GPU (emenda A1) | sim | 10 | 0.0% / 0.0% (n=3) | 0.0% / 0.0% (n=5) | 3.1% / 4.2% (n=5) |
| D1 Perch afinado na GPU (emenda A1) | sim | 20 | indisponível | 0.0% / 1.2% (n=5) | 0.7% / 0.8% (n=5) |

Braços indisponíveis: abuzz k=10: class_2_insufficient_files_without_consuming_test_clusters; abuzz k=20: class_2_insufficient_files_without_consuming_test_clusters.

## Leitura

- **Nenhuma hipótese tornou o detector utilizável.** Sem portão, o recall fica muito abaixo de 20% em quase todos os corpora; com portão, quase nenhum alerta é emitido (o mesmo padrão do 0/197 da v3). A calibração local com 5, 10 ou 20 arquivos não levou nenhum candidato às metas.
- **O fine-tuning na GPU (D1) não superou o Perch congelado (A0):** AUROC médio 0.584 contra 0.590, com IC sobrepostos.
- **CORAL piorou o ranking** (AUROC médio 0.535). Alinhar a distribuição de cada corpus também apaga diferenças reais de composição de espécies entre corpora.
- **O vencedor formal (B1) é indistinguível do A0** dentro dos IC por cluster; foi congelado só porque a regra exige um candidato descritivo.
- **A origem segue decodificável** em todas as representações não alinhadas, inclusive nos 9 descritores físicos do batimento. Restringir a banda reduz pouco.
- Diagnóstico consistente com o plano: com 16 clusters de Aedes, o problema é a diversidade de dados, não a arquitetura.


## A origem ainda é decodificável?

Classificador linear prevendo o corpus de origem (Abuzz, HumBug, MosquitoSong) entre arquivos de mosquito, 5 dobras por cluster. Acaso = 33,3%.

| Representação | Acurácia balanceada |
|---|---|
| perch_plain | 94.0% |
| perch_white | 91.9% |
| tabular_white | 72.7% |
| band_tabular | 69.6% |
| wingbeat | 69.1% |
| coral(perch_white) | 9.7% |

CORAL usa a identidade da origem como lote não rotulado. O valor **abaixo do acaso** com CORAL é um artefato esperado: cada origem é centralizada na mesma média, e os clusters de validação ficam deslocados para o lado oposto da média de treino da própria origem. Ele mostra que a informação de primeira e segunda ordem foi removida, não que a origem deixou de influenciar o modelo.

Alinhamento por grupo (MMD entre origens, só 'outro mosquito', menor = mais alinhado):

| Par | Perch normalizado (PCA-128) | Perch + CORAL |
|---|---|---|
| abuzz × humbug | 0.0478 | 0.0055 |
| abuzz × mosquitosong | 0.0685 | 0.0030 |
| humbug × mosquitosong | 0.0230 | 0.0026 |

## Oclusão de bandas (média nos 3 LODO)

Variação média absoluta de p(Aedes) ao substituir a faixa pela média do ajuste.

| Faixa | B1_band_tabular | C1_aug_band_tabular | R_tabular_full |
|---|---|---|---|
| 300-600 Hz | 0.0056 | 0.0056 | 0.0032 |
| 600-1000 Hz | 0.0039 | 0.0042 | 0.0031 |
| 1000-2000 Hz | 0.0043 | 0.0045 | 0.0034 |
| 2000-3400 Hz | 0.0037 | 0.0035 | 0.0032 |
| 0-300 Hz | — | — | 0.0032 |

## MosquitoSong por sexo (descritivo, sem portão)

| Candidato | Sexo | Aedes detectados | Outros chamados Aedes | AUROC |
|---|---|---|---|---|
| A0 Perch normalizado (referência v3) | F | 3/166 | 11/358 | 0.561 |
| A0 Perch normalizado (referência v3) | M | 15/152 | 39/413 | 0.564 |
| A1 Perch + CORAL | F | 1/166 | 5/358 | 0.542 |
| A1 Perch + CORAL | M | 4/152 | 5/413 | 0.526 |
| A2 Perch + DANN | F | 6/166 | 8/358 | 0.557 |
| A2 Perch + DANN | M | 13/152 | 50/413 | 0.545 |
| B1 bandas ≥ 300 Hz | F | 5/166 | 4/358 | 0.484 |
| B1 bandas ≥ 300 Hz | M | 6/152 | 8/413 | 0.518 |
| B2 batimento | F | 6/166 | 4/358 | 0.625 |
| B2 batimento | M | 0/152 | 27/413 | 0.335 |
| C1 bandas ≥ 300 Hz + aumento | F | 10/166 | 11/358 | 0.505 |
| C1 bandas ≥ 300 Hz + aumento | M | 5/152 | 4/413 | 0.512 |
| C2 Perch + CORAL + aumento | F | 0/166 | 7/358 | 0.563 |
| C2 Perch + CORAL + aumento | M | 3/152 | 4/413 | 0.512 |
| D1 Perch afinado na GPU (emenda A1) | F | 2/166 | 10/358 | 0.588 |
| D1 Perch afinado na GPU (emenda A1) | M | 14/152 | 46/413 | 0.534 |

Cada sexo tem só 2 indivíduos de *Ae. aegypti* no MosquitoSong.

## Seleção e congelamento

Regra: `protocol_v4.json` › `selection`. Resultado: No candidate met the per-corpus development targets; highest mean LODO AUROC frozen as descriptive fallback.

Candidato congelado: `B1_band_tabular` (sem portão), reajustado em todo o desenvolvimento. SHA-256 dos pesos: `B1_band_tabular` `ae8f821864e1e11b…`, `B2_wingbeat` `7b620439bb40cf53…`. Congelado em 2026-10-22T23:26:11.256981+00:00, antes de existir qualquer dado do teste novo.

## Limites

- Sem teste novo, nenhum número desta página sustenta afirmação de melhora ou de generalização.
- 16 clusters com Aedes no desenvolvimento, 2 a 4 deles na calibração de cada dobra; o limiar é escolhido com pouquíssimos positivos independentes.
- Bootstrap por cluster com modelos fixos; não inclui a variabilidade de retreino. Zero eventos geram intervalos degenerados.
- CORAL e A1/C2 são transdutivos: exigem um lote não rotulado do aparelho/corpus avaliado.
- A normalização espectral por gravação (herdada da v3) já remove boa parte de uma equalização estática; o efeito do aumento vem sobretudo de ruído e reverberação externos.
- O fine-tuning do Perch foi executado pela emenda A1 (GPU AMD RX 9070 XT, ROCm 6.4), com a correção numérica da emenda A2. Kernels de GPU não são bit a bit determinísticos; as seeds fixam inicialização e ordem dos dados (as perdas da dobra HumBug repetiram exatamente ao ser refeita).
- Sexo e temperatura não entraram como variáveis do modelo: só existem no MosquitoSong e em parte do HumBug.
- Resultados da v2/v3 permanecem como estão; esta é uma rodada nova, não uma correção delas.

## Reprodução

```bash
python -m reanalysis.v4_augment        # atributos da cópia aumentada (OpenSLR 28 em data/external/openslr28/)
python -m reanalysis.v4_train dev      # LODO, ablação do portão, calibração local
python -m reanalysis.v4_train select
python -m reanalysis.v4_train diagnostics
python -m reanalysis.v4_train final    # congela o candidato antes do teste novo
python -m reanalysis.v4_report
```
