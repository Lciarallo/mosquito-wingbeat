# Avaliação de gênero — 09/10/2026

Análise posterior do ensemble de espécies corrigido, somando probabilidades por gênero. Nenhum peso ou limiar foi ajustado; não foi treinado um classificador dedicado a gênero. A escolha é forçada entre Aedes, Anopheles, Culex e Culiseta; não há rejeição calibrada para gênero nem teste de presença/ruído.

| Avaliação | Corretas / gravações | Acurácia | Acurácia balanceada |
|---|---:|---:|---:|
| Dryad conhecido, dobras reservadas | 549/569 | 96,49% | 93,66% |
| Mesmos 17 arquivos externos, cinco espécies | 7/17 | 41,18% | 29,17% |

No teste externo: Aedes 0/5, Culex 0/4 e Anopheles 7/8. Todos os cinco Aedes foram classificados como Anopheles e os quatro Culex como Aedes. Culiseta não está representado nesse teste. Sempre escolher Anopheles acertaria 8/17 nesta amostra, contra 7/17 do modelo; isso não é um baseline universal, apenas mostra a limitação do resultado observado.

O resultado de gênero é forte dentro do Dryad conhecido, mas não se transferiu aos arquivos externos. Ainda não sustenta identificar o gênero em novas gravações com confiança. Os 17 arquivos podem ser recortes relacionados, sem independência de indivíduos/sessões estabelecida; não representam uma coleta confirmatória nova. Esta é uma análise posterior dos mesmos testes, não uma validação adicional independente.

[CSV de métricas](../results/species_nested_2026_10_09/genus_analysis/metrics.csv) · [Matriz externa](../results/species_nested_2026_10_09/genus_analysis/external_recording_calibrated_confusion.csv) · [Auditoria](../results/species_nested_2026_10_09/genus_analysis/audit.json) · [Protocolo de espécie e teste externo](ensemble_especies_calibracao_separada_2026_10_09.md).

Código: `evaluate_nested_genus.py`. As probabilidades internas são lidas dos arrays OOF com IDs de linha; as externas foram reexecutadas com os modelos congelados. O script recusa sobrescrever uma análise concluída. Os hashes dos modelos e do código de inferência foram conferidos antes e depois da execução.
