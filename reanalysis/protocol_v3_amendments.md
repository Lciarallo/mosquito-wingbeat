# Emendas ao protocolo v3

## A1 — 2026-10-20 — antes de qualquer modelo
- **Clusters provisórios do BioDCASE** (`bdc|D|S`) não podem unir sessões reais. Um componente que contém um arquivo com metadado conhecido (v2 ou HumBugDB) herda só esse cluster. Na primeira versão, esse vínculo fundia todas as sessões HumBug numa só, deixando 1 cluster de "outro mosquito".
- **HumBugDB `sound_type = audio`** (600 arquivos) e mosquitos sem espécie (787) foram excluídos por rótulo indefinido, como no v2.
- **Quem é o BioDCASE D1** (estágio 2): 3.542 dos 4.065 arquivos do D1 são cópias do HumBugDB.

## A2 — 2026-10-20 — continuação antes de qualquer classificador v3

O Claude foi interrompido antes de salvar características ou treinar classificadores. A continuação está especificada em `protocol_v3_continuation.json`; o protocolo original e seu hash permanecem preservados.

- Dois clusters provisórios cruzavam treino/teste: 286 arquivos de desenvolvimento entram em quarentena; o teste herdado não muda.
- O conjunto relacionado ao D3 não tem proveniência UFRGS confirmada para todos os arquivos. Separar os 15 componentes já avaliados e declarar a origem não confirmada dos restantes.
- Reservar previamente 107 arquivos de 67 clusters exclusivamente de ruído para avaliação complementar. A independência do teste passa a ser conferida por caminho, componente e cluster.
- Manter as metas de falsos positivos em 2% (outros mosquitos) e 1% (ruído), acrescentando sensibilidade mínima de 20% no teste final. Não usar o relaxamento de 5% do protocolo original. Distinguir metas pontuais de apoio estatístico e de validação de campo.
- Ajuste, escolha interna e calibração ficam separados por cluster; não refazer o treino usando a calibração. Escolher e congelar a configuração pelo desenvolvimento, antes das predições finais.
- Extração retomável, fila limitada e ganho RMS 0,05 (o script herdado já usa 0,05, embora seu texto diga RMS unitário). Usar média de embeddings por arquivo e registrar explicitamente a mudança para a MLP não linear.
- Adicionar diagnóstico só de duração e ablações sem/com normalização espectral. As métricas históricas permanecem como evidência retrospectiva em seu escopo.

## A3 — 2026-10-20 — correção técnica após o primeiro corpus de desenvolvimento

O diagnóstico auxiliar de duração falhou ao pontuar o Abuzz porque 67 arquivos MP4/M4A têm `duration_s` vazio no manifesto v3 herdado: seu construtor tentou apenas `soundfile.info`. O manifesto v2 versionado já tem as durações desses mesmos arquivos, obtidas com fallback para `ffprobe`. Recuperar somente esses valores para o diagnóstico, sem imputação por rótulo ou escore.

Os seis candidatos acústicos do primeiro corpus já estavam completos. Seus modelos, dados, características, seleções, limiares, partições e predições permanecem idênticos; somente a entrada do diagnóstico auxiliar muda. Nenhum resultado do teste reservado foi aberto. O código e as identidades anteriores foram preservados em `results/v3_continuation/source_before_duration_fix/`; a equivalência dos demais corpos de função e os hashes dos modelos são auditados antes da retomada.
