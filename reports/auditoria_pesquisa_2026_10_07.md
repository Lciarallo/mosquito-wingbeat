# Auditoria da pesquisa sobre classificação de mosquitos — 07/10/2026

## Parecer

A pesquisa reúne evidências úteis e os relatórios mais recentes acertam ao manter o detector experimental. Não está demonstrada a identificação confiável de *Aedes aegypti* nas condições reais do usuário. Há erros na análise LOSO que foram corrigidos nesta revisão; os números internos dos artigos, as nossas adaptações e os diagnósticos externos continuam sendo evidências diferentes.

Esta revisão inspecionou relatórios, protocolos, código de preparação/avaliação, testes e previsões salvas. Conferiu fontes primárias e repetiu a variante LOSO afetada. Não refez todos os treinamentos nem coletou um teste independente novo. As alterações locais que já existiam no repositório foram preservadas.

## O que está sustentado

- As reexecuções de avaliação dos autores têm previsões, pesos e verificações de integridade. Os testes conferem os 85,26% DR-BioL, 79,39% DAT e 92,78% MosquitoSong+ registrados. Isso não equivale a repetir o treinamento ou validar um novo aparelho.
- A auditoria do notebook MosquitoSong+ encontra gravações e amostras sobrepostas nos arrays chamados treino/teste. A falta do manifesto de treinamento dos H5 impede afirmar que todos os pesos usaram exatamente essas partições. O resultado não quantifica a inflação da métrica.
- Os diagnósticos externos e o áudio privado não demonstram confiabilidade. A gravação privada tem apenas a referência positiva informada pelo usuário; 107 janelas do mesmo áudio não são 107 mosquitos independentes e não estimam falsos positivos.
- A separação por gravação, a calibração reservada e a inclusão de inválidos no denominador são decisões corretas. Identidade de gravação, contudo, não garante independência de indivíduos, sessões ou derivados publicados em outra base.

O artigo [MosquitoSong+ (PLOS ONE, 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121) apresenta 91,8% e 93,3% em experimentos distintos. Não é correto escolher a referência mais próxima dos nossos pesos sem identificar o experimento correspondente. O [DR-BioL](https://arxiv.org/abs/2510.00346) e o [baseline BioDCASE 2026](https://arxiv.org/abs/2603.20118) discutem dependência de características do domínio e dificuldade de transferência. Isso apoia investigar mudança de aquisição; não prova a causa dos erros de um modelo específico nosso.

## Erros e limitações corrigidos

| Achado | Consequência | Correção |
|---|---|---|
| Dryad declarado como óptico no LOSO | `hgb_acoustic_only` excluía dados acústicos de celulares ao avaliar HumBug/MosquitoSong | Incluído `dryad` nas fontes acústicas e repetida a variante |
| Afirmação de que usar só áudio acústico piorava e deveria ser descartado | Conclusão baseada em filtro de fontes incorreto | Retirada a conclusão; a repetição foi registrada separadamente |
| Estatísticas de toda a fonte avaliada usadas por `*_source_norm` | A avaliação requer acesso sem rótulos ao lote de teste e não representa inferência isolada | Rotulada como transdutiva; retirada a equivalência com coletar só ruído ambiente |
| Normalização combinada com retirada de Figshare | A comparação muda dois fatores ao mesmo tempo | Explicitado que não estima o efeito isolado da normalização |
| Relatório LOSO afirmava reprovação por precisão/F1 com IC sem calculá-los | Faltava evidência específica nessa análise | Recalculadas essas métricas e ICs por gravação a partir das previsões |
| Seleção de características completava até 40 com variáveis discordantes se faltassem elegíveis | O código podia violar sua própria regra em outro corpus | Retorna somente variáveis elegíveis; erro explícito quando nenhuma existe |
| Reexecução podia substituir a pasta histórica | Perda de rastreabilidade | Nova pasta e recusa a sobrescrever resultados já existentes |

O [registro oficial Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.98d7s) identifica gravações acústicas de celulares para o DOI 10.5061/dryad.98d7s. O corpus óptico citado em Fanioudakis é outro conjunto. A diferença de aparelho, ambiente, composição biológica e seleção de arquivos ainda pode explicar parte da mudança entre fontes; não foi isolada experimentalmente.

A correção da seleção de características é preventiva: nesta base há 59, 77 e 68 variáveis elegíveis ao retirar Dryad, HumBug e MosquitoSong, respectivamente, sem Figshare. Como todas superam 40, não atribuímos os resultados históricos ruins ao preenchimento indevido.

## Repetição da variante afetada

Mantidos o estimador, a semente, as características, os rótulos, o limiar 0,5 e os pesos de amostragem. Corrigido apenas o conjunto declarado acústico. Os resultados são retrospectivos: essas fontes já foram examinadas.

| Fonte retirada | AUC histórica | AUC corrigida | Precisão *Ae. aegypti* | Sensibilidade | Especificidade | Macro F1 | Acurácia balanceada |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dryad | 0,694 | 0,694 | 3,95% | 0,25% | 99,69% | 48,90% | 49,97% |
| HumBug | 0,412 | 0,633 | 3,48% | 8,28% | 85,60% | 47,18% | 46,94% |
| MosquitoSong | 0,458 | 0,584 | 15,47% | 4,58% | 92,81% | 45,67% | 48,69% |

AUC corrigida: IC 95% por gravação de 0,580–0,683 no HumBug e 0,324–0,797 no MosquitoSong. A melhora do AUC não resolve o desempenho das decisões no limiar utilizado. Como as fontes presentes neste cache são acústicas, a variante corrigida equivale, neste corpus, a excluir Figshare; ela não compara sensores acústicos contra ópticos.

Artefatos: [métricas da repetição](../results/loso_corrected_2026_10_07/loso_metrics.csv), [protocolo e hashes](../results/loso_corrected_2026_10_07/protocol.json). Os resultados de 03/10 permanecem intactos.

## Reanálise dos 30 resultados históricos

O script `audit_research.py` verifica se AUC, sensibilidade, especificidade e acurácia balanceada recalculadas coincidem com a tabela anterior. A seguir calcula precisão, F1, matriz de confusão e intervalos com 2.000 reamostragens de gravações completas. Inclui também os três resultados corrigidos, identificados em outra coluna. As entradas e o código têm SHA-256 registrados.

Exemplo: `lr_baseline`, que teve AUC 0,671 no HumBug, apresenta precisão de apenas **7,40%**, sensibilidade **23,08%** e macro F1 **49,45%** no limiar 0,5. Os ICs por gravação são 4,17–11,04% para precisão e 16,91–29,77% para sensibilidade. No MosquitoSong, sua sensibilidade de 88,89% acompanha especificidade de 25,86%: detectar mais positivos não significa discriminar bem as duas classes.

O LOSO contém 24.405 janelas Dryad em 613 grupos, 16.353 HumBug em 592 e 2.739 MosquitoSong em apenas 10. Figshare tem 1.867 janelas agrupadas em uma única família de origem, refletindo a falta de identidade dos indivíduos. Muitos segmentos não substituem diversidade de gravações e indivíduos.

No bootstrap MosquitoSong do `lr_baseline`, 18/2.000 réplicas não têm positivos e uma não tem negativos. Esses casos ficam explicitamente registrados como indefinidos para as métricas correspondentes. Os percentis são condicionais às réplicas definidas; poucos grupos e identidade biológica incompleta limitam sua interpretação. Não são intervalos de garantia para novos ambientes.

Nenhum candidato histórico ou da repetição satisfaz simultaneamente os mínimos de precisão, sensibilidade, macro F1 e especificidade de 60% nas três fontes. Isso pode ser observado nas estimativas pontuais, sem usar um IC estreito artificial por janela. O LOSO excluiu áudios inválidos antes da avaliação: estas métricas não medem o comportamento completo do sistema diante de falhas de aquisição. Precisão ainda depende da prevalência e do conjunto de negativos desta amostra.

Resultados: [métricas](../results/research_audit_2026_10_07/metrics.csv), [intervalos](../results/research_audit_2026_10_07/intervals.json), [contagem de fontes/classes](../results/research_audit_2026_10_07/source_counts.csv), [manifesto](../results/research_audit_2026_10_07/manifest.json).

## Melhorias para o próximo experimento

1. Fixar a tarefa e a aquisição: presença acústica, espécie e *Ae. aegypti* são alvos diferentes. Definir duração real, microfone, distância, ambiente, número de indivíduos e sessões; identificar a espécie por uma referência independente.
2. Organizar o catálogo por gravação original, indivíduo e sessão, com parentesco entre cortes, canais e versões. Auditar duplicatas de bytes e PCM entre fontes antes de separar dados. Identidades ausentes devem continuar como limitação explícita.
3. Usar seleção interna por fontes/sessões de desenvolvimento. Comparar um baseline linear e os novos candidatos com frontend fixo; calibrar em grupos reservados. Se houver normalização no destino, comparar separadamente um lote de calibração do sensor e adaptação transdutiva, declarando a informação disponível em cada caso.
4. Congelar código, pesos, limiares e critérios de inclusão antes de obter o conjunto confirmatório. Incluir novos *Ae. aegypti*, outras espécies próximas, ruído persistente e períodos reais sem mosquitos. Selecionar por procedência, não por acertos do modelo.
5. Dimensionar a amostra pela incerteza entre indivíduos/sessões, com métricas por espécie e sensor. Anotar eventos para medir sensibilidade por voo e falsos alarmes por hora. Uma coleta piloto permite planejar o tamanho da confirmação; não deve servir simultaneamente para escolher e confirmar o modelo.

## Verificação e reprodução

Foram aprovados **123 testes** em `.venv/bin/python -m pytest -q tests`, incluindo os hashes dos pesos/códigos originais, as métricas das reexecuções dos autores e regressões da nova auditoria. Houve avisos de depreciação do joblib com NumPy; a suíte passou. Testes de código e consistência não demonstram desempenho em campo.

```bash
.venv/bin/python -m pytest -q tests
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python loso_generalization.py hgb_acoustic_only
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python audit_research.py
```

Os dois scripts de análise recusam sobrescrever suas pastas de saída. Para uma nova execução, escolha outra pasta nas constantes `OUT` e, na auditoria, aponte `CORRECTED` para essa repetição. Os áudios/features locais são necessários; os manifestos registram seus hashes. Não foi criada uma nova distribuição para Arduino. Os módulos das avaliações congeladas, firmware e pesos anteriores foram preservados.
