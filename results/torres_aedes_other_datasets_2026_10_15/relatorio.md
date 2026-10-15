# Novo detector: avaliação nas outras bases — 15/10/2026

Foram avaliados os dois novos classificadores de árvores em boosting: **baseline sem Torres** e **modelo com Torres**, já treinados e congelados. Nenhum ajuste de pesos, frontend, hiperparâmetros ou limiares foi realizado nesta avaliação. Estes modelos são distintos da cascata/LSTM/Bi-LSTM da apresentação.

## Identificação de Aedes

| Fonte | Sem Torres no treino | Com Torres no treino | Unidade |
|---|---:|---:|---|
| Dryad | 0/4 (0,00%) | 1/4 (25,00%) | grupos Aedes reservados |
| HumBug | 3/8 (37,50%) | 2/8 (25,00%) | grupos Aedes reservados |
| MosquitoSong | 1/4 (25,00%) | 0/4 (0,00%) | grupos Aedes externos |
| UFRGS | 0/2 (0,00%) | 0/2 (0,00%) | grupos Aedes externos |
| Figshare | 10/1.867 (0,54%) | 260/1.867 (13,93%) | recortes Aedes de0,3s |

No Figshare os recortes compartilham um mesmo grupo de origem Aedes: 1.867 recortes não são 1.867 gravações/animais independentes. No agrupamento conservador por origem, com média dos atributos das três janelas de maior RMS, nenhum modelo reconheceu o único grupo Aedes. Os outros1.077 recortes têm rótulo `unknown_mosquito` e foram previstos, mas não tratados como negativos conhecidos de Aedes. Não há estimativa válida de falso positivo para essa fonte.

A melhoria com Torres no Dryad não se reproduziu no HumBug, MosquitoSong ou UFRGS. O ganho em parte dos recortes Figshare ainda deixa a maioria sem identificação Aedes. Os93,33% anteriormente obtidos em Torres descrevem positivos reservados daquela fonte; não representam uma melhora geral demonstrada em novas bases.

## Acerto ternário e alertas Aedes

| Fonte | Acerto ternário sem → com Torres | Alertas Aedes em negativos sem → com Torres |
|---|---|---|
| Dryad (122 grupos de teste) | 94/122 (77,05%) → 98/122 (80,33%) | 2/118 (1,69%) → 1/118 (0,85%) |
| HumBug (115 grupos de teste) | 97/115 (84,35%) → 97/115 (84,35%) | 3/107 (2,80%) → 5/107 (4,67%) |
| ESC-50 (56 grupos de ruído) | 46/56 (82,14%) → 47/56 (83,93%) | 0/56 → 0/56 |
| MosquitoSong (10 grupos) | 5/10 (50,00%) → 4/10 (40,00%) | 2/6 (33,33%) → 1/6 (16,67%) |

Os negativos de Dryad e HumBug incluem ruído e outros mosquitos; detalhes por classe estão em `group_predictions.csv`. No ESC-50 não há Aedes para medir sensibilidade. Acertos ternários do ESC-50 exigem prever ruído, enquanto falso alerta Aedes conta apenas a saída Aedes; prever outro mosquito em ruído é erro ternário, mas não alerta Aedes.

UFRGS possui dois grupos Aedes e um grupo `not_aedes`, que não distingue ruído de outro mosquito. Por isso só cabe avaliação binária de alerta Aedes. Ambos os modelos emitiram0/1 falso alerta no grupo negativo e reconheceram0/2 Aedes: acerto binário1/3. As15 gravações estão agrupadas em três categorias de origem, não em três indivíduos comprovadamente independentes.

## Protocolo e escopo

- Dryad, HumBug e ESC-50: somente grupos da dobra0 reservada no estudo novo; nenhum grupo do treino foi contado como teste. São293 grupos no total. Este relatório abre por fonte o mesmo teste anterior; não é um segundo teste independente nem uma nova validação cruzada completa.
- MosquitoSong e UFRGS: fontes fora do treino/validação dos dois novos modelos, mas conhecidas de análises anteriores. Por grupo foram utilizadas até três janelas reais de0,992s com maior RMS; janelas curtas de MosquitoSong não foram misturadas neste resultado principal.
- Figshare: sinais reais de0,3s, sem repetir, completar ou alongar artificialmente. O classificador de estatísticas mel mantém a mesma dimensão de entrada com sequências de duração variável. Entretanto, o treino usou0,992s; esta é uma avaliação adicional de transferência de duração, não equivalência ao protocolo original. O resultado por recorte não pode ser comparado diretamente ao resultado por grupo.
- UFRGS `not_aedes` é usado somente como negativo binário. Figshare `unknown_mosquito` fica sem verdade Aedes/não-Aedes e não entra nos denominadores de acerto/falso positivo.
- Áudio privado de enxame do Drive permanece excluído.

O frontend de compatibilização dos tensores Torres continua sendo uma aproximação documentada do estudo anterior. A redução do problema para atributos mel não demonstra equivalência física perfeita entre fontes. Os dados de Torres não contêm IDs biológicos e não fornecem todos os sinais brutos, portanto a ausência de duplicação entre esses tensores e outros sinais não pode ser comprovada integralmente.

## Verificação

As previsões externas foram reproduzidas a partir de modelos recarregados e atributos reextraídos do cache. Os arquivos usados externamente foram auditados contra o conjunto de hashes exatos e PCM alinhado de todo o corpus original (mais amplo que só o treino desta rodada), sem sobreposição detectada. Isso não comprova independência por indivíduo/sessão.

A avaliação inicial que misturava janelas longas/curtas na seleção dos três maiores RMS por grupo foi preservada em `results/torres_aedes_other_datasets_2026_10_15_mixed_duration_diagnostic/`. O relatório atual usa0,992s para MosquitoSong/UFRGS e reserva0,3s apenas para Figshare. Não houve seleção de modelo ou limiar com esses resultados.

Os modelos e o PDF anterior mantiveram seus hashes. A apresentação e seus arquivos no Drive não foram modificados.

## Artefatos

- `metrics.json`: métricas por fonte, separando grupo e recorte.
- `group_predictions.csv`: todas as previsões por grupo e escores.
- `figshare_clip_predictions.csv`: todas as2.944 previsões por recorte, incluindo rótulos desconhecidos sem incluí-los no acerto.
- `selected_external_windows.csv`: janelas externas usadas no agrupamento.
- `overlap_audit.json`: auditoria de duplicação exata/alinhada.
- `verification.json`: hashes preservados e reprodução das previsões.
- Execução: `.venv/bin/python evaluate_torres_aedes_datasets.py` (recusa sobrescrever saídas).
- Verificação: `.venv/bin/python verify_torres_aedes_datasets.py`.
