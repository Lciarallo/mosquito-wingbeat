# Atualização da apresentação fornecida - 14/10/2026

O PDF fornecido pelo usuário, `Apresentacao_Ensemble_v3_Impeccable.pdf`, foi atualizado em uma versão separada de 13 slides: `output/pdf/Apresentacao_Ensemble_v4_Auditoria_Atualizada_2026_10_14.pdf`. O arquivo original foi preservado por SHA-256. A estrutura usa formato 16:9, fontes Inter, a paleta verde do original e as fotografias já incorporadas ao PDF (mosquitos e placa Arduino).

## Conteúdo atualizado

- Comparação congelada v1-v4 recalculada diretamente de `results/species_nested_v4_2026_10_14/comparison_metrics.csv`.
- Tabela v4 de acurácia, acurácia balanceada, Macro-F1 estrito, cobertura e acerto nas emissões aceitas, distinguindo Dryad, D3 e o áudio privado. A tabela não aplica gate adicional de presença.
- Correções espectrais, taxonômicas, MIL e de prior da v4, sem afirmar ganho de generalização ou invariância exata a ruído aditivo.
- Auditoria dos scripts neurais/heurísticos recebidos do Drive, separada das rodadas de ensemble: 26/28 espécies no lote conhecido, mas 6/17 no lote pareado. Não foi reproduzida a alegação de 100%.
- Contagens Brasil emulado, Brasil bruto e global bruto, F0 mediano de 600 Hz e oito intervalos de atividade do padrão de enxame. Não são acurácias biológicas por janela.
- Contraexemplo de dependência entre janelas, intervalos bootstrap descritivos e limites da inferência em campo.
- A referência a 215 testes aprovados e uma falha é o snapshot auditado às 16:20 UTC, não uma certificação dos arquivos posteriormente revisados no Drive. Não foi executado novo treinamento ou nova suíte para esta edição documental.
- Fontes locais e links clicáveis para os dois relatórios científicos já publicados no Drive.

## Verificação

As 13 páginas foram renderizadas a 1440 px e inspecionadas. O texto continua selecionável, com fontes incorporadas e sem spans fora dos limites da página. `verify_freeze` da v4 e `verify_history` passaram antes e depois. Os 1.239 arquivos do snapshot histórico e os documentos anteriores também foram conferidos; nenhuma modificação foi necessária nos pesos, resultados congelados, firmware, instalador, README ou MODEL_VALIDATION protegidos.

O construtor Python, imagens preservadas, gráficos, fontes de métricas e manifesto SHA-256 estão no pacote de reprodução. O pacote mantém os caminhos relativos para permitir a regeneração numa cópia separada do projeto com as dependências PDF instaladas; não contém áudio privado ou pesos neurais.
