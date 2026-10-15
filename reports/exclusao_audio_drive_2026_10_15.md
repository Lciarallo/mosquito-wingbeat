# Exclusão do áudio de enxame do Drive

Por decisão do autor em 15/10/2026, o áudio privado do Drive é fora de escopo
para o detector de voo individual: não deve participar de treinamento,
normalização, seleção, calibração, teste ou comparação acústica corrente.
O motivo é a aquisição de um enxame, diferente do alvo desta avaliação;
não se trata de selecionar a exclusão em função do resultado do modelo.

## Análise de lacunas antes da alteração

- A referência ternária retida contém somente Dryad (24.405 janelas), HumBug
  (16.353) e ESC-50 (1.271), somando 42.029 janelas. Não contém o Drive.
- A LSTM lê exatamente essa população. O estudo auxiliar de espécies lê
  somente Dryad; o de infecção lê os tensores de Torres.
- Os carregadores ternários não tinham exclusão explícita para uma eventual
  inclusão futura do Drive nos metadados.
- A comparação acústica de 15/10 incluiu o Drive como diagnóstico descritivo.
  Essa comparação será regenerada sem ele.
- Resultados históricos são registros de execuções anteriores e não devem ser
  apagados ou reescritos como se o áudio jamais tivesse sido analisado.

## Identidade excluída

SHA256 dos bytes: `138fce9b9780f22013011849d793938282695a06647eb3467491cf95b6838eca`.
As cópias conhecidas são `drive_audio.wav`,
`data/raw/drive_test/drive_audio.wav` e
`tmp/private-audio-test/download.audio`.

Os arquivos originais e estudos históricos são preservados exclusivamente como
arquivo de procedência. Não constituem dados elegíveis da avaliação corrente.
Não é necessário retreinar os modelos já medidos, pois sua população não muda.
Os 90,03% da cascata e os resultados da LSTM permanecem os mesmos.

## Execução e validação

- A exclusão agora é explícita nos carregadores ternários clássico e neural e
  na avaliação por fonte. São reconhecidos os identificadores de procedência,
  os caminhos conhecidos e o SHA256 quando disponível nos metadados. Isso não
  proíbe bases públicas somente por estarem hospedadas no Google Drive.
- A comparação acústica corrente foi regenerada com 1.513 arquivos e sem o
  Drive nos manifestos, métricas, resumos ou imagens. A versão anterior está
  arquivada em `tmp/variacao_acustica_com_drive_antes_exclusao_2026_10_15.tar.gz`.
- O carregador atualizado reproduz exatamente as 42.029 linhas retidas da
  referência, sem Drive. A referência clássica e a neural têm zero linhas do
  áudio excluído. Evidência: `auditoria_exclusao_drive_2026_10_15.json`.
- Suíte completa: **317 testes aprovados** em 45,63 s, incluindo quatro
  contratos novos de exclusão e preservação do corpus. Log em
  `tmp/pytest_exclusao_drive_2026_10_15.log`.
- Os resultados congelados e seus snapshots não foram modificados. Não houve
  novo treinamento nem alteração de limiares. Os slides atuais já não usavam
  o áudio do Drive.

Uma cópia transformada e renomeada sem procedência não é identificável apenas
por esses metadados; requer nova auditoria de aquisição. Os scripts antigos
dedicados ao enxame continuam sendo referência histórica, fora do fluxo
atual documentado em `README_TORRES_2025.md`.
