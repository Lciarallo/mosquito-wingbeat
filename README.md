# Detector acústico ternário de mosquitos

O ponto de entrada deste projeto é o notebook didático de [treinamento, teste e aplicação](notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb). Ele explica o problema, mostra a avaliação de referência e conduz uma nova execução quando os dados locais estão preparados.

## O que está incluído

- O notebook com explicações, métricas de referência, matriz de confusão e aplicação do modelo.
- Os módulos essenciais que implementam o protocolo de divisão por grupos, treinamento aninhado, avaliação e inferência.
- Metadados compactos de janelas e vínculos de gravações em `data/metadata/`.
- Testes do protocolo ternário em `tests/`.

## Executar

1. Crie um ambiente Python e instale `requirements.txt`.
2. Abra o notebook e execute as células na ordem.
3. Para treinar, disponibilize os áudios originais e o cache de descritores localmente. O manifesto e o mapa de grupos versionados indicam o alinhamento esperado.
4. Altere `RUN_NEW_TRAINING` para `True`; a rodada executa seleção interna, teste OOF por grupos e congela os artefatos antes das previsões externas.
5. Defina `AUDIO_PATH` para um WAV e execute a célula de aplicação depois do treino e do teste.

Áudios, caches, modelos de rodada e demais saídas regeneráveis permanecem fora do Git. O treino completo pode levar bastante tempo; abrir o notebook não inicia treinamento automaticamente.

## Limites dos resultados

A avaliação de referência é retrospectiva e mede desempenho nos grupos deste corpus. O critério amostral de falsos positivos entre outras espécies não foi atingido. Os resultados não demonstram validação independente em novos locais, indivíduos, microfones ou condições de campo.
