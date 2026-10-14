# Experimento de transferência do notebook - 14/10/2026

Este pacote inclui fontes, 48 modelos/scalers/calibrações, features por janela e gravação, previsões, bootstrap, auditorias e um notebook executado. Preserve a hierarquia ao extrair numa cópia do repositório mosquito-wingbeat. Os arquivos históricos exigidos permanecem no projeto e na release de pesquisa já publicada; não são duplicados aqui.

O notebook original recebido é uma entrada criptograficamente identificada do frontend. Obtenha-o pelo link fornecido pelo usuário e salve em `external/notebook_reference_2026_10_14/Mosquito_v2.ipynb`: https://drive.google.com/file/d/1zl2sr73xdYJar1UH_crFndrEmNWJgA8Q/view . Confira o SHA em `source_manifest.json`. A cópia original tem saídas e áudio embutido, portanto não foi duplicada neste pacote; incluímos uma cópia sem outputs para inspeção de código. Os dados brutos de treino/cache e o áudio privado também não estão no pacote.

O CLI `predict_notebook_transfer.py` usa somente estes modelos experimentais, valida hashes e não troca a versão neural ativa. As probabilidades foram calibradas por gravação, não por janela. O notebook executado lê os resultados já produzidos. Para repetir treinamento, use uma pasta nova, pela API `train(out=..., workers=4, jobs=2)`; não sobrescreva resultados.

O ganho interno LFCC não se confirmou nos externos. São 232 testes aprovados e uma falha pré-existente v5. O histórico permaneceu intacto. Sem autorização ou compilação de firmware Arduino.
