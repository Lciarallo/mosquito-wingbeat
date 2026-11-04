# mosquito-wingbeat — "Zzzz… é Aedes?"

Detector ternário de mosquitos pelo som: dada uma gravação curta, o som é **ruído**, **outro mosquito** ou **Aedes aegypti**?

## O que há neste repositório

| Pasta | Conteúdo |
|---|---|
| `notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb` | **O caderno é a única fonte da verdade.** Vai do áudio bruto ao resultado (atributos, validação por grupo, 9 modelos, SVM final, curvas, interpretabilidade, aplicação a um WAV), com código, explicação, o porquê de cada passo e glossário. Não importa nenhum script do repositório. |
| `apresentacao/` | Os slides ("Zzzz… é Aedes?", HTML e PDF), alinhados ao caderno. |
| `data/` | Os áudios (não publicados no Git) e dois arquivos de metadados que o caderno lê em `data/metadata/`. |

## Como reproduzir

1. Ambiente: `pip install -r requirements.txt` (para GPU, instale o `torch` da sua plataforma em pytorch.org) e o programa `ffmpeg` no sistema.
2. Dados: o caderno **não baixa** nada; ele lê as pastas de `data/`. A §2 do caderno traz a tabela de onde obter cada coleção e uma célula que confere, arquivo por arquivo, se tudo está no lugar.
3. Abra o caderno e execute tudo (*Run All*). A primeira execução reconstrói os atributos a partir dos áudios, treina os modelos (as redes neurais usam a GPU, se houver) e grava caches em `cache_notebook/` (ignorada pelo Git). Execução completa do zero, medida em uma GPU AMD (ROCm 6.4): cerca de 37 minutos. Apague `cache_notebook/` para refazer tudo.

## Resultado principal (SVM em dois estágios, validação por grupo, 3 repetições)

AUROC 0,83 · precisão 76% · detecção 52% (limiar 0,5) · AUROC Aedes × Culex no mesmo aparelho 0,78.

**Limites:** os resultados são retrospectivos (dados já conhecidos do projeto), não um teste cego. O modelo só viu Aedes de dois datasets e **não** reconhece Aedes de aparelhos novos (AUROC 0,51, igual ao acaso). O resultado também depende de como os poucos aparelhos do Funchal são agrupados: com 8 e 48 kHz juntos (agrupamento usado antes, por um erro de comparação de texto) a AUROC seria 0,86. Detalhes e números nas §11, §14 e §17 do caderno.
