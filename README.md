# mosquito-wingbeat — "Zzzz… é Aedes?"

Detector ternário de mosquitos pelo som: dada uma gravação curta, o som é **ruído**, **outro mosquito** ou **Aedes aegypti**?

## O que há neste repositório

| Pasta | Conteúdo |
|---|---|
| `notebooks/detector_ternario_treinamento_teste_aplicacao.ipynb` | **O caderno é a única fonte da verdade.** Vai do áudio bruto ao resultado (atributos, validação por grupo, 9 modelos, SVM final, curvas, interpretabilidade, aplicação a um WAV), com código, explicação, o porquê de cada passo e glossário. Não importa nenhum script do repositório. |
| `apresentacao/` | Os slides ("Zzzz… é Aedes?", HTML e PDF), alinhados ao caderno. |
| `data/` | Os áudios (não publicados no Git) e os metadados em `data/metadata/` (os dois que o caderno lê e a tabela de janelas do Arduino que ele gera na §19). |

## Como reproduzir

1. Ambiente: `pip install -r requirements.txt` (para GPU, instale o `torch` da sua plataforma em pytorch.org) e o programa `ffmpeg` no sistema.
2. Dados: o caderno **não baixa** nada; ele lê as pastas de `data/`. A §2 do caderno traz a tabela de onde obter cada coleção e uma célula que confere, arquivo por arquivo, se tudo está no lugar.
3. Abra o caderno e execute tudo (*Run All*). A primeira execução reconstrói os atributos a partir dos áudios, treina os modelos (as redes neurais usam a GPU, se houver) e grava caches em `cache_notebook/` (ignorada pelo Git). Execução completa do zero, medida em uma GPU AMD (ROCm 6.4): cerca de 37 minutos. Apague `cache_notebook/` para refazer tudo.

## Resultado principal (SVM em dois estágios, validação por grupo, 3 repetições)

AUROC 0,83 · precisão 76% · detecção 52% (limiar 0,5) · AUROC Aedes × Culex no mesmo aparelho 0,78.

**Limites:** os resultados são retrospectivos (dados já conhecidos do projeto), não um teste cego. O modelo só viu Aedes de dois datasets e **não** reconhece Aedes de aparelhos novos (AUROC 0,51, igual ao acaso). O resultado também depende de como os poucos aparelhos do Funchal são agrupados: com 8 e 48 kHz juntos (agrupamento usado antes, por um erro de comparação de texto) a AUROC seria 0,86. Detalhes e números nas §11, §14 e §17 do caderno.

## Primeiro teste com um aparelho nosso: Arduino de laboratório (§19)

Sete gravações contínuas de *Aedes aegypti* feitas com um Arduino em duas armadilhas (05/11/2026). Uma regra baseada nos harmônicos do bater de asas separa, segundo a segundo, os trechos com mosquito (259 s) dos trechos só com ruído ambiente (137 s); nenhum arquivo é só ruído. Os modelos atuais, **sem retreino**, avaliados nesses trechos: o SVM final acerta quase sempre quando avisa "Aedes" (precisão 98%, 1% de alarmes no ruído), mas só reconhece 34% dos segundos de Aedes (melhor na Armadilha 1 que na 2), e o estágio "tem mosquito?" chama o ruído do Arduino de mosquito. Os 9 modelos ficam entre 0,60 e 0,74 de AUROC Aedes × ruído. A tabela de janelas rotuladas está em `data/metadata/arduino_lab_2026_11_05_janelas.csv`.

Treinando com o Arduino de 05/11 e provando em outros dias (21/10 e 30/09, nunca vistos no treino), a detecção continua baixa: no limiar 0,5 nenhum modelo passa de 1/3 dos segundos de Aedes, embora os alarmes falsos fiquem em 0–3%. O SVM sobe de AUROC 0,65 para 0,86; LSTM e Bi-LSTM pioram (§19.3). Com um limiar próprio para o Arduino, escolhido só com 05/11, a melhor combinação (regressão logística, sem Arduino no treino) acha 48% dos segundos de Aedes de outubro com 1% de alarmes no ruído; o SVM retreinado, 41% com 4% (§19.4). Ajustando os pesos das redes neurais já treinadas (ajuste fino dos dois estágios com 05/11), a LSTM chega a 53% e o MLP a 48%, com 1% de alarmes; o preço é um estágio 2 mais propenso a chamar outros mosquitos de Aedes (na base, alarmes em outros mosquitos sobem de ~12% para 28–40% no limiar 0,5), porque o Arduino só tem Aedes (§19.5).
