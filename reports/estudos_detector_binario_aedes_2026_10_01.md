# Estudos sobre detecção acústica de Aedes aegypti — 01/10/2026

Busca em artigos originais e versões dos autores. O objetivo é decidir se um som contém Aedes aegypti. As métricas são relatadas pelos artigos; seus modelos não foram reproduzidos nesta pesquisa bibliográfica.

| Referência | Tarefa e resultado relatado | Limite para nosso objetivo |
| --- | --- | --- |
| Fernandes, Cordeiro e Recamonde-Mendoza, 2021, Computers in Biology and Medicine | CNN binária: Aedes aegypti versus demais espécies. Acurácia 97,65%; sensibilidade 88,49%; especificidade 98,81%; precisão 90,63%; F1 89,29%. Médias de dez dobras. | Dados Abuzz com limpeza manual de trechos ruidosos; separação por gravação/indivíduo não demonstrada no protocolo descrito. |
| Joelianto et al., 2024, Ecological Informatics | CNN binária: Aedes aegypti versus Culex quinquefasciatus. Teste em ovitrampa com 170 exemplos: acurácia 91,76%; sensibilidade 98,82%; precisão 86,59%. | O negativo é uma espécie específica; não cobre todos os outros mosquitos e ruídos. |
| Paim et al., 2023, preprint arXiv | CNN residual para presença de Aedes, incluindo ruídos e outras espécies. Autores relatam acurácia, precisão, recall e F1 superiores a 80% nos cenários examinados. | Janelas sobrepostas e divisão aleatória em dez dobras após extração de features; risco de compartilhar gravação entre dobras. |
| Supratak et al., 2024, PLOS ONE, MosquitoSong+ | Classificação de quatro espécies: acurácia 89,3% em dados internos, 82,5% no HumBug e 80,5% externos. | São resultados multiclasse; não uma taxa binária de Aedes, nem um teste de domínio totalmente excluído do treino. |

## Interpretação dos estudos

**Fernandes:** é a referência mais diretamente alinhada ao classificador binário amplo. Usa espectrogramas Mel e CNN. Apenas 11,14% dos exemplos são Aedes: os 97,65% gerais não significam detectar essa proporção dos Aedes. O ensemble sobe a sensibilidade a 96,82%, mas sua precisão cai a 68,19%. Não atribuímos vazamento comprovado ao artigo sem auditar seu código e seus grupos. Fontes: [artigo publicado](https://doi.org/10.1016/j.compbiomed.2020.104152), [texto integral dos autores](https://arxiv.org/pdf/2008.09024), [código indicado no artigo](https://github.com/ComputerNetworks-UFRGS/CNN_Aedes_Aegypti_Classifier).

**Joelianto:** treino com gravações de celular em recipientes; teste em ovitrampa de laboratório. Na matriz binária, 13 de 85 Culex recebem Aedes: 15,3% de falsos positivos, calculados da tabela 5. O trabalho demonstra discriminação útil entre essas duas espécies, com erro remanescente considerável nos negativos. Fontes: [DOI](https://doi.org/10.1016/j.ecoinf.2024.102495), [texto integral disponibilizado pelos autores](https://www.researchgate.net/publication/377600573_Convolutional_neural_network-based_real-time_mosquito_genus_identification_using_wingbeat_frequency_A_binary_and_multiclass_classification_approach).

**Paim:** combina Abuzz, gravações UFRGS e ESC-50, com pré-processamento a 8 kHz e espectrograma STFT. É útil como referência de arquitetura residual e treinamento com ruído. O protocolo descrito exige uma reprodução com grupos preservados para comparação justa conosco. O dataset chamado D5 neste artigo é acústico; não é o D5 óptico do BioDCASE. Fontes: [preprint integral](https://arxiv.org/html/2306.10091), [artefatos dos autores](https://github.com/ComputerNetworks-UFRGS/Aedes_Detector_Smartphone_App_Paper).

**MosquitoSong+:** opera em áudio bruto com CNN 1D e mistura ruído real e variações de volume no treino. Os autores informam separação de gravações de mosquito e ruído. Porém, treinam com parcelas dos três corpora: isso não demonstra generalização para um corpus inteiro ausente do treino. O resultado de 93,3% corresponde a espécie e sexo em gravações internas com ruído sobreposto; para espécie e sexo externos cai a 67,3%. Fonte: [artigo integral e métodos](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121).

## Aplicação ao projeto

Minha proposta é reproduzir a CNN binária de Fernandes como referência e comparar uma CNN residual com uma CNN 1D treinada com ruído e variações de volume. A tarefa e o limiar precisam permanecer exclusivamente binários: Aedes versus não-Aedes.

O experimento deve separar arquivos antes de segmentar, agrupar gravações do mesmo indivíduo/sessão quando os metadados permitirem e reservar aparelho/local inteiros. Outros mosquitos, especialmente Aedes albopictus e Anopheles, e ruídos ambientais precisam estar entre os negativos. Avaliar sensibilidade, especificidade, precisão, acurácia balanceada e rejeições; nenhum limiar pode ser escolhido pelo teste.

Os resultados dos artigos não demonstram confiabilidade no enxame em caixa de vidro gravado pelo Nano 33 BLE Sense. Esse domínio precisa de várias gravações independentes de Aedes e negativos com o mesmo sistema, além de novos testes externos após congelamento. Esta busca não treinou, recalibrou nem compilou modelos.
