# Trabalho IFSC: Mosquitos pelo som

**Luiz Eduardo do Nascimento Ciarallo**  
Computação Científica para a Indústria - IFSC  
Experimentos preparados em 11/10/2026; apresentação revisada em 13/10/2026.

## Apresentar à turma

- **Apresentação principal no Google Slides:** [abrir a versão editável no modelo oficial completo do IFSC](https://docs.google.com/presentation/d/1n48d31ukSs-jxyxYAOrLTg8wtTj0B-YNSTBzyE425b4/edit?usp=drivesdk). A revisão de 13/10 tem **18 slides**, notas de fala, cinco fotografias reais (mosquitos e Nano 33 BLE Sense Rev2), quatro gráficos e um diagrama editável da CNN. Inclui três tabelas de frequências, acrescentadas à versão visual inicial de 15 slides. O mestre, os 11 layouts, os fundos e a família tipográfica originais do IFSC foram mantidos; as caixas de conteúdo foram adaptadas para imagens e texto mais curto.
- **PDF atual:** [`output/ifsc_2026_10_13_visual/Apresentacao_Mosquitos_IFSC_Resumida_Visual.pdf`](../output/ifsc_2026_10_13_visual/Apresentacao_Mosquitos_IFSC_Resumida_Visual.pdf), exportado da versão nativa e conferido nos 18 slides.
- Tempo sugerido: **19 min 10 s**, dentro dos 15 a 20 minutos pedidos. Tempos, explicações técnicas e fontes completas ficam nas notas de fala.
- Os arquivos PDF, HTML, roteiro e ZIP de `output/ifsc_2026_10_11/` são o **material anterior de 20 slides**, preservado como arquivo. O HTML anterior continua funcionando offline, com setas para navegar, **F** para tela cheia e **N** para o roteiro; ele não corresponde à revisão visual atual.
- [Cópia editável anterior à revisão visual](https://docs.google.com/presentation/d/1rjeW8qhEkPqsG4Cj-arbjVnVoO6OiJWXy3EO4i0s7Kg/edit?usp=drivesdk).

O material atual cobre os cinco tópicos do enunciado: contextualização (slides 2-7), abordagens e correlatos (8), proposta implementada (9-10), resultados (11-14) e discussão (15-17). O slide 15 apresenta a placa alvo e a ausência de validação física; o slide 18 contém referências resumidas. As fotos vêm do CDC e da loja oficial Arduino, com fontes e legendas das espécies. Os resultados externos continuam identificados como retrospectivos.

Os slides **4–6** adaptam a **Tabela 1 de Sinka et al. (2021), página 1853**, do [artigo HumBug](https://doi.org/10.1111/2041-210X.13663): 15 espécies, uma observação publicada por espécie, machos/fêmeas e condições de idade/temperatura. A [transcrição completa](../output/ifsc_2026_10_13_visual/Tabela_Frequencias_Sinka_2021_Completa.csv) preserva todas as 25 observações originais e as respectivas fontes. Não se trata de uma frequência universal de cada espécie nem das vinte classes completas da rede.

## Implementação neural executada

`trabalho_ifsc/experimento_cnn.py` implementa uma CNN em PyTorch com quatro convoluções e duas camadas lineares: **130.900 parâmetros treináveis**. Entrada com dois canais espectrais, 64 bandas e 55 quadros; saída de vinte espécies. Usa o frontend fixo de `large_audio_model.py`, que está incluído no pacote com os mesmos bytes do repositório.

Houve treinamento real em GPU/ROCm. A validação selecionou a época 6 entre 12 executadas. Amostragem equilibrada por espécie e gravação, AdamW, dropout e parada antecipada. Os pesos foram salvos antes da inferência no teste. A partição histórica de calibração ficou reservada e não foi usada; esta rodada não ajusta rejeição ou limiar de presença.

| Modelo novo | Gravações corretas no teste | Acerto por gravação |
|---|---:|---:|
| CNN | 105/190 | 55,26% |
| Regressão logística de referência | 136/190 | 71,58% |

Os dois modelos usam as mesmas gravações e o mesmo frontend. O modelo linear usa média e desvio temporal dos canais, com padronização ajustada no treino. A CNN não demonstrou vantagem nesse protocolo. Na primeira rodada, a CNN acertou **0/17 arquivos** externos, de cinco espécies. Depois foram treinados dois modelos revistos, descritos abaixo.

Resultados detalhados em `results/ifsc_cnn_2026_10_11/`: pesos `cnn.pt`, referência `baseline.joblib`, protocolo, partições, curvas, matrizes de confusão e previsões por gravação. `freeze.json` identifica os arquivos por SHA-256. `verification.json` registra as verificações executadas.

**Limites:** o teste interno é histórico e já tinha orientado pesquisas anteriores; os 17 áudios externos já haviam sido examinados no projeto. Não houve nova coleta cega. Separação por arquivo/grupo/hash declarado nos metadados não prova independência de indivíduos/sessões ou ausência de recortes deslocados. A validação tem 19 das 20 espécies. A CNN sempre escolhe uma classe, inclusive para ruído; ela não valida presença nem falsos alarmes. Esta rodada não autoriza concluir confiabilidade em campo. Os resultados da CNN maior e do ensemble clássico são históricos e estão identificados separadamente nos slides.

Nenhum áudio privado, dataset completo, credencial, firmware ou instalador acompanha o pacote. O código do trabalho não modifica pesos anteriores, firmware ou instaladores. Não houve teste em placa física.

## Nova rodada com alterações

A pedido do usuário, foram treinadas duas redes adicionais em `trabalho_ifsc/comparar_melhorias.py`, preservando os resultados originais e as mesmas partições:

- **CNN com frequência preservada:** quatro convoluções; dezesseis posições de frequência no final, contra quatro na primeira rede; 224.036 parâmetros.
- **MLP profunda:** média e desvio temporal dos mesmos canais; padronização ajustada no treino; três camadas ocultas de 256, 128 e 64 unidades; 109.140 parâmetros.

Dropout e regularização dos pesos foram aumentados; a taxa de aprendizado foi reduzida e recebeu agendamento por cosseno. A seleção passou a usar perda por gravação na validação, equilibrada por espécie. **A MLP foi selecionada pela validação antes da inferência no teste**, com época 32 de 40. A CNN revista escolheu a época 7 de 15.

| Modelo | Acertos por gravação | Acerto global | Acerto balanceado | F1 macro |
|---|---:|---:|---:|---:|
| CNN inicial | 105/190 | 55,26% | 66,43% | 62,49% |
| CNN revista | 132/190 | 69,47% | 75,82% | 71,46% |
| MLP profunda, selecionada | 132/190 | 69,47% | 77,44% | 77,06% |
| Referência linear | 136/190 | 71,58% | 77,22% | 74,51% |

As redes revistas ganharam **27 acertos e 14,21 pontos percentuais** sobre a primeira CNN. Não superaram a referência linear em acerto global; não foi demonstrada superioridade estatística. No conjunto externo retrospectivo, cada uma acertou **3/17 arquivos (17,65%)**: dois *Anopheles minimus* e um *Anopheles stephensi*. Não acertaram Aedes ou Culex nesses arquivos. Melhora nesses conjuntos não prova transferência para uma campanha nova.

As evidências estão em `results/ifsc_melhorias_2026_10_11/`: pesos das duas redes, scaler, protocolo, seleção, curvas, previsões e matrizes de confusão internas/externas. O pacote inclui a primeira rodada e as melhorias, junto com notebook executado.

## Demonstrar a rede com os pesos prontos

No repositório, execute na raiz:

```bash
.venv/bin/python trabalho_ifsc/inferir_melhorias.py caminho/arquivo.wav
```

No pacote extraído, use um Python com as dependências de `trabalho_ifsc/requirements-inferencia.txt` e execute:

```bash
python trabalho_ifsc/inferir_melhorias.py caminho/arquivo.wav
```

O arquivo precisa ter ao menos 0,992 s. A rotina converte para mono e 16 kHz, usa janelas completas sem repetir ou preencher o áudio e mostra as três classes de maior escore. A cauda incompleta é descartada. **Esses escores não comprovam presença nem confiabilidade da identificação.** Para a aula, o gráfico e as tabelas permitem demonstrar o resultado sem depender de captura ao vivo.

## Notebook e repetição

Abra `trabalho_ifsc/CNN_Mosquitos_IFSC.ipynb`. Ele já foi executado e contém verificações das partições, do hash dos pesos e das contagens de acertos. O notebook lê resultados congelados; abrir ou executar suas células não inicia treinamento.

Para treinar outra rodada, é necessário o checkout completo de `mosquito-wingbeat`, com os CSVs e caches de áudio preparados. Os dados grandes não estão neste pacote:

```bash
.venv/bin/python trabalho_ifsc/experimento_cnn.py --output results/ifsc_nova_rodada
```

O programa recusa sobrescrever uma pasta existente. A inferência externa desta entrega está em `trabalho_ifsc/avaliar_transferencia.py`; sua repetição requer os arquivos públicos já presentes no repositório e uma pasta de resultados adequada. Não chamar uma repetição dos mesmos áudios de novo teste independente.

## Modelo oficial, edição e fontes

Modelo: [Documentos úteis do IFSC](https://www.ifsc.edu.br/en/documentos-uteis), link **Modelo de Apresentação de Slides do IFSC**; pacote baixado de `https://ifsc.edu.br/documents/d/guest/slides?download=true`. Foram usados os fundos JPEG autênticos das páginas 1 e 4 do modelo `slides 16_9 horizontal.pptx`, junto com a fonte Oswald. A fonte foi obtida do repositório Google Fonts, com pesos estáticos derivados do arquivo variável, e sua licença OFL acompanha o pacote.

`modelo_oficial_IFSC_16x9_original.pptx` é o **modelo original sem o conteúdo deste trabalho**. A apresentação principal preenchida está no Google Slides, criada por importação do arquivo oficial completo e edição de uma cópia nativa. O PDF atual está em `output/ifsc_2026_10_13_visual/`; os PDFs e HTML de 11/10 preservam a versão anterior.

O conteúdo atual está em `output/ifsc_2026_10_13_visual/conteudo_editavel.json`. `trabalho_ifsc/revisar_slides_visuais.py` prepara requisições para editar os elementos nativos do Google Slides a partir da captura da versão anterior; não se autentica nem envia alterações ao Drive. As fotografias e seus créditos estão em `output/ifsc_2026_10_13_visual/assets/`. O registro de revisão está em `output/ifsc_2026_10_13_visual/verificacao_google_slides.json`.

O JSON de `output/ifsc_2026_10_11/` e o gerador `trabalho_ifsc/preparar_material.py` correspondem ao **material anterior de 20 slides**. Executar o comando abaixo recria esse material anterior, não a revisão atual. Fundos e fontes estão em `trabalho_ifsc/assets/`:

```bash
python trabalho_ifsc/preparar_material.py --assets trabalho_ifsc/assets
```

O gerador requer as bibliotecas de análise do projeto, ReportLab e nbformat. Ele lê as evidências desta rodada e os caches do repo para produzir o gráfico do áudio real. A regeneração também requer os resultados das duas rodadas. No pacote offline, os gráficos já calculados e a apresentação pronta estão disponíveis; os caches grandes não foram incluídos. O gerador também recria o notebook sem saídas: execute o notebook novamente após regenerar.

Referências principais:

- Mukundarajan et al. (2017). *Using mobile phones as acoustic sensors for high-throughput mosquito surveillance*. https://doi.org/10.7554/eLife.27854
- Fanioudakis, Geismar e Potamitis (2018). *Mosquito wingbeat analysis and classification using deep learning*. https://doi.org/10.23919/EUSIPCO.2018.8553542
- Supratak et al. (2024). *MosquitoSong+: A noise-robust deep learning model for mosquito classification from wingbeat sounds*. https://doi.org/10.1371/journal.pone.0310121
- Dryad: https://doi.org/10.5061/dryad.98d7s
- BioDCASE, origem dos arquivos D3 já examinados: https://zenodo.org/records/19095788
- Código e dados dos autores MosquitoSong+: https://github.com/akaraspt/mosquitosongp

As avaliações dos autores, os resultados históricos do projeto e a nova rodada são experiências distintas. Não houve reprodução integral dos treinamentos dos artigos nesta entrega.
