# Mosquitos pelo som — revisão visual IFSC

Revisão de 13/10/2026 para Luiz Eduardo do Nascimento Ciarallo.

[Apresentação editável no Google Slides](https://docs.google.com/presentation/d/1n48d31ukSs-jxyxYAOrLTg8wtTj0B-YNSTBzyE425b4/edit?usp=drivesdk) · [PDF atual](Apresentacao_Mosquitos_IFSC_Resumida_Visual.pdf).

São **18 slides**, com notas de fala para aproximadamente **19 min 10 s**. A sequência cobre contextualização, literatura, implementação, resultados e discussão. A revisão inicial tinha 15 slides; a pedido do usuário, foram acrescentadas as frequências publicadas das espécies nos slides **4–6**. Os resultados principais e seus limites continuam explícitos.

O material inclui cinco fotografias, quatro gráficos, quatro tabelas nativas e um diagrama editável da CNN. O mestre, os 11 layouts, os fundos e a família tipográfica do modelo oficial IFSC foram mantidos. Todos os slides foram conferidos em renderização de 120 dpi; o verificador estrutural não apontou problemas. O registro está em [verificacao_google_slides.json](verificacao_google_slides.json).

As tabelas de frequências foram adaptadas da **Tabela 1 de Sinka et al. (2021), página 1853** ([artigo HumBug](https://doi.org/10.1111/2041-210X.13663)). Cada uma das 15 espécies tem uma observação publicada nos slides, com valores separados para machos/fêmeas e condições de idade/temperatura. Não foram agregados estudos diferentes em uma faixa artificial. Os valores variam; uma frequência isolada não garante identificação da espécie. As 25 observações originais, inclusive as repetidas por espécie, estão em [Tabela_Frequencias_Sinka_2021_Completa.csv](Tabela_Frequencias_Sinka_2021_Completa.csv), com autoria das medições e [proveniência](Tabela_Frequencias_Sinka_2021_Fontes.json). As espécies dessa tabela não equivalem às vinte classes da rede.

As fotografias dos mosquitos vêm do [CDC](https://www.cdc.gov/mosquitoes/about/about-mosquitoes-in-the-united-states.html): *Aedes aegypti*, *Anopheles quadrimaculatus* e *Culex tarsalis*. A placa é o **Nano 33 BLE Sense Rev2**, fotografado pela [Arduino](https://store.arduino.cc/products/nano-33-ble-sense-rev2). As imagens são referências visuais dos organismos e da placa, e não registros de coleta ou montagem deste experimento. Créditos, URLs, dimensões e hashes estão em [assets/fontes_imagens.json](assets/fontes_imagens.json); cada slide contém fontes completas nas notas.

O áudio e os gráficos são os dados reais já utilizados no projeto. A revisão não realizou treinamento, compilação, instalação de firmware ou teste em placa física. A avaliação externa de 17 arquivos continua identificada como retrospectiva; não representa uma nova coleta independente.

[conteudo_editavel.json](conteudo_editavel.json) registra os textos, notas e fontes da versão atual. [revisar_slides_visuais.py](../../trabalho_ifsc/revisar_slides_visuais.py) prepara a revisão inicial de 15 slides; [aplicar_tabela_frequencias.py](../../trabalho_ifsc/aplicar_tabela_frequencias.py) prepara a transcrição e inclusão das tabelas. Os scripts geram requisições para revisão contra suas capturas de origem; não se autenticam nem enviam alterações ao Drive.

A versão visual anterior está preservada em [PDF com 15 slides](Apresentacao_Visual_Anterior_15_Slides.pdf). A primeira entrega foi preservada em [PDF com 20 slides](Apresentacao_Anterior_20_Slides.pdf) e em uma [cópia editável no Google Slides](https://docs.google.com/presentation/d/1rjeW8qhEkPqsG4Cj-arbjVnVoO6OiJWXy3EO4i0s7Kg/edit?usp=drivesdk). Os arquivos de `../ifsc_2026_10_11/` são o material anterior.
