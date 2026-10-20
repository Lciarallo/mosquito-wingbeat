# Emendas ao protocolo v2

Toda mudança depois do congelamento (`protocol_freeze.json`) é registrada aqui com data, motivo e se foi feita antes ou depois de ver resultados.

## A1 — 2026-10-20 — antes de qualquer resultado de modelo
- **Descritor f0**: a soma harmônica linear em 150–1000 Hz travava em sub-harmônicos e no ruído de manuseio/sala (picos de 100–180 Hz nas gravações HumBug em copo). Passou a ser soma harmônica no domínio log (produto harmônico, 3 harmônicos) em **250–1000 Hz**, faixa biológica do batimento de asa. Só afeta 1 dos 89 descritores tabulares; o log-mel não muda. A mudança foi motivada apenas por inspeção de espectros de 20 arquivos, sem nenhuma métrica de classificação.
- **Quadros do log-mel**: STFT sem padding central, o que dá exatamente os 40 × 59 declarados no protocolo.

## A2 — 2026-10-20 — depois de E1, antes de abrir E2/E3/S2/S3
Motivo: leitura de Paim et al. (arXiv 2306.10091, UFRGS), artigo de origem do corpus externo UFRGS.
- **Análise secundária S4 (exploratória, declarada agora):** teste de estresse contra sons enganosos, que é o requisito R3 do artigo. Tons puros e harmônicos sintéticos de 300–900 Hz, zumbido elétrico de 60/120 Hz e ruído rosa, gerados com seed fixa e passados pelo mesmo frontend. Métrica: fração de arquivos chamados de Aedes ou de "outro mosquito". Os sinais são **sintéticos** e nunca entram em treino nem em métricas primárias.
- **Diagnóstico D1 (exploratório):** importância por oclusão de bandas mel no modelo final do E3, para ver se a decisão usa a faixa do batimento de asa (~300–1.200 Hz) ou bandas de dispositivo (<250 Hz).
- **UFRGS:** a planilha `identified_recordings.xlsx` dos autores mostra que cada arquivo vem de um recipiente diferente (`id_container`) e que cada classe foi gravada em datas distintas: Aedes F em 13/02, Aedes M em 15/02, não-Aedes em 07, 08, 18 e 19/02. Com n=15, os resultados UFRGS são reportados como contagens descritivas, sem IC por bootstrap. O manifesto congelado não muda.
- O artigo avalia com 10 dobras aleatórias sobre segmentos com 50% de sobreposição. Isso tem o mesmo tipo de vazamento corrigido aqui, por isso as acurácias do artigo **não** servem de referência comparável.
