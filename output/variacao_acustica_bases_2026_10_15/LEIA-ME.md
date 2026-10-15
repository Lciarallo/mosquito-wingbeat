# Variação acústica entre bases — 15/10/2026

Análise descritiva de 1.513 arquivos locais, sem retreinar ou recalibrar modelos. Todos os 60 grupos Aedes do estudo ternário (20 Dryad, 40 HumBug) e os 1.322 arquivos dos dez grupos MosquitoSong foram processados. Controles negativos: primeiros 12 grupos por fonte/classe, em ordem lexicográfica; nenhum controle escolhido pelo resultado. Adicionais: todos os 15 WAVs UFRGS disponíveis; amostra anteriormente documentada de CDMSC (12 arquivos); somente os 2 WAVs de exemplo publicados por Torres. Não representa os 300 tensores de Torres nem o CDMSC inteiro.

## Método

Decodificação, média dos canais, reamostragem com filtro polifásico a 8 kHz, remoção de média e filtro Butterworth passa-faixa de quarta ordem 200–2000 Hz. RMS em blocos de 100 ms. Variação = P90 menos P10 do RMS em dB, dentro de blocos completos sem sobreposição de 1 s, com análise adicional em 2 s. Mediana dos blocos por arquivo, dos arquivos por grupo e dos grupos por fonte. Arquivos menores que o bloco são excluídos daquela métrica; constam nos CSVs. MosquitoSong Aedes: 269/362 arquivos elegíveis em 1 s, 88/362 em 2 s. Não concatenamos trechos diferentes para criar continuidade artificial.

Valores são variação relativa de intensidade, não SNR, frequência fundamental, quantidade de mosquito ou precisão do classificador. Incluem voo e ruído de fundo. O filtro causal pode introduzir transiente no início; não foi feita separação de fontes. Comparações com e sem trechos iniciais podem ser úteis em investigação posterior.

Grupos são unidades do corpus, não indivíduos comprovadamente independentes. UFRGS foi agrupado por arquivo, pois a independência biológica não foi demonstrada. Torres: duas amostras, nenhuma extrapolação ao dataset completo.

## Resultados

`resumo.csv` contém todos os resultados. Aedes em blocos de 1 s, mediana por grupo: MosquitoSong 2,80 dB; HumBug 5,95; Dryad 7,10; UFRGS 6,78; CDMSC amostrado 8,31; dois WAVs Torres 7,80. Em 2 s a ordenação entre MosquitoSong, HumBug e Dryad persiste (3,67; 7,17; 9,18).

O padrão de maior estabilidade do MosquitoSong se repete na amostra ampliada. Entretanto, dentro de Dryad e HumBug, os grupos Aedes perdidos têm mediana de variação maior que os acertos: Dryad 8,93 vs 7,02 dB; HumBug 6,22 vs 5,75. Logo, a análise não sustenta que maior variação por si só cause o acerto. Outros mosquitos MosquitoSong também são estáveis (2,38 dB), sugerindo característica da aquisição/fonte, sem provar causalidade.

ESC-50 entrou como controle de ruído, não contém Aedes. Não atribuímos novas métricas de detecção a UFRGS, CDMSC ou Torres nesta análise; o vínculo com acerto/erro é somente o resultado congelado por grupo da cascata para Dryad, HumBug e MosquitoSong.

## Figuras e rastreabilidade

`variacao_por_base.png`: distribuição das medianas dos grupos Aedes em 1 s.

`espectrogramas_bases.png`: um exemplo por fonte, escolhido pela proximidade à mediana de variação entre arquivos de pelo menos 2 s; recorte central de 2 s, sem seleção visual ou por acerto. STFT Hann 1024 amostras, passo 128 a 8 kHz. Intensidade espectral relativa ao pico de cada trecho, mesma escala -65 a 0 dB; não comparar nível absoluto ou SNR pela cor.

`manifesto.csv`, `metricas_por_arquivo.csv` (com SHA256), `metricas_por_grupo.csv`, `acertos_erros.csv`, `exemplos_figura.csv` e `analisar.py` permitem rastrear e reproduzir os resultados. `erros.json` vazio na execução concluída. Arquivo MP4 original decodificado via ffmpeg em memória, sem alterar o original. Nenhum áudio, treino, limiar ou apresentação foi modificado.

## Exclusão de escopo

O áudio privado do Drive é de enxame e foi retirado desta comparação por orientação do autor. Não participa de treino, calibração ou teste do detector atual. A versão anterior foi arquivada em tmp, sem apagar evidência histórica.
