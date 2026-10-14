# Relatório atualizado: classificação acústica de mosquitos

Atualizado em 30/09/2026. Modelo analisado: CNN multifonte de 3.325.397 parâmetros, para computador. Este documento consolida os resultados recentes; não substitui evidência experimental por uma promessa de desempenho.

## Conclusão principal

O modelo tem bons resultados em parte das gravações conhecidas, especialmente na seleção HumBug, mas permanece experimental. A generalização para a seleção UFRGS falhou, e o áudio privado informado pelo usuário como enxame de Aedes aegypti não recebeu identificação confiável. Não há acurácia demonstrada para o microfone do Arduino ou uma caixa de vidro. Nenhum firmware foi compilado ou liberado nesta atualização.

## Como interpretar os números

Acerto forçado é a proporção de janelas em que a espécie real ficou em primeiro lugar, sem rejeição. Cobertura é a proporção de janelas conhecidas em que o sistema emitiu uma espécie. Acerto nas emissões é calculado somente entre as janelas aceitas. São métricas diferentes; alta precisão com baixa cobertura não significa identificar todos os mosquitos. Janelas correlacionadas não equivalem a indivíduos independentes. As análises de pré-processamento são retrospectivas.


# Retreinamento com múltiplas fontes de áudio

**Execução concluída em 30/09/2026.** Os datasets adicionais foram baixados, auditados e usados em treinamento ou avaliação. A CNN melhorou em algumas tarefas e reduziu identificações erradas, mas **não ficou confiável em novos aparelhos/ambientes**. A UFRGS continua falhando e o enxame privado não foi identificado. Arduino, instalador e binários publicados não foram alterados.

## Dados baixados e uso efetivo

- **Figshare:** 2.988 WAV baixados e extraídos; 1.867 janelas de Aedes e 1.077 de rótulos taxonômicos não resolvidos entraram na adaptação. Culex pipiens & quinquefasciatus e Culiseta não foram inventados como espécies exatas. Todos os arquivos desse corpus ficaram no treino, pois não temos identidade da gravação longa/indivíduo para justificar um teste aleatório de recortes. Variantes e recortes não são indivíduos novos.
- **MosquitoSong:** todos os 1.324 WAV dos autores foram baixados; cortes e canais do mesmo grupo permaneceram juntos. 1.400 janelas foram usadas no treino, 987 na validação e 352 no desafio histórico de Aedes de cinco dias. Aedes de 21 dias é adaptação; as gravações de cinco dias antes examinadas continuam fora do treino. **Não é possível comprovar que os indivíduos de idades diferentes são distintos**: o resultado de MosquitoSong é retrospectivo e não valida transferência para indivíduos desconhecidos. Para as outras três espécies, machos ficaram no treino e fêmeas na validação, mantendo as idades e ambos os microfones do mesmo grupo juntos.
- **UFRGS:** 15 WAV baixados, todos fora do treino/validação/calibração: dez Aedes aegypti (271 janelas) e cinco referências **não-Aedes** (146 janelas). O nome `No_Aedes_aegypti` não identifica outra espécie nem comprova ausência de mosquito. Corrigimos a descrição inicial do catálogo que mencionava albopictus nessa seleção: não há essa pasta no repositório atual.
- **BioDCASE Development:** ZIP completo de 3.799.065.219 bytes baixado; MD5 `dbf35b9066e82cdee5461fca138f4986` conferido. Usado somente como teste auxiliar dos domínios D1–D4. De 4.238 arquivos do teste oficial nesses domínios, 3.893 foram retidos, com 36.963 janelas. Foram removidos 280 arquivos por identidade binária com treino ou duplicação no benchmark, e 65 menores que 0,3 s. A origem/identidade individual não está completamente mapeada; **não é um teste comprovadamente independente dos corpora anteriores**. D5 foi excluído porque a modalidade/origem de sensor não foi confirmada. O conjunto oficial Evaluation, sem rótulos públicos de espécie, não foi usado.
- **Synthetic Swarm:** 315 WAV e rótulos foram baixados. Não entraram na rodada: misturas derivam do Abuzz já usado, sem mapeamento completo dos ancestrais. Inserir uma mistura de origens de teste no treino contaminaria a avaliação. Mantivemos apenas a mistura sintética feita dentro do nosso treino, com fontes do treino.

Ao preparar as fontes adicionais foram detectados 42 arquivos binariamente duplicados e quatro curtos demais. O experimento reuniu 39.247 janelas, preservando todas as partições originais; 6.100 janelas foram adicionadas ao conjunto completo, **4.344 ao treino**. Treino válido total: **19.167 janelas**. [Contagens](../results/multisource/additional_counts.csv) · [Auditoria](../results/multisource/data_audit.json).

Fontes: [Figshare](https://figshare.com/articles/dataset/Mosquitoes_Bioacoustic_Features_-_A_Public_Dataset/11902125), [MosquitoSong](https://github.com/akaraspt/mosquitosongp), [UFRGS](https://github.com/ComputerNetworks-UFRGS/Aedes_Detector_Smartphone_App_Paper), [BioDCASE](https://zenodo.org/records/20478577), [Synthetic Swarm](https://github.com/duydinhthai27/synthetic-swarm-mosquito).

## Treinamento e controle

Mantivemos a CNN residual de **3.325.397 parâmetros** e ajustamos seus pesos a partir da versão grande anterior, sem substituí-la. Os lotes alternam 0,3 s (4.800 amostras reais) e 0,992 s (15.872); janelas longas podem ser cortadas para o lote curto, mas os áudios curtos não são repetidos, concatenados nem preenchidos. O frontend aceita esses comprimentos e usa o mesmo log-mel/contraste e normalização RMS. Isso permite aproveitar o Figshare sem fabricar continuidade acústica.

AdamW com taxa inicial 0,0002, amostragem equilibrando classe/domínio/origem, ruído e sobreposição da mesma espécie provenientes apenas do treino. Foram executadas **10 épocas**, com melhor perda de validação na **época 2**. O ajuste é exploratório; a validação já tinha sido usada na rodada anterior. Calibração ficou na partição antiga separada, com temperatura 0.908, limiar de presença 0.620 e de espécie 0.595. Uma emissão temporal exige pelo menos 2,976 s reais e concordância causal; recortes curtos isolados têm classificação forçada avaliada, sem fabricar identificação temporal.

Pesos e limiares congelados em `2026-09-30T22:33:01.763392+00:00`, SHA-256 `c754c0d48e5ceb43946bc6e3b1b4c3ff44ade2c693becc75be785ee540b43acf`. Nenhum limiar foi ajustado com os testes abaixo. A UFRGS e o áudio privado não entraram no treino. Os nomes de arquivo/rótulos/domínios não são entradas da rede; somente o áudio é processado.

## Comparação pareada

As duas CNNs foram aplicadas às **mesmas janelas**, com o mesmo controle temporal. Cada versão conserva sua própria calibração. Acerto sem rejeição força uma espécie; cobertura é a proporção de janelas com emissão; acerto nas emissões exclui as rejeições. `nan` indica ausência de emissões, não acerto de 0%.

| Corpus | Modelo | Janelas conhecidas | Acerto sem rejeição | Cobertura | Acerto nas emissões |
|---|---|---:|---:|---:|---:|
| dryad | large_cnn | 7845 | 62.7% | 31.0% | 91.1% |
| humbug | large_cnn | 383 | 89.0% | 66.6% | 95.7% |
| mosquitosong | large_cnn | 352 | 5.1% | 0.0% | Sem emissão |
| ufrgs | large_cnn | 271 | 5.5% | 8.5% | 0.0% |
| dryad | multisource_cnn | 7845 | 62.6% | 26.6% | 93.5% |
| humbug | multisource_cnn | 383 | 90.3% | 55.9% | 96.7% |
| mosquitosong | multisource_cnn | 352 | 40.3% | 0.9% | 66.7% |
| ufrgs | multisource_cnn | 271 | 6.6% | 3.7% | 0.0% |

A melhora em HumBug é pequena: 89,0% → 90,3%, enquanto a cobertura cai de 66,6% para 55,9%. No Dryad, o acerto sem rejeição permanece em cerca de 62,6%, com menor cobertura. **Mais precisão nas emissões não significa mais sons identificados.** Anopheles minimus no HumBug ainda tem recall zero em dez janelas de uma origem.

No MosquitoSong, 5,1% → 40,3% de acerto forçado, mas apenas três emissões, duas corretas e uma errada. Na UFRGS, 5,5% → 6,6% forçado; as dez emissões da nova versão são todas incorretas. A comparação não demonstra confiabilidade fora das fontes de treinamento. Nos cinco arquivos negativos da UFRGS, nenhuma versão emitiu Aedes; isso não valida as outras espécies emitidas.

Nas 212 janelas HumBug de mosquito sem rótulo resolvido para as 20 classes, emissões caíram de 34,9% para 11,3%. Não houve emissão de espécie nas 56 janelas de ruído Dryad ou 915 HumBug. Esses resultados amostrais não garantem ausência de falsos alarmes em campo.

![Comparação](assets/multisource_comparison.png)

### Benchmark BioDCASE auxiliar

| Domínio | Modelo | Janelas | Acerto sem rejeição | Acerto nas emissões |
|---|---|---:|---:|---:|
| biodcase_D1 | large_cnn | 27257 | 36.9% | 58.5% |
| biodcase_D3 | large_cnn | 5997 | 5.7% | 0.0% |
| biodcase_D2 | large_cnn | 1119 | 6.7% | 0.0% |
| biodcase_D4 | large_cnn | 2590 | 46.8% | 82.3% |
| biodcase_D1 | multisource_cnn | 27257 | 45.1% | 80.6% |
| biodcase_D3 | multisource_cnn | 5997 | 8.2% | 0.6% |
| biodcase_D2 | multisource_cnn | 1119 | 10.9% | 2.3% |
| biodcase_D4 | multisource_cnn | 2590 | 50.3% | 92.5% |

Há ganhos nos quatro domínios, mas D2/D3 permanecem muito ruins. Janelas de uma gravação são correlacionadas. Não comparar esses números diretamente com a classificação por clipe do desafio: nosso recorte de duração, exclusões e protocolo temporal diferem. A verificação de duplicatas binárias não garante ausência de recortes irmãos, versões reamostradas ou sobreposição de indivíduos.

## Cinco áudios inéditos após o congelamento

Após concluir o treino, procuramos novamente a fonte pública HumBug e selecionamos, **antes da inferência**, os cinco primeiros IDs de Aedes de grupos distintos, duração ≥2,976 s e ano ≥2018, excluindo todas as origens dos 307 trechos anteriores. Downloads novos passaram em CRC32 e SHA-256. Não houve duplicatas binárias com o treinamento ou os áudios BioDCASE já avaliados; uma comparação das formas de onda reamostradas também não encontrou equivalências completas. Isso não prova ausência de todo recorte deslocado no benchmark. Os cinco são da mesma campanha/país e tipo de microfone presentes no HumBug: **novas gravações, não um domínio inteiramente novo**.

| ID | Modelo | Janelas | Aedes como primeira classe | Emissões corretas | Emissões erradas |
|---|---|---:|---:|---:|---:|
| 220301 | large_cnn | 5 | 5 | 3 | 0 |
| 220301 | multisource_cnn | 5 | 5 | 3 | 0 |
| 220314 | large_cnn | 15 | 15 | 13 | 0 |
| 220314 | multisource_cnn | 15 | 15 | 10 | 0 |
| 220324 | large_cnn | 12 | 12 | 10 | 0 |
| 220324 | multisource_cnn | 12 | 12 | 8 | 0 |
| 220391 | large_cnn | 5 | 5 | 0 | 0 |
| 220391 | multisource_cnn | 5 | 5 | 0 | 0 |
| 220403 | large_cnn | 48 | 47 | 39 | 0 |
| 220403 | multisource_cnn | 48 | 47 | 20 | 0 |

Ambas as versões identificaram Aedes em **quatro de cinco arquivos**. Em 85 janelas, ambas acertaram 84 quando forçadas. Emissões corretas: CNN anterior **65/85**; nova **41/85**, sem emissões erradas em nenhuma versão. A nova ficou mais conservadora; este teste **não demonstra superioridade da nova CNN**. Não tratar 4/5 arquivos como estimativa da acurácia geral, nem 85 janelas como 85 mosquitos.


## Seu áudio privado de enxame

A referência de Aedes aegypti é a identificação informada pelo usuário, sem confirmação entomológica independente. Das 107 janelas: 99 SEM_EVIDENCIA, seis INCERTO e duas AUDIO_INVALIDO. Nenhuma identificação foi emitida, contra dez emissões erradas de Anopheles gambiae na CNN anterior. Aedes ficou em primeiro lugar em 46 janelas (antes 22), porém isso **não é identificação aceita**. O áudio permanece privado, fora do treino. Há redução de falsos rótulos, mas o enxame continua sem resolução de espécie.

## Decisão e reprodução

**Manter em avaliação offline.** Não substituir a versão anterior como se estivesse validada: ganhos são parciais, houve queda de cobertura e a generalização à UFRGS continua falhando. Antes de outro treinamento, será preciso definir como usar uma parte da UFRGS para adaptação sem chamar os mesmos arquivos de teste independente, e reservar outra campanha real de celular/microfone com espécie confirmada. Toda gravação usada para orientar novas decisões passa a ser histórica; a próxima rodada exige novamente áudio inédito após congelar pesos.

Scripts: `fetch_additional_audio.py`, `fetch_biodcase.py`, `prepare_multisource.py`, `train_multisource.py`, `prepare_biodcase_benchmark.py`, `evaluate_multisource.py`, `fetch_fresh_multisource_test.py`, `summarize_multisource.py`. O treinamento sobrescreve somente `results/multisource`; modelos anteriores são preservados. Inferência usa `test_large_audio.py --checkpoint results/multisource/models/multisource_cnn.pt ...`. Pesos e datasets ficam locais, ignorados pelo Git.



# Pré-processamento adicional: comparação com o modelo congelado

**Nenhum dos cinco métodos adicionais fez o modelo identificar o enxame privado como Aedes aegypti.** Há ganhos modestos em alguns dados e regressões em outros. Mantivemos o processamento padrão, pesos e limiares da CNN multifonte. Não houve retreinamento nem compilação para Arduino.

## O que já existia

Todos os sons já passam por conversão mono/16 kHz, remoção da média, normalização RMS por janela, espectrograma log-mel e contraste espectral local. Aumentar o volume isoladamente não deve resolver a falha: a amplitude é normalizada dentro da rede. O áudio privado continua fora do treino e sua espécie é a referência informada pelo usuário, sem confirmação independente nem anotação de eventos por instante.

## Métodos testados

1. Original: somente o frontend já existente.
2. `highpass_120`: Butterworth de ordem quatro, corte 120 Hz, para reduzir componentes de baixa frequência.
3. `bandpass_150_3500`: passa-faixa 150–3.500 Hz.
4. `bandpass_250_2500`: passa-faixa 250–2.500 Hz, mais restritivo.
5. `spectral_mild`: FFT 1.024/hop 256; potência de ruído estimada no percentil 20 dentro de cada janela; subtração com fator 0,75, ganho mínimo 0,5. Estimativa sem rótulos ou perfil de ruído externo. Tons de voo contínuos também podem entrar nessa estimativa e ser atenuados.
6. `bandpass_spectral`: passa-faixa 150–3.500 Hz seguido da redução espectral leve.

Filtros Butterworth em SOS, aplicados para frente e para trás com `padtype=None`, `padlen=0`, portanto não causais e destinados à análise offline. A reconstrução espectral usa overlap-add com janela Hann de extremidades não nulas; eventual cauda não coberta conserva as amostras originais. Não há repetição, preenchimento nem extensão da duração do som. Documentação primária: [SciPy butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html), [SciPy sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).

Os métodos foram definidos em um protocolo antes da execução. Mesmas janelas, máscaras de qualidade originais, temperatura, limiares e regra temporal para todos. Áudio originalmente inválido não é resgatado por um filtro. Código: `audio_preprocessing.py`, `evaluate_preprocessing.py`. Sem publicação do áudio privado, filtrado ou original.

## Seu áudio de enxame — 107 janelas

| Método | Aedes em primeiro lugar, sem rejeição | Identificações de Aedes emitidas | Emissões de outras espécies |
|---|---:|---:|---:|
| original | 46 | 0 | 0 |
| highpass_120 | 21 | 0 | 0 |
| bandpass_150_3500 | 4 | 0 | 0 |
| bandpass_250_2500 | 11 | 0 | 0 |
| spectral_mild | 42 | 0 | 0 |
| bandpass_spectral | 3 | 0 | 0 |

Mesmo quando Aedes é a primeira classe, os escores de presença/espécie e a concordância temporal precisam satisfazer os critérios congelados. Nenhum método os satisfez para emitir a espécie nesse arquivo. Os filtros mais restritivos reduziram drasticamente o número de janelas com Aedes em primeiro lugar; isso não demonstra que a identidade fornecida pelo usuário esteja errada.

## Testes pareados

Acerto sem rejeição força uma das 20 espécies; cobertura conta emissões em todas as janelas conhecidas; acerto nas emissões exclui rejeições. `nan` significa ausência de emissões. MosquitoSong é retrospectivo; UFRGS inteira ficou fora do treino. Os cinco Aedes que foram inéditos na rodada anterior já foram examinados e **agora são retrospectivos**, de uma campanha familiar no HumBug. Nenhum resultado abaixo é validação final de um filtro selecionado com estes mesmos testes.

| Método | Corpus | Janelas | Acerto sem rejeição | Cobertura | Acerto nas emissões |
|---|---|---:|---:|---:|---:|
| original | dryad | 7845 | 62.6% | 26.6% | 93.5% |
| original | humbug | 383 | 90.3% | 55.9% | 96.7% |
| original | mosquitosong | 352 | 40.3% | 0.9% | 66.7% |
| original | ufrgs | 271 | 6.6% | 3.7% | 0.0% |
| original | fresh_humbug_retrospective | 85 | 98.8% | 48.2% | 100.0% |
| highpass_120 | dryad | 7845 | 52.9% | 18.8% | 91.7% |
| highpass_120 | humbug | 383 | 68.7% | 21.4% | 91.5% |
| highpass_120 | mosquitosong | 352 | 5.7% | 0.0% | Sem emissão |
| highpass_120 | ufrgs | 271 | 1.8% | 0.0% | Sem emissão |
| highpass_120 | fresh_humbug_retrospective | 85 | 68.2% | 3.5% | 100.0% |
| bandpass_150_3500 | dryad | 7845 | 36.4% | 8.7% | 64.6% |
| bandpass_150_3500 | humbug | 383 | 32.1% | 1.3% | 0.0% |
| bandpass_150_3500 | mosquitosong | 352 | 1.1% | 0.0% | Sem emissão |
| bandpass_150_3500 | ufrgs | 271 | 0.0% | 0.4% | 0.0% |
| bandpass_150_3500 | fresh_humbug_retrospective | 85 | 8.2% | 0.0% | Sem emissão |
| bandpass_250_2500 | dryad | 7845 | 20.4% | 0.0% | Sem emissão |
| bandpass_250_2500 | humbug | 383 | 3.7% | 0.0% | Sem emissão |
| bandpass_250_2500 | mosquitosong | 352 | 0.6% | 0.0% | Sem emissão |
| bandpass_250_2500 | ufrgs | 271 | 2.6% | 0.0% | Sem emissão |
| bandpass_250_2500 | fresh_humbug_retrospective | 85 | 0.0% | 0.0% | Sem emissão |
| spectral_mild | dryad | 7845 | 60.8% | 24.1% | 93.4% |
| spectral_mild | humbug | 383 | 91.4% | 58.5% | 97.3% |
| spectral_mild | mosquitosong | 352 | 41.2% | 0.9% | 33.3% |
| spectral_mild | ufrgs | 271 | 5.5% | 5.2% | 0.0% |
| spectral_mild | fresh_humbug_retrospective | 85 | 98.8% | 54.1% | 100.0% |
| bandpass_spectral | dryad | 7845 | 34.1% | 8.0% | 59.5% |
| bandpass_spectral | humbug | 383 | 34.2% | 1.6% | 0.0% |
| bandpass_spectral | mosquitosong | 352 | 1.1% | 0.0% | Sem emissão |
| bandpass_spectral | ufrgs | 271 | 0.0% | 0.7% | 0.0% |
| bandpass_spectral | fresh_humbug_retrospective | 85 | 9.4% | 0.0% | Sem emissão |

A redução espectral leve melhora o HumBug de **90,3% para 91,4%**, com cobertura de 55,9% para 58,5%. Nos cinco Aedes retrospectivos, mantém 84/85 acertos forçados e aumenta emissões corretas de **41 para 46**, sem emissão errada. Porém, reduz o Dryad de **62,6% para 60,8%** e UFRGS de **6,6% para 5,5%**, com nenhuma emissão correta na UFRGS. Não existe melhoria consistente entre fontes.

No ruído HumBug, o original e a redução espectral leve não emitiram espécie em 915 janelas. O highpass emitiu espécie em **11/915** janelas; o bandpass 150–3.500 Hz em **31/915**; a combinação em **27/915**. Filtrar não é sinônimo de reduzir falsos alarmes: a normalização interna pode tornar resíduos relativamente mais destacados. Não houve emissão de Aedes nos cinco arquivos negativos não-Aedes da UFRGS em nenhum método; os rótulos dessas pastas não permitem validar as outras espécies.

## Decisão

**Não substituir o pré-processamento padrão.** Os filtros adicionais não resolveram seu enxame e alguns introduziram regressões relevantes. Esta é uma análise de inferência com rede treinada no pipeline original, não prova de que esses filtros jamais ajudariam. Aplicá-los apenas na inferência muda a distribuição da entrada.

Uma próxima rodada deve aplicar o mesmo processamento no treino, validação e calibração, escolher a abordagem sem consultar o teste final e, depois de congelar a nova versão, buscar outro áudio inédito. Os métodos e resultados desta análise não foram usados para ajustar pesos, recalibrar limiares ou liberar firmware. A CNN permanece experimental e a adaptação para Arduino continua aguardando autorização posterior explícita.



# Taxas de acerto por espécie

CNN multifonte atual, com pré-processamento original mantido. Contagens de janelas dos testes já examinados, não de mosquitos independentes. Acerto sem rejeição é recall da espécie real ao forçar a escolha de uma das 20 classes; não é a probabilidade de uma previsão individual estar correta. Identificação correta / todas as janelas inclui as rejeições no denominador. Precisão quando a espécie é emitida usa todas as emissões dessa espécie no corpus, incluindo erros vindos de outras espécies e rótulos desconhecidos. Não misturar taxas de corpora distintos.

## dryad

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 6 | 240/360 | 66.7% | 126/360 (35.0%) | 100.0% |
| Aedes albopictus | 3 | 99/180 | 55.0% | 45/180 (25.0%) | 93.8% |
| Aedes mediovittatus | 1 | 50/60 | 83.3% | 37/60 (61.7%) | 100.0% |
| Aedes sierrensis | 20 | 51/76 | 67.1% | 7/76 (9.2%) | 100.0% |
| Anopheles albimanus | 14 | 408/738 | 55.3% | 98/738 (13.3%) | 86.7% |
| Anopheles arabiensis | 4 | 89/240 | 37.1% | 50/240 (20.8%) | 73.5% |
| Anopheles atroparvus | 2 | 70/120 | 58.3% | 41/120 (34.2%) | 91.1% |
| Anopheles dirus | 19 | 463/776 | 59.7% | 168/776 (21.6%) | 94.9% |
| Anopheles farauti | 15 | 409/725 | 56.4% | 127/725 (17.5%) | 95.5% |
| Anopheles freeborni | 18 | 563/840 | 67.0% | 252/840 (30.0%) | 90.6% |
| Anopheles gambiae | 25 | 317/532 | 59.6% | 64/532 (12.0%) | 95.5% |
| Anopheles merus | 2 | 88/120 | 73.3% | 35/120 (29.2%) | 94.6% |
| Anopheles minimus | 17 | 344/901 | 38.2% | 35/901 (3.9%) | 97.2% |
| Anopheles quadriannulatus | 2 | 39/120 | 32.5% | 22/120 (18.3%) | 91.7% |
| Anopheles quadrimaculatus | 2 | 120/120 | 100.0% | 98/120 (81.7%) | 100.0% |
| Anopheles stephensi | 19 | 654/901 | 72.6% | 308/901 (34.2%) | 95.4% |
| Culex pipiens | 3 | 148/164 | 90.2% | 111/164 (67.7%) | 79.3% |
| Culex quinquefasciatus | 4 | 117/195 | 60.0% | 41/195 (21.0%) | 100.0% |
| Culex tarsalis | 4 | 196/217 | 90.3% | 108/217 (49.8%) | 100.0% |
| Culiseta incidens | 10 | 446/460 | 97.0% | 182/460 (39.6%) | 98.9% |

## humbug

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 7 | 143/154 | 92.9% | 99/154 (64.3%) | 88.4% |
| Anopheles arabiensis | 8 | 120/130 | 92.3% | 66/130 (50.8%) | 97.1% |
| Anopheles dirus | 8 | 83/89 | 93.3% | 42/89 (47.2%) | 72.4% |
| Anopheles minimus | 1 | 0/10 | 0.0% | 0/10 (0.0%) | Sem emissão |

## ufrgs

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 2 | 18/271 | 6.6% | 0/271 (0.0%) | Sem emissão |

## mosquitosong

| Espécie | Origens | Acertos / janelas, sem rejeição | Taxa sem rejeição | Identificações corretas / todas as janelas | Precisão quando a espécie é emitida |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 2 | 142/352 | 40.3% | 2/352 (0.6%) | 100.0% |

MosquitoSong é desafio retrospectivo, com identidade individual entre idades não comprovada. UFRGS permaneceu inteira fora do treino, validação e calibração. No áudio privado informado como enxame de Aedes, nenhuma identificação foi emitida. Esse áudio não fornece referência das outras 19 espécies nem anotação de eventos por janela.



# Condições de gravação e aplicação pretendida

## Fontes efetivamente avaliadas

Dryad/Abuzz: celulares diversos, laboratório e campo; acompanhamento de mosquitos individuais, com aproximação do microfone. O corpus de referência reúne 20 espécies de fêmeas. Há diferenças de ruído, aparelho e protocolo entre gravações.

HumBug, seleção do teste conhecido: 367/383 janelas de mosquitos em copos usando microfone Telinga e gravadores Tascam/Olympus; 16 janelas de Anopheles arabiensis em campo com celular itel A16, sem quantidade de mosquitos informada. Todas as 154 janelas de Aedes eram fêmeas da Tanzânia, anotadas Single, em copos, a 44,1 kHz. Nenhuma dessas 383 janelas tinha anotação Plural. O resultado não demonstra identificação de enxames.

MosquitoSong baixado: gravações de laboratório com animais individuais em cilindros de tela, microfones Behringer ECM8000 e Primo EM172, originalmente 96 kHz/24 bits. O desafio histórico de Aedes utiliza animais de cinco dias. Identidade entre animais de diferentes idades não está comprovada. Não confundir com o conjunto externo W-OUTDOOR do artigo MosquitoSong+.

UFRGS: o protocolo do artigo descreve insetário em Porto Alegre, celular LG K10, animais de 1-12 dias e recipiente para manter o mosquito próximo. Nossa seleção pública contém somente 15 arquivos: dez Aedes (machos e fêmeas) e cinco referências não-Aedes; não é o corpus completo do artigo. Não há garantia de que cada arquivo represente um indivíduo distinto.

## Caixa de vidro

Uma caixa pode padronizar posição do microfone, distância e ruído, mas reflexões, ressonâncias e ventilação também modificam a captação. A melhora é uma hipótese, não um resultado medido. Começar com um mosquito por vez e espécie confirmada, usando o microfone real da placa. Registrar temperatura, dias de aquisição, caixa vazia e interferências. Reservar indivíduos e dias distintos para teste. Vários mosquitos simultâneos exigem avaliação específica de sons sobrepostos. Os 92,9% de Aedes no HumBug não podem ser transferidos automaticamente para a caixa.

# Referências recentes e prioridades

DR-BioL: aprendizado contrastivo e alinhamento entre fontes. Na Tabela 1, a identificação de Ae. albopictus em outra origem melhora de 41,40% na CNN para 74,92% no DR-BioL; o exemplo mostra como uma alta acurácia na origem familiar pode ocultar dificuldade de transferência. Método apresentado no ICASSP 2026; a versão consultada é o manuscrito arXiv. Não reproduzido por nós.

MosquitoSong+ (2024): CNN de áudio bruto, com ruído e variação de volume no treinamento. No experimento de quatro espécies: 89,3% interno, 82,5% HumBug e 80,5% externo. O treinamento inclui porções das três fontes; não é um teste de fonte totalmente ausente do treino. É referência promissora para robustez a ruído, sem demonstração abrangente de enxames reais.

BioDCASE baseline (2026): acurácia balanceada de 88,06% em condições vistas e 17,51% em condições não vistas para a espécie. Prioriza avaliar mudança de origem. Trata-se de nove espécies conhecidas, não detecção de espécies desconhecidas ou enxames.

Scientific Reports (2024), populações mistas: armadilha optoacústica em semicampo; eventos de captura diferem do som coletivo registrado pelo microfone. Útil para limites de aplicações reais, sem validação direta do nosso aparelho.

Synthetic Swarm (2025): prova de conceito de misturas artificiais. Pode orientar aumento de dados, mas não é evidência de generalização para enxames reais. Evitar contaminação por gravações ancestrais já usadas no teste.

Próxima prioridade científica: comparar DR-BioL e MosquitoSong+ com separação por origem e indivíduo, sem selecionar métodos pelo teste final. Após novo treinamento, congelar pesos e limiares e buscar um novo áudio externo, mantendo-o fora da adaptação. A redução para Arduino continua aguardando autorização posterior.

## Links das fontes

- https://elifesciences.org/articles/27854
- https://arxiv.org/html/2306.10091v1
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121
- https://arxiv.org/html/2510.00346v1
- https://www.cmsworkshops.com/ICASSP2026/view_session.php?SessionID=1301&bare=1
- https://arxiv.org/html/2603.20118v1
- https://biodcase.github.io/challenge2026/task5
- https://www.nature.com/articles/s41598-024-54233-3
- https://arxiv.org/abs/2512.12365

Nenhum áudio privado é incluído neste documento; somente resultados agregados. Não houve novo treinamento para gerar o relatório.
