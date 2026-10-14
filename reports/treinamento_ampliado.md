# Treinamento ampliado e avaliação externa

**O modelo melhorou no HumBugDB, mas ainda não é confiável para liberação.**
Falhou no áudio novo de Aedes aegypti, falhou no enxame relatado pelo usuário
e piorou no corpus antigo. Por instrução do usuário, não foram gerados headers,
compilados programas Arduino, alterados instaladores ou publicados binários.

## Dados e controle de vazamento

Foram mantidas as 24.405 janelas anteriores, das 20 espécies e dos ruídos,
e acrescentadas **8.742 janelas de 307 trechos**, pertencentes a **214 gravações
de origem** do [HumBugDB](https://zenodo.org/records/4904800).
Total: 33.147 janelas disponíveis; nem todas são utilizadas para ajustar pesos.

A seleção adicional usa semente 20260930, anos ≥2018, duração ≥0,992 s, limites
por classe/gravação e um trecho de maior duração por rótulo em cada origem.
Isso evita que um trecho longo de fundo elimine o trecho de mosquito da mesma
gravação. Foram obtidos os dados por HTTP Range e conferidos CRC32 e SHA-256 de
cada entrada. O CSV oficial teve o MD5 integral conferido. Os quatro ZIPs
inteiros não foram baixados e seus MD5s integrais não foram conferidos.

Há novas gravações de Aedes aegypti, Anopheles arabiensis, Anopheles dirus,
Anopheles gambiae, Anopheles minimus e Anopheles stephensi. As outras 14 espécies
continuam representadas somente pelos dados antigos. Espécies externas fora das
20 classes e anotações em nível de complexo são positivas para o detector de
presença, sem serem convertidas em uma espécie exata no treino do classificador.

Os arquivos HumBug 220292 e 222024, que já haviam sido examinados, passaram a
adaptação: todas as suas gravações de origem ficaram no treino. O áudio privado
do Drive **não entrou** no treinamento. A espécie desse áudio foi informada pelo
usuário como Aedes aegypti, em enxame, sem confirmação entomológica independente.

| Corpus | Partição | Janelas | Gravações de origem |
|---|---|---:|---:|
| Dryad anterior | Treino | 10.282 | 252 |
| Dryad anterior | Validação | 2.755 | 68 |
| Dryad anterior | Calibração | 3.467 | 89 |
| Dryad anterior | Teste | 7.901 | 204 |
| HumBug adicional | Treino | 4.571 | 113 |
| HumBug adicional | Validação | 1.405 | 30 |
| HumBug adicional | Calibração | 1.256 | 30 |
| HumBug adicional | Teste | 1.510 | 41 |

As contagens incluem janelas inválidas: 32 das janelas externas não passaram
pelos guardas de amplitude/clipping e foram excluídas do ajuste dos pesos.
A soma de gravações por rótulo não deve ser usada como total, pois a mesma origem
pode ter trechos de mosquito e de fundo. Os grupos não se cruzam entre partições;
entradas externas com bytes idênticos compartilham grupo. A data posterior a
2017 exclui coleções antigas que poderiam compartilhar material com o Dryad.

O teste Dryad é exatamente o da antiga dobra 2. Ele não entrou neste treino,
mas seus resultados já haviam sido analisados anteriormente: não é um teste
que nunca foi consultado no projeto. O teste HumBug foi fixado antes do ajuste.

## Modelo e calibração

Foram comparadas redes 68→64→20 e 68→128→64→20 na validação separada,
com equilíbrio por classe, corpus e gravação. A primeira foi escolhida:
47,55% de acerto ponderado de validação contra 46,04%, com época 12.
O detector de presença é uma rede 68→16→1, com época 3 escolhida pela validação.
O frontend continua sendo FFT de 512 amostras e janelas de 31 frames a 16 kHz.

Calibração em uma terceira partição: presença 0,6900000572; espécie
0,3891417086, além da confirmação causal da mesma espécie em duas das últimas
três janelas, incluindo a janela atual. As metas de calibração não garantem
desempenho no teste. A rejeição continua sendo regra, não uma 21ª classe treinada.

Os dados externos foram reamostrados para PCM de 16 bits a 16 kHz, sem ganho
ou filtro adicional de ruído. A reamostragem inclui filtragem antialias.
O corpus anterior permanece com seu cache normalizado de replay; isso é uma
diferença de processamento entre as fontes e uma limitação da comparação.

Checkpoint congelado: `results/expanded/models/species_fold0.npz`.
Apesar do nome de arquivo herdado, esse checkpoint é da ampliação, e não uma
nova avaliação de três dobras. O campo `fold=-1` diferencia essa versão.
SHA-256: `475a978d597c62690dc9bba3f6fb4196804dc23084ec62721a529f87e3246a40`.
Pesos compactados/restaurados concordaram com o cálculo Python dos modelos.
Não houve nova compilação C++ nem Arduino.

## Comparação nos mesmos testes

Escolha forçada ignora rejeição. Precisão entre as identificações aceitas deve
ser lida junto com cobertura. As coberturas abaixo incluem todas as janelas
positivas de espécie conhecida, inclusive início e lacunas, e portanto não são
o denominador antigo de 5.983 endpoints contíguos.

| Corpus/modelo | Escolha forçada: acerto | Acerto entre identificações aceitas | Cobertura | Fração total corretamente identificada |
|---|---:|---:|---:|---:|
| Dryad/antigo | 42,2% | 87,9% (231 emissões) | 2,9% | 2,6% |
| Dryad/ampliado | 33,5% | 59,4% (384 emissões) | 4,9% | 2,9% |
| HumBug/antigo | 3,1% | 0% (7 emissões) | 1,8% | 0% |
| HumBug/ampliado | 74,2% | 90,3% (144 emissões) | 37,6% | 33,9% |

O teste HumBug de espécie conhecida contém somente **383 janelas de 23 origens
e quatro espécies**, não as 20. A acurácia balanceada da escolha forçada passou
de 3,0% para 57,6%; a acurácia por gravação, agregando escores, de 0% para 82,6%.
O agregado de 74,2% por janela não deve esconder diferenças por espécie.

| Espécie no teste HumBug | Origens | Janelas | Recall forçado novo | Cobertura nova | Acerto entre emissões |
|---|---:|---:|---:|---:|---:|
| Aedes aegypti | 7 | 154 | 87,0% | 24,0% | 94,6% (37 emissões) |
| Anopheles arabiensis | 8 | 130 | 54,6% | 32,3% | 100% (42 emissões) |
| Anopheles dirus | 8 | 89 | 88,8% | 64,0% | 93,0% (57 emissões) |
| Anopheles minimus | 1 | 10 | 0% | 80,0% | 0% (8 emissões) |

Em 915 janelas externas de fundo, o modelo ampliado não emitiu espécie, contra
10 emissões antigas. Presença candidata no fundo caiu de 19,8% para 2,6%.
Isso é uma taxa neste subconjunto, sem afirmar zero alarmes por hora no campo.
Nos 212 trechos com espécie externa não resolvida/fora das 20 classes, houve
emissão de espécie em 17,5%, contra 2,4% antes. Anotações de complexo podem
conter uma espécie das 20, portanto esse número é taxa de emissão em rótulos
não resolvidos, sem automaticamente equivaler a taxa de erro.

## Novo áudio pesquisado somente após o congelamento

Após terminar o treino e congelar o SHA-256, foi pesquisada uma nova fonte:
[MosquitoSong+](https://github.com/akaraspt/mosquitosongp), coleta da Mahidol
University, diferente de Dryad e HumBug. A procedência e metodologia estão no
[artigo de 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11524479/).

Foi escolhido o primeiro corte por ordem natural que tivesse pelo menos
2,976 s, sem olhar previsões: `Ae.Aegypti_1F_5D_25C_SmallCylinder_cut1_16bit.wav`.
O próprio primeiro corte já cumpriu o critério. Fonte pública:
[arquivo dos autores no Drive](https://drive.google.com/file/d/128vI81VPQTnUIlhzQpLepA-XFhOD--sk/view).
Referência: Aedes aegypti fêmea, segundo a nomenclatura dos autores.
O arquivo disponibilizado tem 5,622 s, mono, PCM de 16 bits, **8 kHz**; foi
reamostrado para 16 kHz sem recuperar frequências ausentes acima de 4 kHz.

SHA-256: `6308063ef206a3b4ffd4fd11b175acc3c9cc9ea4787b43a7f554d1ade48a30ee`.
Não houve coincidência exata com os 307 trechos HumBug selecionados nem com os
888 WAV originais locais. A evidência principal de independência é a procedência
da coleta, não somente a diferença de hash. Não foi incorporado a treinamento,
validação ou calibração. A busca/download aconteceu após congelar o modelo, com
horários e SHA-256 registrados. Os pesos ficaram inalterados depois deste teste.

| Modelo | Janelas | Resultado |
|---|---:|---|
| Antigo | 5 | 5 incertas; nenhuma espécie emitida |
| Ampliado | 5 | 4 incertas e 1 sem evidência; nenhuma espécie emitida |

Nem a escolha forçada foi Aedes aegypti em qualquer dessas cinco janelas.
**A nova gravação não foi reconhecida.** Não houve repetição, preenchimento
ou concatenação de cortes para forçar a confirmação temporal.

## Áudio privado de enxame

Reteste do PCM original de 106,24 s, mantido fora do treino, com referência
Aedes aegypti fornecida pelo usuário. A nova versão produziu 104 janelas sem
evidência, 1 incerta e 2 inválidas: **nenhuma identificação de espécie**.
O resultado anterior foi 95 sem evidência, 10 incertas e 2 inválidas.
Não houve escuta perceptiva direta nesta sessão; a análise foi computacional.
Os dados privados e previsões por janela permanecem somente em `tmp/`, ignorado
pelo Git. Não existem anotações temporais independentes de cada som do enxame.

## Decisão

**Não liberar para Arduino.** O ganho na coleta HumBug não se transferiu para a
gravação recém-obtida do MosquitoSong+ ou para o enxame. Houve regressão no corpus
anterior, e o teste externo cobriu apenas quatro das 20 classes. Um acerto em um
único áudio futuro também não asseguraria confiabilidade global.

Para uma próxima rodada, é necessário ampliar a diversidade de aparelhos,
populações, sexos e enxames com espécies verificadas, investigar a representação
acústica e o desequilíbrio entre domínios. Os arquivos já usados para orientar
essa rodada deixarão de ser inéditos; após o próximo treino será necessário
outro teste novo, conforme o [protocolo](../MODEL_VALIDATION.md).

## Reprodução e evidências

```bash
.venv/bin/python fetch_humbug.py
.venv/bin/python train_expanded.py
.venv/bin/python test_expanded_audio.py CAMINHO.wav --species 'Aedes aegypti' --source ORIGEM --output PASTA
```

- [Contagens](../results/expanded/data_counts.csv), [partições](../results/expanded/data_protocol.json).
- [Comparação](../results/expanded/comparison.csv), [por espécie](../results/expanded/per_class.csv).
- [Seleção das redes](../results/expanded/species_selection.json), [auditoria](../results/expanded/training_audit.json).
- [Teste novo](../results/expanded/fresh_mosquitosong/analysis.json), [seleção e congelamento](../results/expanded/fresh_mosquitosong/selection.json).
- [Decisão de confiabilidade](../results/expanded/reliability_decision.json).
