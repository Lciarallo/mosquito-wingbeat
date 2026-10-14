# Teste externo de Aedes aegypti — HumBugDB 220292

**O modelo atual não identificou Aedes aegypti neste arquivo.** Nenhuma espécie
foi emitida pelas regras de confirmação do firmware. O resultado foi de rejeição
ou incerteza, diferentemente do teste anterior de Anopheles arabiensis, que emitiu
uma espécie incorreta.

## Referência e seleção

- Origem: [HumBugDB, Zenodo 4904800](https://zenodo.org/records/4904800).
- Entrada: `220292.wav`, arquivo `humbugdb_neurips_2021_3.zip`.
- [Metadados dos autores](https://github.com/HumBug-Mosquito/HumBugDB/blob/master/data/metadata/neurips_2021_zenodo_0_0_1.csv):
  `ae aegypti`, fêmea, `Single`, Ifakara, Tanzânia, 13/03/2020;
  `location_type=cup`, microfone Telinga, gravador Tascam.
- Nome da gravação de origem: `IFA_40_27_1587.wav`.
- Duração: 48,580 s, mono, 44.100 Hz, PCM de 24 bits.
- Seleção antes de executar a inferência: menor ID com espécie Aedes aegypti,
  país Tanzânia, indivíduo único e duração ≥5 s. Não houve seleção de áudio
  por resultado favorável, treinamento ou ajuste de limiares.

O registro é de 2020, de uma coleta distinta do corpus Dryad de 2017 usado no
treinamento. Aedes aegypti pertence às 20 classes do modelo. Não houve
coincidência exata de SHA-256 com os 888 WAV locais de `data/raw`; essa verificação
não compara similaridade acústica nem todos os demais formatos.

O áudio foi obtido por HTTP Range. A entrada descomprimida teve tamanho e CRC32
conferidos contra o diretório ZIP. O ZIP inteiro não foi baixado e seu MD5
integral não foi verificado. SHA-256 do WAV original:
`f373980b2b43288a48aa1a5de80606f5011a426a89be4f3931ab8719614e0269`.

## Procedimento e resultado

Modelo padrão `species_fold2.npz`, limiar de presença 0,6712744832 e de espécie
0,4800000191. Confirmação exige a mesma espécie em duas das últimas três
janelas, com a atual elegível e reset após áudio inválido.

A gravação foi reamostrada com `scipy.signal.resample_poly(up=160, down=441)`
para PCM mono de 16 bits a 16 kHz. A reamostragem inclui filtragem antialias;
não houve filtro adicional de ruído nem ajuste de volume ou normalização.
SHA-256 da entrada convertida:
`4efbf73cda381eebcedc9d746691bf62471cfcbb9afff744b56b92dac27c7e05`.

Foram avaliadas 48 janelas completas de 0,992 s (47,616 s); a sobra final de
0,964 s ficou fora da inferência. As classes mais pontuadas e decisões de
confirmação do C++ compilado com os headers reais do firmware coincidiram
com Python nas 48 janelas. Não foi testada a placa física nem a captura PDM.

| Resultado do firmware | Janelas |
|---|---:|
| Sem evidência suficiente | 45 |
| Incerto | 3 |
| Identificação de Aedes aegypti | **0** |
| Identificação de outra espécie | 0 |
| Áudio inválido | 0 |

Uma janela passou isoladamente pelos limiares de presença e espécie, mas
indicava Culex pipiens e não passou pela confirmação temporal. Forçando a
classe mais pontuada, sem respeitar rejeição, Aedes aegypti apareceu em apenas
1/48 janelas; isso não constitui identificação emitida pelo firmware.

## Conclusão e limites

O sistema não reconheceu a espécie neste exemplo externo de um indivíduo.
Isso acrescenta evidência de dificuldade de generalização para outra coleta,
mesmo quando a espécie está entre as classes treinadas. Não permite atribuir
a causa a aparelho, ruído, população ou outro fator isolado.

Um arquivo e suas janelas correlacionadas não estimam acurácia geral. A espécie
de referência é a anotada pelos autores, sem nova identificação entomológica.
Sem anotação acústica por janela, as 45 rejeições não devem ser automaticamente
tratadas como 45 falsos negativos de presença. Não houve escuta perceptiva
direta nesta sessão; a análise foi computacional.

## Evidências

- [Auditoria](../results/external_validation/humbug_220292/analysis.json).
- [Previsões por janela](../results/external_validation/humbug_220292/predictions.csv).
- [Seleção e metadados](../results/external_validation/humbug_220292/selection.json).
- WAV original, entrada convertida, script e gráfico: `tmp/external-aedes/`.
