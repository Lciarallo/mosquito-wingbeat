# Teste externo: HumBugDB, Anopheles arabiensis

**O firmware atual errou este exemplo externo.** A espécie de referência é
Anopheles arabiensis; as três identificações provisórias emitidas foram
Aedes sierrensis. Não houve identificação correta de espécie.

## Origem e escolha antes da inferência

- Dataset: [HumBugDB, Zenodo 4904800](https://zenodo.org/records/4904800).
- Áudio: `222024.wav`, dentro de `humbugdb_neurips_2021_4.zip`.
- Metadados: [CSV dos autores](https://github.com/HumBug-Mosquito/HumBugDB/blob/master/data/metadata/neurips_2021_zenodo_0_0_1.csv), ID 222024.
- Referência: `an arabiensis`, fêmea, `Single`, Ifakara, Tanzânia,
  13/07/2020, gravação em recipiente (`cup`), microfone Telinga e gravador Tascam.
- Duração: 23,040 s; original mono, 44.100 Hz, PCM de 24 bits.
- Critério fixado antes de inferir: menor ID presente no quarto ZIP com
  espécie Anopheles arabiensis, país Tanzânia, indivíduo único e duração ≥5 s.
  Não foram testados vários arquivos para escolher um resultado favorável.

O modelo foi treinado no corpus Dryad de 2017, DOI 10.5061/dryad.98d7s.
Este registro é de outra coleta, em 2020, e foi acessado após o treinamento.
Além dessa evidência de procedência, não houve coincidência exata de SHA-256
com os 888 WAV locais de `data/raw` (comparação de arquivos, sem busca por
similaridade acústica ou comparação com todos os outros formatos).
Anopheles arabiensis pertence às 20 classes do firmware.

O WAV foi extraído por requisições HTTP Range, sem baixar os 592,8 MB do ZIP
inteiro. Foram conferidos tamanho descomprimido e CRC32 da entrada ZIP.
O MD5 do ZIP inteiro não foi verificado, pois o arquivo completo não foi baixado.
SHA-256 do WAV: `19e4fcf0f77d022974147b801dff4b24a8167a8fcc1687b0591cc4dc42595d48`.

## Procedimento

Foi usado o modelo padrão da dobra 2, com limiar de presença 0,6712744832,
limiar de espécie 0,4800000191 e confirmação da mesma espécie em duas das
três janelas, incluindo a janela atual elegível. Pesos e limiares permaneceram
os existentes antes deste teste; não houve ajuste nem treinamento no áudio.

A entrada foi convertida para mono PCM de 16 bits a 16 kHz, com
`scipy.signal.resample_poly(up=160, down=441)`. Essa reamostragem inclui
filtragem antialias; não foi aplicado filtro adicional de ruído, ajuste de
volume ou normalização. SHA-256 da entrada convertida:
`71577cf324688dcb2ee58f60cbda61def98f3719ce28985e1813291f63c54e58`.

Foram processadas 23 janelas completas de 0,992 s (22,816 s); os 0,224 s finais
não completaram uma janela e ficaram fora da inferência. O código C++ com os
headers reais do firmware foi compilado e executado em computador; suas classes
mais pontuadas e decisões de confirmação coincidiram com Python nas 23 janelas.
Não houve teste em placa física ou captura pelo microfone da Nano 33 BLE Sense.

## Resultado

| Saída | Janelas |
|---|---:|
| Sem evidência suficiente | 18 |
| Incerto | 2 |
| Identificação provisória de Aedes sierrensis — incorreta | 3 |
| Identificação de Anopheles arabiensis — correta | 0 |
| Áudio inválido | 0 |

As identificações incorretas ocorreram em 4,960–5,952 s,
20,832–21,824 s e 21,824–22,816 s. Mesmo forçando a maior saída entre as
20 classes, nenhuma das 23 janelas teve Anopheles arabiensis como primeira
opção. Esse resultado forçado não é o comportamento de rejeição do firmware.

Os escores altos de algumas previsões erradas não representam acurácia nem
probabilidade confiável em uma nova coleta. Este resultado mostra que a
confirmação temporal atual também pode aceitar uma classificação errada
quando o erro persiste entre janelas.

## Interpretação e limites

O teste respondeu à pergunta para este arquivo: **não acertou a espécie**.
Não se trata de uma espécie ausente do treinamento nem de um enxame:
o metadado é de um indivíduo. Logo, os problemas externos não podem ser
atribuídos somente à sobreposição de sons de enxames.

Mudanças de aparelho, local, população e condições de gravação são possíveis
contribuições para a falha; este teste não isola a causa. A reamostragem
também faz parte do procedimento avaliado.

Um único registro não estima a acurácia geral externa, e suas janelas não
são amostras independentes. A referência de espécie foi obtida nos metadados
dos autores, sem nova identificação entomológica. Sem anotar manualmente a
atividade acústica por janela, os 18 resultados de rejeição não devem ser
automaticamente contabilizados como 18 falsos negativos de presença.

A próxima avaliação necessária é um conjunto externo fixado antes dos
resultados, com vários indivíduos de cada espécie, separação por origem e
aparelho, e métricas que apresentem conjuntamente erros e rejeições. Se esses
dados forem usados para melhorar o modelo, outro conjunto independente terá
de ficar reservado para avaliação.

## Evidências locais

- [Auditoria e resultado](../results/external_validation/humbug_222024/analysis.json).
- [Previsões por janela](../results/external_validation/humbug_222024/predictions.csv).
- [Seleção, origem e metadados](../results/external_validation/humbug_222024/selection.json).
- WAV original, entrada convertida, script e gráfico: `tmp/external-humbug/`.

Referência metodológica: [artigo HumBugDB](https://arxiv.org/html/2110.07607).
