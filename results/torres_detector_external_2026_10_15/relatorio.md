# Teste do detector ternário nos exemplos WAV de Torres — 15/10/2026

Os dois WAVs originais disponíveis foram submetidos aos modelos finais congelados de cascata clássica, LSTM e Bi-LSTM. Nenhum treinamento, ajuste de limiar ou mudança da apresentação foi realizado.

| Áudio | Duração | Cascata | LSTM | Bi-LSTM |
|---|---:|---|---|---|
| infectado (`infected0001.wav`) | 4,085 s | Outro mosquito | Outro mosquito | Outro mosquito |
| saudável (`healthy0001.wav`) | 1,981 s | Outro mosquito | Outro mosquito | Outro mosquito |

A classe correta dos dois é Aedes aegypti. Cada modelo detectou presença de mosquito nos dois áudios, mas nenhum reconheceu a espécie Aedes: 0/2 acertos na classificação Aedes, incluindo 0/1 no exemplo infectado. Isso revela uma falha de transferência nesses exemplos; não estima a sensibilidade na população de infectados. Não foram incluídos negativos para medir falso positivo neste teste.

Os modelos neurais mantiveram a decisão de outro mosquito em todas as 100 passagens de MC dropout por áudio. Concordância entre passagens não comprova acerto.

## Dados e método

A distribuição pública baixada oferece 300 tensores mel de 188 × 32, mas apenas dois sinais WAV exemplificativos (um infectado e um saudável). Os tensores não foram interpolados ou tratados como áudio: a rede ternária recebe 55 × 64 e a cascata exige atributos extraídos do sinal. Este teste não equivale a avaliar os 300 exemplos com o detector ternário. O estudo separado de classificação de infecção permanece independente.

Usados trechos reais de 0,992 s: quatro do infectado, um do saudável. Caudas incompletas foram descartadas pelo procedimento original, sem repetição ou preenchimento. Os dois arquivos passaram pela checagem exata de bytes/PCM alinhado contra o corpus de desenvolvimento, sem sobreposição detectada. Isso não demonstra independência biológica.

O verificador do modelo clássico inicialmente bloqueou a execução com o código atual, alterado posteriormente pela exclusão do áudio privado. A execução final usou uma cópia isolada dos dez arquivos originais com hashes iguais aos do freeze clássico e os quatro arquivos de fonte arquivados na rodada neural, também verificados. O checkout e os freezes foram preservados. O script de inferência é o existente, com modelos finais e limiares originais.

## Evidências

- `predictions.csv`: seis decisões por modelo/arquivo.
- `predictions.json`: detalhes, probabilidades e votação MC.
- `summary.json`: hashes dos modelos/PDF antes e depois, conclusão e auditoria de sobreposição.
- `source_snapshot_hashes.json`: hashes das dez fontes congeladas clássicas.
- `source_snapshot/`: cópia isolada de fontes para reprodução.
- `avaliar.py`: rotina desta avaliação; executar com `.venv/bin/python results/torres_detector_external_2026_10_15/avaliar.py` a partir da raiz.

Não foi feita qualquer chamada de edição da apresentação nem atualização de seus arquivos no Drive.
