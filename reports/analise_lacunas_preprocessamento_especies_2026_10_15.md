# Análise de lacunas: pré-processamento das 20 espécies

## Objetivo

Verificar se uma representação acústica mais adequada melhora a identificação
das 20 espécies em relação à rodada LSTM/Bi-LSTM de Torres. Preservar os dados,
pesos e resultados anteriores.

## Evidências e lacunas

- O frontend anterior usa 16 kHz, janelas de 0,992 s, FFT 1.024, salto 256,
  32 bandas mel entre 80 e 4.000 Hz, remoção da média e normalização RMS.
  A resolução da FFT é 15,625 Hz; 32 bandas ainda comprimem a posição dos tons.
- A Bi-LSTM anterior teve 100/190 acertos e a referência linear 110/190. As
  identidades PCM já foram auditadas: 209 grupos de treino, 59 de validação e
  190 de teste, sem o mesmo bloco PCM atravessando essas partições.
- Os filtros de `audio_preprocessing.py` foram avaliados em uma rede congelada,
  treinada em outro processamento. A avaliação antiga não responde à pergunta
  sobre retreinar com o mesmo frontend em todas as partições.
- Contraste espectral já existe no projeto, e PCEN já foi usado em outros
  modelos. Falta comparar essas técnicas no protocolo das redes recorrentes.

## Experimento definido antes da nova avaliação

1. Reutilizar exatamente as partições da rodada Torres e registrar seus hashes.
2. Comparar três novas representações, sempre em treino, validação e teste:
   - `hires_logmel`: FFT 2.048, salto 256, 64 bandas mel, 80–4.000 Hz;
   - `hires_contrast`: mesma entrada com compensação parcial da envoltória
     espectral local, preservando a posição absoluta dos picos;
   - `hires_pcen`: mesma entrada com normalização de energia por banda e
     compressão dinâmica, sem deslocar o pitch.
3. Manter 16 kHz e janelas de 0,992 s. A nova FFT tem espaçamento de 7,8125 Hz
   e janela de 128 ms; o salto continua em 16 ms. O resultado tem 55 passos.
4. Treinar referência linear com média/desvio temporal e Bi-LSTM com duas
   camadas de 256 unidades, dropout 0,20, Adam 0,0003, lote 128, clipping 3,
   máximo 80 épocas e paciência 15. A entrada passa de 32 para 64 atributos;
   essa diferença e a quantidade de parâmetros serão registradas.
5. Ajustar escalas apenas no treino. Usar a mesma semente, ponderação por
   classe/gravação e critério de perda de validação da rodada anterior.
6. Congelar frontend/modelo escolhido, pesos e transformações antes da nova
   inferência de teste. Manter todos os candidatos nas tabelas.
7. Reportar acerto por gravação, F1 macro, erros pareados e Monte Carlo dropout
   com cobertura para a nova rede escolhida na validação. Não fazer busca
   adicional após observar os novos resultados de teste.

O teste histórico já foi observado em rodadas anteriores: esta comparação é
retrospectiva. As 20 classes não incluem ruído; a métrica aqui é identificação
de espécie. MosquitoSong continua fora do treino. A tarefa de infecção DENV-2
permanece separada.

## Fontes técnicas

Os testes históricos protegem também o hash do `README.md`. A documentação nova
foi movida para `README_TORRES_2025.md`, mantendo os bytes originais do README e
os manifests anteriores. A suíte completa com os dez testes novos passou com
308 testes aprovados.

- Mukundarajan et al.: frequência fundamental e harmônicos do voo.
  https://elifesciences.org/articles/27854
- Wang et al.: PCEN e compressão dinâmica por banda.
  https://research.google/pubs/trainable-frontend-for-robust-and-far-field-keyword-spotting/
