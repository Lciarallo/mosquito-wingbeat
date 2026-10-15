# LSTM ternária e revisão da apresentação: análise de lacunas

## Objetivo e critérios do trabalho

O modelo deve distinguir `no_mosquito`, `other_mosquito` e `aedes_aegypti`.
O PDF dos professores Gregory Chagas Da Costa Gomes e Maurício Edgar Stivanello
exige contextualização, trabalhos correlatos, proposta implementada com rede
neural profunda, resultados e discussão final, em apresentação de 15–20 minutos.
O nome completo do aluno é Luiz Eduardo do Nascimento Ciarallo.

## Lacunas constatadas

- A apresentação atual concentra CNN/MLP para 20 espécies e trata o detector
  ternário apenas como extensão. Isso não representa o objetivo confirmado.
- Os 90,03% são a acurácia OOF da cascata clássica, em 1.494 grupos, com 78,33%
  de sensibilidade Aedes. Não são resultados de uma LSTM nem significam 90% de
  sensibilidade de Aedes.
- As LSTM/Bi-LSTM implementadas até agora foram avaliadas em infecção DENV-2 e
  identificação de 20 espécies. Falta a comparação de três classes solicitada.
- Há 45 grupos com rótulos diferentes entre janelas. A verdade de grupo precisa
  seguir a regra congelada do detector: maior rótulo entre as janelas do grupo.

## Protocolo fixado antes do treino

1. Copiar as 42.029 janelas, 1.494 grupos e cinco dobras externas do detector
   `ternary_verified_2026_10_14_corrections`. Manter isolamento por bloco PCM e
   exclusão de MosquitoSong. Não alterar o detector de referência.
2. Usar as formas de onda já auditadas a 8 kHz, todas com 0,992 s nesta população.
   Frontend: FFT 1.024, Hann 128 ms, salto 128 amostras/16 ms, 64 bandas mel de
   80 a 4.000 Hz, com 55 passos temporais. Não reamostrar para alegar informação
   de frequência ausente.
3. Comparar LSTM e Bi-LSTM, duas camadas de 128 unidades, dropout 0,20,
   Adam 0,0006, lote 256, clipping 3, máximo 30 épocas, paciência 5.
   Trata-se de adaptação da técnica recorrente do artigo, que usa 256 unidades.
4. Em cada treino externo, usar uma dobra interna de quatro dobras por PCM para
   escolher época pela perda de grupo equilibrada por classe. Ajustar o scaler
   só nas linhas de treino efetivo. Pesos compostos usam gamma 0,50 e massa de
   cada classe igual a um terço da massa total.
5. Calibrar a cascata neural apenas nas previsões dessa validação interna,
   com os tetos empíricos de 0,75% e 1,75%. Reajustar a rede do zero em todo o
   treino externo pelo número de épocas escolhido, congelar e avaliar a dobra
   externa. Não usar seus resultados para alterar arquitetura/limiar.
6. Manter também a decisão por argmax para tornar o efeito da calibração visível.
   Comparar acurácia geral, sensibilidade Aedes, FPR nas duas classes negativas
   e matrizes das mesmas 1.494 unidades. OOF externo cobre todas as unidades.
7. Ajustar modelos finais pelo mesmo procedimento interno e disponibilizar
   inferência de três classes, incluindo Monte Carlo dropout opcional de 100
   passagens. A comparação principal OOF é determinística.

## Atualização da apresentação

Editar a apresentação nativa existente, preservando o modelo oficial do IFSC,
a foto real do câmpus, as imagens técnicas pertinentes e as notas de fala.
Reformular a narrativa em torno das três saídas, do programa implementado,
das redes recorrentes, do baseline clássico e da comparação efetivamente medida.
Mostrar 90,03% em destaque com denominação e denominador corretos. Manter
limitações e interpretação de alcance no encerramento. Exportar e inspecionar
todas as páginas antes de declarar a entrega concluída.
