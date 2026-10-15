# Critérios do trabalho e revisão da apresentação

Autor: Luiz Eduardo do Nascimento Ciarallo. Tema: detecção acústica ternária de Aedes aegypti.

Fonte: Trabalho_Descrição-1.pdf, páginas 2–4, Computação Científica para a Indústria, IFSC Florianópolis.

| Exigência | Evidência | Slides |
|---|---|---|
| Estudo de técnicas e programa com rede neural profunda | LSTM/Bi-LSTM de duas camadas treinadas; scripts, pesos, previsões, teste de inferência e verificador | 6–13 |
| Contextualização da aplicação | Voo, detecção de Aedes e três saídas | 1–2 |
| Abordagens comuns e trabalhos correlatos | Mukundarajan, Torres, Supratak, Sinka e cascata clássica | 3, 5, 10, 17 |
| Apresentação da proposta implementada | STFT, mel, partições, arquitetura, otimização e regularização | 4–13 |
| Resultados obtidos | Curvas reais, comparação pareada de população, sensibilidade/FPR e matriz de confusão | 13–16 |
| Discussão final | Interpretação, limites, testes futuros | 18 |
| Equipe com no máximo dois integrantes | Um autor identificado por nome completo | 1 |
| Apresentação de 15–20 minutos | Roteiro com tempos por slide | 18 slides; 18 minutos aproximadamente |
| Entrega da apresentação ao professor | Slides nativos e PDF para entrega pelo autor | Artefatos de saída; envio não solicitado |

O nome, a foto real do IFSC e a identidade do modelo oficial são preservados. Os 90,03% são atribuídos somente à cascata clássica. As redes têm resultados próprios para as três classes; não são substituídos pelos resultados das 20 espécies nem pela tarefa de infecção do artigo.

Os valores de Adam, dropout, amostragem e parada são os executados, sem alegar otimização causal sem ablação. As limitações estão concentradas no encerramento.
