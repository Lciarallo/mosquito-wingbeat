# Detector binário: Aedes aegypti ou não-Aedes aegypti

Treinamento concluído em 30/09/2026, horário de São Paulo. **Modelo dedicado implementado e testado no computador, mas ainda experimental.** Não identifica as outras espécies. Não-Aedes inclui outras espécies rotuladas e ruído; INCERTO e AUDIO_INVALIDO são estados de rejeição, não rótulos negativos.

## Algoritmo

CNN residual de 3.315.650 parâmetros, cabeça binária, inicializada com o extrator do modelo multifonte. Mono/16 kHz, remoção DC, normalização RMS e dois canais de log-mel/contraste. Treino alterna janelas reais de 0,3 e 0,992 s, sem preenchimento ou repetição. Adição de ruído real de treino, SNR 5–30 dB, pequenas máscaras espectrais e variação de coloração; não mistura espécies diferentes sob um rótulo puro.

17.391 janelas válidas de treinamento: 3.209 Aedes e 14.182 negativos. Amostragem com 50% de massa para Aedes e 50% para negativos, dividindo estes entre rótulos, origens e grupos. Mosquito sem identificação resolvida foi excluído da supervisão binária; pode ser Aedes. As partições anteriores foram preservadas. UFRGS, áudio privado e desafio histórico de MosquitoSong não entraram no treino. O extrator herdado usou supervisão de presença de alguns mosquitos desconhecidos na rodada anterior, mas não rótulos binários inventados.

Seleção pelo erro binário ponderado na validação: seis épocas, melhor época 2. A calibração logística e os limiares usam exclusivamente a partição de calibração. Alvo de falsos positivos de 5% **ponderado por rótulo/origem/grupo** na calibração, não garantia de 5% em novos dados. Taxa não ponderada da calibração: 6,35%.

## Uso

```bash
.venv/bin/python aedes_binary.py caminho/gravacao.wav --output tmp/teste_aedes
```

Salva predictions.csv por janela e summary.json. Precisa do checkpoint local results/aedes_binary/models/aedes_binary.pt. O arquivo de pesos é ignorado pelo Git, não está publicado remotamente nesta rodada. A chamada é offline; não exige Arduino. Janelas de duração mínima 0,992 s no comando de áudio; não são estendidas artificialmente. Não há decisão temporal de enxame ou confirmação por indivíduo neste detector: cada janela produz um escore experimental.

## Testes históricos

Sensibilidade forçada: Aedes em primeiro lugar com corte 0,5, sem rejeição. Detecção aceita: emissão AEDES_AEGYPTI com o limiar congelado e qualidade válida. Falso positivo: emissão de Aedes em janela rotulada não-Aedes. Janelas são correlacionadas, não indivíduos independentes.

| Fonte | Janelas Aedes | Janelas negativas | Sensibilidade forçada | Detecção aceita de Aedes | Falsos positivos aceitos nos negativos |
|---|---:|---:|---:|---:|---:|
| dryad | 360 | 7541 | 90.0% | 84.4% | 1.2% |
| humbug | 154 | 1144 | 94.8% | 93.5% | 24.4% |
| mosquitosong | 352 | 0 | 54.3% | 52.0% | Não avaliável |
| ufrgs | 271 | 146 | 34.7% | 22.5% | 2.7% |

Não há negativos no desafio MosquitoSong; por isso não se pode inferir especificidade ou confiabilidade de uma indicação positiva a partir dele. O teste HumBug tem ainda 212 janelas sem espécie resolvida: foram avaliadas separadamente e não contadas como verdadeiros positivos/negativos. Em 74 dessas janelas o detector emitiu Aedes; sua correção é desconhecida.

No Dryad, a acurácia global de 97,9% é influenciada pela predominância de negativos; a acurácia balanceada é 94,1%. No HumBug, 144 emissões corretas e 279 falsos positivos conhecidos: somente 34,0% das emissões de Aedes em janelas rotuladas foram corretas. O binário é mais sensível que a classificação de 20 espécies em algumas fontes, mas não é universalmente melhor.

## Áudios novos após congelamento

Busca online e seleção posteriores ao congelamento. Seis gravações HumBug novas para esta rodada: três Aedes (220428, 220451, 220482) e três outras espécies (206578, 2958, 2330). A proposta inicial de cinco por classe não pôde ser preenchida: restavam só três grupos positivos elegíveis, e a seleção foi reduzida antes de qualquer inferência. Excluídos grupos de dados antes usados e dos WAV numéricos baixados. Conferidos CRC e SHA-256; sem bytes idênticos aos áudios locais de datasets externos. Não foi concluída auditoria de sobreposição de formas de onda deslocadas ou identidade dos indivíduos entre campanhas. Fonte conhecida: não é validação de domínio totalmente novo. Não há Aedes albopictus nesse pequeno conjunto negativo.

42 janelas positivas: **41 detectadas corretamente, uma declarada não-Aedes**. Treze negativas: **uma emissão falsa de Aedes, onze negativas corretas e uma incerta**. Acerto forçado global: 94,5%; sensibilidade aceita 97,6%; falso positivo aceito 7,7% (1/13). Os três arquivos positivos tiveram ao menos uma emissão correta; um dos três negativos teve uma emissão falsa. Amostra pequena, desigual entre gravações: um único arquivo Aedes contribui com 30 das 42 janelas positivas. Não chamar esse percentual de acurácia do aparelho.

## Áudio privado informado como enxame de Aedes

107 janelas: **20 AEDES_AEGYPTI, 72 NAO_AEDES_AEGYPTI, 13 INCERTO e 2 AUDIO_INVALIDO**. Sem treino ou ajuste de limiares com esse arquivo. A referência da espécie foi informada pelo usuário, sem confirmação independente nem anotação por instante. Não sabemos se todo trecho contém mosquito; portanto as 72 janelas negativas não são uma contagem demonstrada de falsos negativos. Surgiram indicações de Aedes, mas não há confirmação confiável do enxame, considerando os falsos positivos no HumBug. Nenhum áudio privado, original ou transformado, foi publicado.

## Decisão

Manter em avaliação offline; não substituir automaticamente o modelo anterior e não adaptar para Arduino. O foco binário trouxe ganhos de sensibilidade na UFRGS e MosquitoSong, mas ainda perdeu muitos Aedes e gerou falsos positivos relevantes. Próximo passo: mais negativos parecidos, especialmente Ae. albopictus, treinamento orientado a origens diferentes e gravações da caixa com o microfone real; manter novos indivíduos/dias/fontes para validação. Qualquer novo treino exige outro áudio inédito depois de congelar a versão.

Checkpoint SHA-256: `31825194c64b48b81f24329b2dd0aacf74aade6fae5d5ee00256af31c5f7e243`. Congelado em `2026-10-01T00:08:13.785727+00:00`. Limiares: negativo ≤ 0.484; Aedes ≥ 0.643; intervalo intermediário incerto. Escore não é probabilidade demonstrada de correção fora da calibração.
