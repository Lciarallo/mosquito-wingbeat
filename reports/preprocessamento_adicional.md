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
| highpass_120 | mosquitosong | 352 | 5.7% | 0.0% | nan% |
| highpass_120 | ufrgs | 271 | 1.8% | 0.0% | nan% |
| highpass_120 | fresh_humbug_retrospective | 85 | 68.2% | 3.5% | 100.0% |
| bandpass_150_3500 | dryad | 7845 | 36.4% | 8.7% | 64.6% |
| bandpass_150_3500 | humbug | 383 | 32.1% | 1.3% | 0.0% |
| bandpass_150_3500 | mosquitosong | 352 | 1.1% | 0.0% | nan% |
| bandpass_150_3500 | ufrgs | 271 | 0.0% | 0.4% | 0.0% |
| bandpass_150_3500 | fresh_humbug_retrospective | 85 | 8.2% | 0.0% | nan% |
| bandpass_250_2500 | dryad | 7845 | 20.4% | 0.0% | nan% |
| bandpass_250_2500 | humbug | 383 | 3.7% | 0.0% | nan% |
| bandpass_250_2500 | mosquitosong | 352 | 0.6% | 0.0% | nan% |
| bandpass_250_2500 | ufrgs | 271 | 2.6% | 0.0% | nan% |
| bandpass_250_2500 | fresh_humbug_retrospective | 85 | 0.0% | 0.0% | nan% |
| spectral_mild | dryad | 7845 | 60.8% | 24.1% | 93.4% |
| spectral_mild | humbug | 383 | 91.4% | 58.5% | 97.3% |
| spectral_mild | mosquitosong | 352 | 41.2% | 0.9% | 33.3% |
| spectral_mild | ufrgs | 271 | 5.5% | 5.2% | 0.0% |
| spectral_mild | fresh_humbug_retrospective | 85 | 98.8% | 54.1% | 100.0% |
| bandpass_spectral | dryad | 7845 | 34.1% | 8.0% | 59.5% |
| bandpass_spectral | humbug | 383 | 34.2% | 1.6% | 0.0% |
| bandpass_spectral | mosquitosong | 352 | 1.1% | 0.0% | nan% |
| bandpass_spectral | ufrgs | 271 | 0.0% | 0.7% | 0.0% |
| bandpass_spectral | fresh_humbug_retrospective | 85 | 9.4% | 0.0% | nan% |

A redução espectral leve melhora o HumBug de **90,3% para 91,4%**, com cobertura de 55,9% para 58,5%. Nos cinco Aedes retrospectivos, mantém 84/85 acertos forçados e aumenta emissões corretas de **41 para 46**, sem emissão errada. Porém, reduz o Dryad de **62,6% para 60,8%** e UFRGS de **6,6% para 5,5%**, com nenhuma emissão correta na UFRGS. Não existe melhoria consistente entre fontes.

No ruído HumBug, o original e a redução espectral leve não emitiram espécie em 915 janelas. O highpass emitiu espécie em **11/915** janelas; o bandpass 150–3.500 Hz em **31/915**; a combinação em **27/915**. Filtrar não é sinônimo de reduzir falsos alarmes: a normalização interna pode tornar resíduos relativamente mais destacados. Não houve emissão de Aedes nos cinco arquivos negativos não-Aedes da UFRGS em nenhum método; os rótulos dessas pastas não permitem validar as outras espécies.

## Decisão

**Não substituir o pré-processamento padrão.** Os filtros adicionais não resolveram seu enxame e alguns introduziram regressões relevantes. Esta é uma análise de inferência com rede treinada no pipeline original, não prova de que esses filtros jamais ajudariam. Aplicá-los apenas na inferência muda a distribuição da entrada.

Uma próxima rodada deve aplicar o mesmo processamento no treino, validação e calibração, escolher a abordagem sem consultar o teste final e, depois de congelar a nova versão, buscar outro áudio inédito. Os métodos e resultados desta análise não foram usados para ajustar pesos, recalibrar limiares ou liberar firmware. A CNN permanece experimental e a adaptação para Arduino continua aguardando autorização posterior explícita.

[Protocolo](../results/preprocessing/protocol.json) · [Resultados completos](../results/preprocessing/comparison.csv) · [Resumo privado sem áudio](../results/preprocessing/private_audio_summary.json) · [Auditoria](../results/preprocessing/audit.json) · [Decisão](../results/preprocessing/decision.json).
