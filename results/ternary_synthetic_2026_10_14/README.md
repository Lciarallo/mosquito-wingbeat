# Dataset sintético e teste do detector ternário

Data: 14/10/2026. Modelo: `ternary_verified_2026_10_14_corrections`. Nenhum modelo foi retreinado nem teve limiares alterados.

## Resultado observado

Foram criados e avaliados **2.740 WAVs**. A cascata final classificou **0/160 negativos totalmente artificiais como Aedes**; 158 foram classificados como ruído e dois como outros mosquitos. No teste com doadores fora do treino de cada modelo, a detecção de Aedes caiu de **36/60 no controle** para **8/60 com ruído branco a +20 dB** e **2/60 com ruído rosa a +10 dB**.

O teste encontrou resistência aos falsos alarmes de Aedes nesses negativos e uma fragilidade de sensibilidade às perturbações simuladas. Esses resultados foram preservados sem escolher novos limiares ou regenerar exemplos por seus acertos.

## Análise de lacunas e desenho do teste

O resultado anterior agrega várias janelas por grupo. Faltava medir de forma controlada a alteração de decisão após acrescentar ruído, filtrar, atenuar ou misturar sinais. A simulação usa trechos reais anotados para preservar a referência de espécie; frequência de um oscilador, sozinha, não recebe rótulo de Aedes.

- **300 grupos doadores:** 60 Aedes, 120 outros mosquitos e 120 ruídos; correspondem a 297 blocos PCM.
- **2.400 clipes semissintéticos:** oito versões pareadas de cada trecho, incluindo o controle sem alteração.
- **180 misturas:** Aedes e outro mosquito simultâneos em relações de +10, 0 e −10 dB.
- **160 negativos inteiramente sintéticos:** oito perfis com 20 sementes cada.
- Áudio mono a 16 kHz. Doadores com **0,992 s reais**, sem preencher ou repetir trechos; negativos procedurais de quatro segundos.
- Famílias procedurais: ruído branco, rosa, marrom, harmônicos de rede elétrica, harmônicos mecânicos, chirp, cliques e zumbido mecânico modulado. São controles de engenharia, sem pretensão de reproduzir aparelhos específicos.

As receitas, sementes, doadores e WAVs foram registrados antes da inferência. A escolha de trechos foi aleatória dentro dos grupos selecionados, sem consultar scores. Os doadores negativos foram distribuídos por fonte/táxon. Todos os 60 grupos positivos do desenvolvimento participaram.

O SNR é a razão entre o RMS do **trecho inteiro** e o componente acrescentado; não mede o SNR isolado das asas. Componentes de mistura foram escalados antes de uma redução conjunta de pico quando necessária, preservando a razão entre eles. O filtro é Butterworth de quarta ordem, passa-baixa de 1,2 kHz; a passagem usa envelope gaussiano com piso de 0,05, centro de 0,50 s e desvio de 0,13 s.

## Isolamento e unidade de avaliação

A avaliação principal de derivados reais usa o modelo externo congelado da dobra que deixou **todos os componentes da mistura** fora do ajuste, incluindo seus blocos PCM. Os ruídos procedurais usam o modelo final congelado. Os parâmetros dos modelos permanecem os mesmos do experimento original.

Também foram calculadas previsões do modelo final nos derivados reais como diagnóstico secundário. Esse modelo foi ajustado nos grupos doadores; seus resultados não são uma estimativa independente. As colunas do CSV distinguem `heldout_` de `final_`.

Cada observação aqui é um clipe de 0,992 s, e não a agregação original de todas as janelas de um grupo. Por isso, **36/60 no controle não substitui nem contradiz os 47/60 da avaliação anterior por grupo**. A comparação controlada é entre cada perturbação e este controle de janela.

As 2.740 versões não representam 2.740 indivíduos independentes. Os resultados por condição compartilham os mesmos doadores. A prevalência de Aedes nas condições pareadas é de 20%, escolhida pelo desenho; precisão não deve ser transportada para a prevalência do corpus original ou do campo.

## Condições pareadas

Modelo principal: cascata congelada correspondente à dobra do doador. Os denominadores incluem os clipes tecnicamente inválidos, registrados como ausência de classificação.

| Condição | Aedes detectados | Sensibilidade | Ruído → Aedes | Outro → Aedes | Cobertura | Aedes no modelo final¹ |
|---|---:|---:|---:|---:|---:|---:|
| Controle | 36/60 | 60,00% | 0/120 | 6/120 | 100,00% | 60/60 |
| Branco +20 dB | 8/60 | 13,33% | 0/120 | 2/120 | 100,00% | 14/60 |
| Rosa +10 dB | 2/60 | 3,33% | 0/120 | 1/120 | 100,00% | 9/60 |
| Rosa 0 dB | 0/60 | 0,00% | 0/120 | 0/120 | 100,00% | 0/60 |
| Rosa −5 dB | 0/60 | 0,00% | 0/120 | 0/120 | 100,00% | 0/60 |
| Atenuação 18 dB | 33/60 | 55,00% | 0/120 | 5/120 | 84,00% | 56/60 |
| Passa-baixa 1,2 kHz | 3/60 | 5,00% | 0/120 | 0/120 | 99,67% | 5/60 |
| Envelope de passagem | 39/60 | 65,00% | 0/120 | 6/120 | 99,67% | 60/60 |

¹ Diagnóstico com doadores presentes no treino do modelo final, sem valor de confirmação independente.

A atenuação produziu 48 clipes sem janela válida; o filtro e o envelope produziram mais um caso cada. Os 50 permaneceram nos denominadores, incluindo os positivos. O baixo FPR nas condições mais difíceis acompanha uma queda da detecção de positivos.

## Misturas de mosquitos

| Relação Aedes / outro, em RMS | Aedes detectados fora do treino | Sensibilidade | Modelo final¹ |
|---|---:|---:|---:|
| +10 dB | 14/60 | 23,33% | 22/60 |
| +0 dB | 7/60 | 11,67% | 10/60 |
| -10 dB | 5/60 | 8,33% | 7/60 |

Todas essas misturas contêm um componente com referência de Aedes. Elas medem detecção em coocorrência simulada; não permitem estimar especificidade nessa seção.

## Negativos totalmente procedurais

| Perfil | Ruído → Aedes | Ruído → qualquer mosquito |
|---|---:|---:|
| white | 0/20 | 0/20 |
| pink | 0/20 | 0/20 |
| brown | 0/20 | 0/20 |
| mains_harmonics | 0/20 | 0/20 |
| fan_harmonics | 0/20 | 1/20 |
| chirp | 0/20 | 0/20 |
| clicks | 0/20 | 0/20 |
| modulated_fan | 0/20 | 1/20 |

As taxas descrevem estes geradores e estas sementes. Não são uma estimativa da variedade de ruídos reais.

## Conferências e testes

- **51 testes relacionados aprovados**, 135 avisos, em 4,02 s, incluindo mistura por componentes independentes, SNR, controle sem alteração, separação de blocos PCM, frontend idêntico à inferência e tratamento de áudio inválido.
- As características dos controles foram comparadas ao cache original. A diferença absoluta máxima foi `0,0001220703125`, a mediana foi zero e **nenhuma das 300 decisões da cascata mudou**. Ambos os caminhos deram 36/60 Aedes. Isso descarta divergência de frontend como explicação desses controles.
- A razão RMS registrada ficou a menos de `3,56e-15 dB` da receita antes da escrita FLOAT. Todos os WAVs tiveram duração e taxa de amostragem conferidas.
- Foram preservados hashes do gerador, do dataset, dos modelos, das previsões e dos resultados. O protocolo foi encerrado antes de iniciar a inferência.

O gerador novo e os testes foram adicionados sem alterar o frontend, os classificadores ou os limiares da rodada anterior. Este conjunto é um teste de robustez por simulação. Se for usado futuramente para ajustar o modelo, deixa de ser um teste reservado dessa nova versão.

## Artefatos e reprodução

- [Áudios WAV](audio/), [manifesto e hashes](manifest.csv), [doadores e linhagem](parents.csv), [protocolo](protocol.json).
- [Previsões de todos os modelos](predictions.csv), [métricas completas](metrics.json), [resumo CSV](metrics_summary.csv), [auditoria de frontend](frontend_audit.json).
- [Gráfico das condições](robustness.png).

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python evaluate_ternary_synthetic.py \
  --run results/ternary_verified_2026_10_14_corrections \
  --out results/ternary_synthetic_reproducao --phase all

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_ternary_synthetic.py tests/test_ternary_verified.py tests/test_ternary_detector.py
```

A reprodução exige o repositório, as entradas originais e os modelos congelados disponíveis localmente. O diretório de saída deve ser novo. Os dados e a inferência foram executados nesta entrega; não houve retreinamento.
