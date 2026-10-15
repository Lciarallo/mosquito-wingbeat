# Revisão e direcionamento: refinamento da decisão ternária

Data: 14/10/2026  
Repositório: `mosquito-wingbeat`  
Referência da revisão: commit `20d6fe9`

> **Atualização:** a implementação posterior em `b562b35` foi conferida na [revisão da implementação](revisao_implementacao_ternaria_b562b35_2026_10_14.md). As correções seguintes e a nova execução estão no [relatório de entrega](correcoes_ternarias_2026_10_14.md). O texto abaixo preserva o diagnóstico e o direcionamento relativos a `20d6fe9`; as expressões “atual” e “atualmente” devem ser lidas nesse contexto histórico.

## Parecer

**Primeiro corrigir o isolamento da avaliação; depois buscar os 50/64 Aedes.** A proposta de melhorar a decisão por gravação faz sentido, mas o diagnóstico causal e a garantia estatística do plano original ainda não se sustentam.

Este documento registra a revisão do plano. As propostas abaixo ainda não foram implementadas ou retreinadas nesta revisão.

## 1. Análise de lacunas no diagnóstico dos 16 erros

A matriz da cascata para os 64 grupos positivos de *Aedes aegypti* é:

```text
[0, 16, 48]
```

Nenhum desses positivos foi classificado como ruído. Os 16 erros terminaram na classe `other_mosquito`.

Recalculando a decisão com as previsões salvas, os erros se distribuem assim:

| Fonte | Perdidos / total de grupos de Aedes |
|---|---:|
| Dryad | 1/20 |
| HumBug | 11/40 |
| MosquitoSong | 4/4 |
| **Total** | **16/64** |

Apenas **3 dos 16** têm score condicional entre 0,35 e 0,45. Os scores dos 16 perdidos vão de aproximadamente 0,153 a 0,462. Portanto, baixar o limiar naquela faixa não resolveria a maior parte dos erros.

Os números 128–190 encontrados nos grupos do MosquitoSong são **contagens de janelas provenientes de vários WAVs**, não segundos de uma gravação contínua. Nos quatro grupos positivos dessa fonte, há entre 74 e 108 arquivos por grupo.

A diluição temporal e o efeito de `gamma` merecem investigação, mas ainda não são causas demonstradas. O direcionamento é comparar `gamma=0.50` e `gamma=0.75` na validação interna, mantendo os demais parâmetros fixos.

Também é necessário distinguir duas métricas: a cascata teve **0/426 ruídos classificados como Aedes**, mas **91/426 ruídos classificados como outro mosquito**. O zero observado não representa ausência de falsos alarmes de presença de mosquito.

## 2. Corrigir o isolamento antes de otimizar a decisão

Substituir a exigência de apenas “OOF por `group`” por isolamento completo e validação aninhada.

O plano original mantém três problemas:

- O bloco com **54 grupos ligados por colisões PCM** precisa permanecer inteiro em uma dobra. No particionamento atual, seus grupos aparecem nas cinco dobras.
- Pesos de classe, duração e normalização devem ser calculados **somente no treino de cada divisão**. Atualmente, os pesos compostos são calculados antes da separação das dobras.
- Para avaliar a dobra externa, todas as previsões usadas para escolher pesos, atributos e limiares precisam vir de modelos treinados **sem qualquer participação dessa dobra externa**.

Reaproveitar as previsões OOF globais dos outros quatro folds não satisfaz a última condição: os modelos que produziram essas previsões podem ter treinado com a dobra externa que será avaliada.

O direcionamento é usar divisões internas dentro do treino externo. A dobra externa permanece excluída do ajuste dos modelos e de toda seleção de hiperparâmetros, atributos, fusão e limiares.

A unidade de isolamento também deve considerar os vínculos conhecidos entre arquivos, sessões e indivíduos. Disjunção de caminhos ou hashes não estabelece independência biológica.

Referência metodológica: [validação aninhada na documentação do scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## 3. Remover a promessa de garantia pelo slack

Limitar os falsos positivos internos a 0,75% e 1,75% pode ser uma hipótese experimental útil. **Isso não garante** que a aplicação cega ficará abaixo dos limites externos.

Controle empírico e garantia probabilística são coisas diferentes. Uma busca de limiares com limites internos arbitrários não constitui, por si só, um procedimento Neyman–Pearson com controle estatístico demonstrado.

Mesmo supondo observações independentes, **20 erros em 1.014 negativos** produzem:

- FPR observado de aproximadamente **1,97%**.
- Limite superior unilateral exato de 95% de aproximadamente **2,85%**.

Para **4 erros em 426 ruídos**, o FPR observado é aproximadamente **0,94%**, mas o limite superior unilateral de 95% é aproximadamente **2,14%**.

Esses valores ilustram a diferença entre a taxa observada e uma afirmação sobre a taxa populacional. As dependências entre grupos também precisam ser consideradas na avaliação real.

Descrever a etapa como **otimização com restrições empíricas**, registrando a incerteza e as violações observadas. Caso seja exigida uma garantia probabilística, definir um protocolo específico com calibração independente, tamanho amostral, nível de confiança e tratamento da seleção de regras.

Referência: [Tong, Feng e Li — Neyman–Pearson classification algorithms and NP-ROC](https://arxiv.org/abs/1608.03109).

## 4. Definir a sequência temporal e o uso dos atributos

A persistência deve ser calculada **dentro do mesmo WAV**, usando timestamps ordenados e verificando continuidade. Uma média móvel não pode atravessar arquivos diferentes do mesmo `group`.

Os dados misturam janelas curtas e longas. Por isso, “duas janelas” não representam sempre o mesmo intervalo de tempo. A agregação temporal deve ter uma definição explícita em segundos e tratar lacunas, sobreposições e metadados ausentes.

Separar três conceitos:

- **Grupo de isolamento:** mantém materiais relacionados na mesma partição.
- **Trecho temporal:** sequência contígua dentro de um WAV, usada para calcular persistência.
- **Unidade de decisão e avaliação:** unidade sobre a qual se calcula a matriz de confusão e se aplicam as metas.

Manter métricas por grupo para comparação com os resultados atuais. Se a unidade de avaliação mudar, recalcular os denominadores e identificar a mudança no relatório. Não assumir que os 1.504 grupos representam 1.504 eventos de voo independentes.

O plano lista novos atributos, mas não especifica como eles alteram a decisão. No código atual, alguns descritores já são calculados e não entram na cascata.

Começar com uma regra pequena e explícita. Depois, se necessário, avaliar um classificador simples por evento, treinado com previsões internas. Com apenas 64 grupos positivos, acrescentar muitos graus de liberdade exige cautela.

## 5. Separar critérios de desempenho dos testes de correção

As metas podem ser critérios para promover uma versão. Não devem obrigar o pesquisador a ajustar repetidamente o modelo até obter “245 testes verdes”.

Há um erro numérico no contrato proposto:

```text
20 / 1014 = 0.019723865...
```

Portanto, esta verificação reprova os próprios 20 erros que o plano pretende permitir:

```python
other_to_aedes_fpr <= 0.01972
```

Preferir contagens inteiras e denominadores explícitos:

| Critério experimental | Contagem pretendida | Taxa correspondente |
|---|---:|---:|
| Aedes corretamente identificados | Pelo menos 50/64 | Pelo menos 78,125% |
| Ruído identificado como Aedes | No máximo 4/426 | No máximo 0,938967% |
| Outro mosquito identificado como Aedes | No máximo 20/1014 | No máximo 1,972387% |

Essas contagens pressupõem a manutenção da unidade e da população de avaliação. Se houver alterações justificadas nos dados, os critérios precisam explicitar como se adaptam aos novos denominadores.

Os testes de software devem verificar:

- Isolamento por blocos PCM e demais vínculos conhecidos.
- Linhagem dos modelos e exclusão da dobra externa de toda seleção.
- Pesos e transformações ajustados somente no treino.
- Continuidade temporal e proibição de médias móveis entre WAVs diferentes.
- Consistência entre previsões, contagens, matrizes e métricas.
- Registro da configuração efetivamente usada no experimento.

O relatório experimental deve registrar aprovação ou reprovação das metas, preservando resultados desfavoráveis. A quantidade exata de testes aprovados não é uma meta científica.

Também incluir precisão como resultado central: atingir exatamente **50 verdadeiros positivos e 24 falsos positivos** produziria **67,57% de precisão**, abaixo dos 73,02% observados no boosting atual, embora com maior sensibilidade.

## 6. Sequência de implementação recomendada

1. **Preservar os resultados atuais.** Criar um diretório novo para a rodada, com configuração, partições, modelos, previsões e hashes.
2. **Corrigir o isolamento e produzir uma nova referência.** Reavaliar boosting, ensemble e cascata sob o mesmo protocolo corrigido.
3. **Comparar os valores de `gamma`.** Selecionar entre 0,50 e 0,75 exclusivamente na validação interna, com os demais parâmetros fixos.
4. **Avaliar a agregação temporal.** Calcular persistência por WAV e comparar a alteração com a referência corrigida, sem mudar simultaneamente todos os componentes.
5. **Selecionar a regra e os limiares internamente.** Registrar restrições, critérios de desempate e procedimento quando nenhum candidato satisfizer as restrições. A dobra externa permanece excluída.
6. **Comparar os modelos sob os mesmos limites de falsos positivos.** Incluir resultados por fonte, sensibilidade, precisão, F1, falsos positivos por classe e incerteza. Uma alteração que melhora dois acertos em 64 positivos precisa ser interpretada com essa escala amostral.
7. **Congelar a solução antes da confirmação externa.** Avaliar novos áudios independentes, com Aedes, outros mosquitos e ruído, sem escolher limiares ou versões pelo resultado desse teste.
8. **Documentar o resultado real.** Publicar a metodologia, o que foi executado, os resultados e as limitações. Uma rodada que não atinge as metas continua sendo um resultado válido e deve permanecer registrada.

O conjunto OOF já examinado serve para desenvolvimento retrospectivo. Uma nova execução aninhada corrige dependências computacionais, mas não transforma automaticamente esse conjunto conhecido em uma confirmação independente.

## Direcionamento para o implementador

> Maximizar a sensibilidade de Aedes sob limites pré-declarados de falsos positivos, com isolamento PCM e calibração aninhada; avaliar as metas sem prometer que serão atingidas. Preservar os resultados anteriores, comparar com referências sob o mesmo protocolo e congelar a solução antes da confirmação em novos áudios independentes.

## Evidências locais usadas na revisão

- [Treinamento ternário](../train_ternary_models.py).
- [Agregação e calibração por grupo](../evaluate_ternary_event_pipeline.py).
- [Previsões OOF atuais](../results/ternary_detection_2026_10_14/predictions_oof.csv).
- [Métricas por grupo atuais](../results/ternary_detection_2026_10_14/event_level_metrics.json).
- [Auditoria anterior de colisões PCM](../results/species_nested_2026_10_13/collision_audit.json).
- [Mapa anterior de grupos e blocos de colisão](../results/species_nested_2026_10_13/recording_predictions.csv).
- [Metadados das janelas](../results/sex_balanced_reliability_2026_10_01/segments.csv).

Os resultados numéricos deste documento se referem aos artefatos existentes no commit indicado. A implementação proposta poderá produzir resultados diferentes, inclusive inferiores, depois da correção da avaliação.
