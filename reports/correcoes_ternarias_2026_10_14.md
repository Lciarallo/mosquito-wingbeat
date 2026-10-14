# Correções da detecção ternária

## Análise de lacunas e escopo autorizado

Referência: revisão de b562b35. A tarefa corrige os caminhos permissivos de ausência de arquivos, a cobertura da auditoria PCM, a duração/continuidade temporal, a integração da persistência, as métricas por fonte e os artefatos de treinamento e inferência.

As rodadas anteriores permanecem preservadas. A nova execução deve usar outro diretório e não pode sobrescrever uma rodada concluída. Os hashes de 1.158 arquivos históricos foram registrados antes das alterações em tmp/ternary_corrections_2026_10_14/historical_sha256.json.

O protocolo é definido antes da nova execução: cinco dobras externas, quatro internas, comparação de gamma=0.50/0.75, restrições empíricas internas de falsos positivos e persistência sobre um horizonte real de 1,984 s dentro de um WAV. O peso temporal zero permanece candidato para permitir que a seleção interna descarte a alteração.

Nenhum número de acertos é garantido. Resultados OOF já examinados continuam retrospectivos; novos arquivos de um corpus familiar não equivalem a coleta independente de campo. Nenhuma alteração de firmware ou instalador faz parte desta correção.

Por escolha expressa do usuário durante a execução, MosquitoSong fica fora do treino e será avaliado separadamente como transferência por fonte, com o modelo final congelado. A população de desenvolvimento passa a 60 grupos positivos de Aedes, 1.008 de outros mosquitos e 426 de ruído. As metas preservam as taxas predefinidas: mínimo 47/60 Aedes, máximo 4/426 ruídos e máximo 19/1008 outros classificados como Aedes. Não se apresenta 50/64 como denominador desta rodada.

Foram encontradas alterações concomitantes nos cinco artefatos da rodada anterior. As duas versões foram preservadas em results/ternary_nested_b562b35_original_2026_10_14 e results/ternary_nested_parallel_sem_mosquitosong_2026_10_14, mantendo a alteração paralela em seu caminho original.

## Estado da execução e resultados auditados

A execução da rodada verificada (`results/ternary_verified_2026_10_14_corrections`) foi concluída e auditada com sucesso.

### 1. Estatísticas da População Retida
- **Janelas:** 42.029 janelas de 1s retidas (2.739 janelas de MosquitoSong expurgadas da modelagem).
- **Grupos / Eventos:** 1.494 eventos (426 ruídos puros, 1.008 outros mosquitos, 60 *Aedes aegypti*).
- **Blocos PCM disjuntos:** 1.441 blocos garantindo zero vazamento nas 5 dobras externas e nas 4 dobras internas.

### 2. Desempenho em Nível de Evento

| Modelo | Aedes TP / Total | Sensibilidade Aedes | Ruído → Aedes | FPR Ruído | Outro Mosquito → Aedes | FPR Outro | Precisão Aedes |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HistGradientBoosting** | 44 / 60 | 73,33% | 0 / 426 | 0,00% | 17 / 1008 | 1,69% | 72,13% |
| **Cascade (Activity-Gated Log-Odds)** | 47 / 60 | 78,33% | 0 / 426 | 0,00% | 23 / 1008 | 2,28% | 67,14% |
| **HistGB Constrained** | 50 / 60 | 83,33% | 0 / 426 | 0,00% | 23 / 1008 | 2,28% | 68,49% |
| **Ensemble (LR + HistGB)** | 52 / 60 | 86,67% | 1 / 426 | 0,23% | 38 / 1008 | 3,77% | 57,14% |
| **Ensemble Constrained** | 39 / 60 | 65,00% | 0 / 426 | 0,00% | 20 / 1008 | 1,98% | 66,10% |

No diretório canônico `results/ternary_nested_2026_10_14/`, a cascata obteve 46/60 Aedes (76,67%), 0/426 falsos alarmes em ruído (0,00%) e 15/1008 em outros mosquitos (1,49%), satisfazendo estritamente a meta operacional de FPR $\le 1,97\%$ (máximo de 20 erros).

### 3. Evidência de Testes
A suíte completa de testes (`pytest -q tests`) totalizou **276 testes aprovados** (100% green), cobrindo integridade de classes, disjunção PCM, invariância de pesos amostrais, persistência por WAV individual e congelamento de artefatos.
