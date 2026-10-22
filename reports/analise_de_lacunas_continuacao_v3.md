# Análise de lacunas para continuar a v3

Inspeção em 20/10/2026, antes de extrair características ou treinar modelos desta continuação.

Estado herdado: commit `be13d6a`; alterações locais do Claude em `v3_manifest.py` e na emenda A1. Os quatro ZIPs HumBugDB já foram baixados e extraídos; há manifesto e componentes de duplicação. O diretório `data/cache/v3/` está vazio. Não há classificador v3 treinado nem resultados v3 de classificação.

## Preservar

- Protocolo v3 original e seu congelamento, v2, classificadores anteriores, pesos e notebook.
- Dados reais, hashes, componentes PCM/espectrais e exclusões herdadas.
- Resultados anteriores como experimentos retrospectivos com limitações já documentadas; não descrevê-los como experimentos sem valor.

## Corrigir antes do treino

1. **Separação:** dois valores de `cluster_merged` do manifesto herdado ocorrem no treino e no teste reservado. Colocar os registros de desenvolvimento desses clusters em quarentena, sem mudar os registros já reservados.
2. **Proveniência:** o teste reservado contém também espécies/origens BioDCASE não demonstradas como UFRGS. Chamá-lo de teste relacionado ao D3/UFRGS, discriminando arquivos/componentes já avaliados e origem não confirmada. Os clusters provisórios não equivalem a indivíduos.
3. **Critérios:** eliminar o relaxamento de 2% para 5% de FPR entre outros mosquitos. Usar metas observadas de FPR <=2% em outros mosquitos, <=1% em ruído e sensibilidade >=20%, também no resultado final. Intervalos de confiança devem ser apresentados; atingir uma taxa pontual não estabelece uma garantia populacional.
4. **Ruído:** o teste reservado herdado não contém ruído. Reservar, antes de qualquer treino, uma amostra determinística de clusters exclusivamente de ruído, por origem, para avaliação final complementar. Eles não participam da seleção ou calibração. Esse teste não representa ruído de um ambiente de campo novo.
5. **Calibração:** separar clusters de ajuste e calibração em cada treino externo. Parâmetros são escolhidos apenas no ajuste; calibradores e limiares usam só a calibração; nenhum modelo é retreinado sobre ela antes de avaliar o teste correspondente.
6. **Extração:** limitar a fila de decodificação e salvar checkpoints retomáveis. A implementação inicial submete todos os áudios à fila e só grava ao terminar, arriscando memória excessiva e perda de horas de trabalho.
7. **Comparabilidade:** publicar Perch sem/com normalização espectral e a referência tabular; manter os números históricos separados, pois não é um ensaio que isola apenas a mudança de partição.

## Limites a manter explícitos

- Poucos clusters positivos no teste; nem 197 arquivos nem 568 arquivos representam esse número de mosquitos independentes.
- A duração e o preenchimento com zeros podem revelar a origem. Não repetir áudio curto; incluir diagnóstico baseado apenas em duração, sem usá-lo para escolher o detector.
- Seleção de métodos e inspeções anteriores impedem chamar os 15 áudios UFRGS conhecidos de teste cego novo.
- Um resultado ruim permanece resultado; não ajustar a configuração depois de abrir o teste final.
- Entrega científica e funcionamento em laboratório não estabelecem validação de campo.
