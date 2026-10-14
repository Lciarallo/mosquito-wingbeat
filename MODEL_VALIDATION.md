# Protocolo de retreinamento e liberação

Rodada v2 de 13/10/2026: [relatório técnico](reports/ensemble_especies_v2_2026_10_13.md). Auditoria PCM uniu 613 grupos em 560 blocos, corrigindo colisões que atravessavam as partições históricas. Treinamento aninhado offline com normalização por janela, pesos espécie/aparelho, saliência, SHS multi-pico, MIL e rejeição própria de gênero. Nos mesmos 17 arquivos externos retrospectivos: espécie 4/17 (23.53%), gênero 8/17 (47.06%). Intervalos pareados, F1 por suporte e pela união, emissões e regressões estão no relatório. **Confiabilidade externa não atingida; sem liberação para Arduino.** Novos arquivos foram adquiridos somente depois do congelamento. Histórico e instalador preservados; 156 passed, 4117 warnings in 17.10s.

Rodada 09/10/2026: [ensemble com calibração separada](reports/ensemble_especies_calibracao_separada_2026_10_09.md). Três bases treinadas nos mesmos grupos de ajuste; pesos, correções de classe e limiares ajustados em calibração reservada; nenhuma base reentrou na calibração antes de avaliar o teste externo da dobra. Os dados históricos de todas as dobras continuam retrospectivos. O modelo final foi congelado antes de adquirir 17 arquivos ainda não usados de cinco espécies BioDCASE D3; acertou 4/17 sem rejeição e 2/8 emissões com limiar congelado. Não houve alteração pelo teste. A identidade biológica/sessão desses recortes não é estabelecida; não representam coleta confirmatória nova. **Segue reprovado para liberação: generalização por espécie não comprovada.** O comando antigo `optimize_ensemble_oof.py` passou a encaminhar para o experimento corrigido. Pesos anteriores, firmware e instalador foram preservados.

Revisão 08/10/2026 após atualizar o Git para `1ab88b8`: [avaliação por espécie](reports/revisao_git_classificacao_especies_2026_10_08.md). As métricas do ensemble calibrado são retrospectivas e a calibração reutiliza previsões-base cuja linhagem inclui a dobra externa. Exigir predições internas produzidas exclusivamente no treino externo antes de declarar validação OOF sem vazamento. O certificado de grupos/caminhos/hashes não verifica essa condição. Quatro reexecuções do áudio conhecido, com/sem prior Brasil e emulação, não confirmaram espécie. O experimento TinyML + score Dryad mede presença, não valida espécie nem treinamento conjunto de dois corpora. Continua experimental.

Revisão 07/10/2026: [auditoria científica](reports/auditoria_pesquisa_2026_10_07.md). O Dryad 10.5061/dryad.98d7s é acústico, gravado com celulares, e não o corpus óptico de Fanioudakis. A variante LOSO afetada foi repetida em `results/loso_corrected_2026_10_07/`; a auditoria das previsões está em `results/research_audit_2026_10_07/`. Continua experimental, sem demonstração de confiabilidade no cenário do usuário.

Rodada 03/10/2026: avaliações originais dos autores reexecutadas em `results/faithful_reproduction_2026_10_03/`. DR-BioL/DAT usam o pickle oficial e MC dropout; MosquitoSong+ mantém o notebook e os H5. Os treinos completos dos artigos não foram refeitos. Acurácia interna próxima da publicação não autoriza liberação: a auditoria do notebook MosquitoSong+ encontrou sobreposição entre arrays de treino/teste, e o diagnóstico externo dos pesos originais falhou no cenário mono. A agregação de 3 sub-janelas/10 folds é uma extensão declarada, não a reprodução do protocolo de aquisição de dois microfones. DR-BioL não teve avaliação exata de WAV externo por falta do extrator completo original. Arduino continua sem autorização para compilação, redução ou nova versão. Consulte [o relatório](reports/reproducao_estudos_originais_2026_10_03.md).

Instrução do usuário: não compilar nada para Arduino enquanto o modelo não
demonstrar confiabilidade. Ao terminar cada treinamento, procurar na internet
um áudio inédito para testar. O áudio privado do Drive foi informado pelo usuário
como um enxame de Aedes aegypti.

Atualização de escopo: o usuário pediu um modelo grande para computador.
Estratégias para reduzi-lo, encaixá-lo no Arduino, compilar ou empacotar essa
versão ficam para uma autorização posterior explícita, mesmo que os testes
apresentem melhora. Esta rodada se restringe ao treinamento e à avaliação offline.

1. Escolher os dados adicionais por procedência e metadados, sem escolher
   gravações pelos resultados do classificador.
2. Separar treinamento, validação, calibração e teste por gravação de origem.
   Todos os trechos do mesmo arquivo original permanecem na mesma partição.
   Exemplos já examinados que entrarem no treino são adaptação, não validação.
3. Congelar pesos, frontend e limiares; registrar SHA-256 e horário.
4. Somente depois do congelamento, procurar e baixar uma nova gravação externa
   com espécie identificada. Conferir procedência e ausência da gravação no
   treino/validação/calibração/teste. Se forem necessários critérios de duração,
   defini-los antes de executar o classificador. Não repetir ou concatenar
   sons curtos para produzir uma identificação artificial.
5. Mostrar acertos, erros e rejeições, comparando com a versão anterior. Verificar
   também regressões no corpus antigo. Um acerto isolado não demonstra
   confiabilidade das 20 classes, dos enxames ou de outros aparelhos.
6. Após usar um áudio externo para tomar decisões sobre um novo treinamento,
   ele deixa de ser um teste inédito para esse ciclo: procurar outro após
   congelar a próxima versão. Não retocar os limiares com o resultado do novo teste.
7. Enquanto a decisão de confiabilidade estiver negativa, manter a versão em
   avaliação offline. Não gerar nova distribuição de firmware, compilar para
   Arduino, atualizar o instalador ou publicar binários dessa versão.

O áudio privado é uma referência de espécie fornecida pelo usuário, sem
confirmação entomológica independente. O usuário informou que todo o áudio
contém Aedes aegypti: usar essa referência no denominador completo, incluindo
rejeições e janelas inválidas. Isso não equivale a uma anotação independente de
cada voo. Não publicar a gravação privada.

Rodada inicial corrigida: [validação corrigida de 01/10/2026](reports/validacao_corrigida_2026_10_01.md)
e [decisão corrigida](results/corrected_2026_10_01/reliability_decision.json),
**experimental, não liberado**. As cinco dobras OOF foram separadas por
gravação; normalização e limiares foram congelados antes dos novos testes.
Os novos áudios acústicos desta rodada são apenas negativos. Portanto, não
medimos sensibilidade em Aedes acústico inédito nem comprovamos confiabilidade.

A rede profunda do outro modelo foi preservada para esta auditoria. Seu
frontend repete internamente entradas curtas de 0,3 s até o tamanho esperado;
essa limitação está explicitada no relatório e não satisfaz o item 4 para uma
nova aquisição positiva. Não se deve interpretar a duração repetida como
observação adicional. Uma próxima arquitetura deve aceitar a duração real,
ser treinada com essa entrada e receber outro teste após congelamento.

A versão Arduino já publicada é anterior aos retreinamentos.

Rodada de 03/10/2026: contraste por espécie entre Dryad/HumBug, alinhamento MMD,
representação com múltiplos picos e enxames simulados apenas no treino. Cinco
candidatos novos foram congelados antes de adquirir 32 arquivos acústicos D3,
sem coincidências de bytes ou fragmentos PCM não constantes conhecidos.
Escolhido pela validação: `ridge_contrast_swarm`; precisão 72,9%, sensibilidade
10,3%, especificidade 87,2% e macro F1 27,0% no teste novo; Drive 30/107.
SVM com os novos descritores detectou 45,0% dos Aedes novos e 57/107 no Drive,
mas não foi promovida com base nesses testes. Nenhum candidato atingiu a
sensibilidade mínima de 60% na nova amostra. Segue experimental e fora do Arduino.
[Relatório e limitações](reports/estrategias_contrastivas_enxames_2026_10_03.md).

**Estado da rodada de reprodução, ampliação e adaptação de 01/10/2026:**
[relatório final](reports/confiabilidade_binaria_aedes_adaptacao_2026_10_01.md) e
[decisão da adaptação](results/sex_balanced_reliability_2026_10_01/reliability_decision.json).
Ainda experimental e reprovado. A nova aquisição contém positivos e negativos
acústicos: precisão 17,5%, recall 2,8% e F1 Aedes 4,8% no modelo escolhido na
validação. Foram selecionados 168 arquivos novos depois do congelamento; 151
foram mantidos pela regra de colisões PCM. Não houve liberação para Arduino.
As três gravações MosquitoSong de fêmeas negativas antes usadas na validação
viraram treino na adaptação e não são mais evidência independente nessa rodada.

## Critério binário de confiabilidade — métricas além da acurácia

Considerar o [artigo indicado pelo usuário](https://www.dio.me/articles/por-que-a-acuracia-accuracy-nao-e-a-melhor-metrica-para-avaliar-o-desempenho-de-classificadores):
uma maioria de negativos pode produzir acurácia alta mesmo sem detectar Aedes.
O relatório deve trazer matriz de confusão, precisão de Aedes, sensibilidade,
F1 de Aedes, macro-F1, especificidade, cobertura e falsos positivos por táxon.
Espécie desconhecida não recebe rótulo negativo. Áudio inválido não conta como
rejeição correta. Áudio contendo apenas Aedes mede sensibilidade; não comprova
precisão ou especificidade para detecção binária.

O patamar solicitado de 60% é um mínimo experimental, não uma certificação:
exigir simultaneamente precisão, sensibilidade, macro-F1 e especificidade de
ao menos 60% no novo teste completo **e no subconjunto apenas de mosquitos**.
Exigir sensibilidade de Aedes e especificidade em cada táxon negativo de ao
menos 60%, além de intervalos por gravação com limites inferiores de 95% de
ao menos 60% nas métricas principais. Ruído fácil não pode compensar a falha
entre espécies. Mostrar quando a identidade biológica não é fornecida: um
intervalo por arquivo não equivale a um intervalo por mosquito independente.

Esses limites foram registrados antes do teste da rodada ampliada, em
`results/reliability_2026_10_01/protocol.json`. Limiares são escolhidos somente
na calibração, e a seleção do modelo usa somente a validação. Um modelo que
supere os mínimos offline ainda exige validação independente nas condições
reais pretendidas; atingir 60% isoladamente não autoriza Arduino.

A precisão depende da frequência de Aedes na amostra. Projeções para outras
prevalências são cenários, não precisão observada em campo. ROC-AUC e average
precision dos scores brutos são diagnósticos complementares; a decisão usa
também as emissões reais e mantém os inválidos no denominador.

Correção conceitual do artigo: para scores fixos, aumentar o limiar nunca
aumenta a sensibilidade; costuma reduzir falsos positivos. Melhorar o modelo
pode melhorar precisão e sensibilidade simultaneamente. Ver as definições
na [documentação oficial do Google](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall).

## Regras adicionais após a auditoria de 07/10/2026

- Definir separadamente presença de mosquito, identificação de espécie e identificação de *Aedes aegypti*. `not_aedes` inclui outras espécies de Aedes: o nome da coluna não significa ausência de todo o gênero.
- Reservar gravações, indivíduos e sessões antes de criar janelas. Quando não houver identidade biológica, declarar que o agrupamento disponível é apenas uma aproximação e auditar derivados entre bases.
- Para avaliar uma fonte inédita, excluir toda essa fonte do ajuste, incluindo representações aprendidas, normalização, escolha de características e calibração. Nenhum embedding pré-treinado no teste pode entrar como representação supostamente independente.
- Relatar protocolos indutivos e transdutivos separadamente. Normalização com estatísticas do lote avaliado precisa ser declarada; não representa um sistema capaz de avaliar uma gravação isolada sem esse lote.
- Registrar sementes, hashes de dados, código e modelos, candidatos realmente executados, identidade das linhas das previsões e horário do congelamento. Comparar somente modelos no mesmo protocolo, nas mesmas gravações e com os mesmos denominadores.
- Reportar número de gravações positivas e negativas, cobertura, matriz de confusão, métricas por espécie/fonte e intervalos por gravação. Precisão indefinida e réplicas bootstrap sem determinada classe não viram automaticamente zero ou evidência de aprovação. Intervalos com poucos grupos continuam frágeis.
- O conjunto já examinado permanece exploratório. Para escolher uma nova versão, usar seleção interna por fontes/sessões de desenvolvimento; deixar a confirmação final em outra aquisição independente, sem escolher arquitetura, limiar ou filtros por seu resultado.
- Dimensionar a nova coleta pela precisão desejada dos intervalos e pela variabilidade entre indivíduos/sessões, não pelo total de janelas. Medir sensibilidade por evento e falsos alarmes por hora com anotações temporais no aparelho/ambiente pretendidos; resultados por janela não substituem essas medidas.
