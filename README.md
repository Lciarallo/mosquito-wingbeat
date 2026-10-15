# Estudo de classificação acústica de mosquitos

Rodada v2 de 13/10/2026: [relatório técnico](reports/ensemble_especies_v2_2026_10_13.md). Auditoria PCM uniu 613 grupos em 560 blocos, corrigindo colisões que atravessavam as partições históricas. Treinamento aninhado offline com normalização por janela, pesos espécie/aparelho, saliência, SHS multi-pico, MIL e rejeição própria de gênero. Nos mesmos 17 arquivos externos retrospectivos: espécie 4/17 (23.53%), gênero 8/17 (47.06%). Intervalos pareados, F1 por suporte e pela união, emissões e regressões estão no relatório. **Confiabilidade externa não atingida; sem liberação para Arduino.** Novos arquivos foram adquiridos somente depois do congelamento. Checkpoints históricos foram preservados.

## Presença, gênero e espécie — comparação de 09/10/2026

[Comparação dos três níveis](reports/tres_niveis_2026_10_09.md): o gênero acerta 96,49% das gravações internas, mas 41,18% das 17 externas; espécie, 71,70% e 23,53%. O detector histórico separado de presença tem sensibilidade de 72,63% e falsos positivos em 29,17% das janelas de ruído no modo padrão. Os modelos/testes de presença e identificação são diferentes. Ainda não há demonstração de confiabilidade em campo.

## Ensemble com calibração separada — 09/10/2026

**Correção implementada e treinamento concluído; identificação por espécie continua experimental.** SVM, ExtraTrees e HistGradientBoosting foram treinados com grupos de ajuste, calibração e teste separados. O ensemble corrigido tem **71,70% de acerto por gravação** no Dryad conhecido; com rejeição, **88,89% de acerto nas emissões e 61,69% de cobertura**. A vantagem sobre HistGradientBoosting no mesmo protocolo é pequena e seu intervalo pareado inclui zero.

Depois do congelamento, adquirimos 17 arquivos ainda não usados de cinco espécies, com auditoria de bytes/PCM. O ensemble acertou **4/17** na escolha forçada e **2/8 identificações emitidas**; não acertou os quatro *Ae. aegypti*. A separação metodológica foi corrigida, mas a transferência acústica continua falhando. Esses arquivos pertencem a um corpus conhecido e não equivalem a uma nova campanha independente. **136 testes passaram.**

[Relatório, comparação e resultados por espécie](reports/ensemble_especies_calibracao_separada_2026_10_09.md) · [Partições, pesos congelados e avaliações](results/species_nested_2026_10_09/) · [Treinamento corrigido](nested_species_ensemble.py). `optimize_ensemble_oof.py` agora encaminha para esse experimento, em pasta separada; os resultados do otimizador anterior continuam históricos.

## Atualização do Git e avaliação por espécie — 08/10/2026

Atualizado para `1ab88b8`, preservando o trabalho local. **Promissor no corpus conhecido; classificação confiável em novas gravações ainda não demonstrada.** Recalculamos os seis métodos: HistGradientBoosting tem 73,46% de acerto por gravação e 52,69% por janela nas 20 espécies. O ensemble calibrado registra 74,34% por gravação, mas sua calibração requer validação aninhada para eliminar dependência da dobra externa. A auditoria de separação de arquivos não verifica esse caminho.

No áudio conhecido do usuário, repetimos o pipeline atual com prior global/Brasil e entrada bruta/emulada: **nenhuma espécie confirmada nas quatro combinações**. O prior Brasil muda a candidata dominante para *Aedes aegypti*; sem ele, a candidata dominante é *Culex tarsalis*. Isso não comprova reconhecimento acústico da espécie. Também corrigimos a interpretação do experimento TinyML + score Dryad, que mede presença e não comprova melhoria por espécie.

[Revisão completa e resultados por espécie](reports/revisao_git_classificacao_especies_2026_10_08.md) · [Métricas e quatro reexecuções](results/git_review_2026_10_08/) · [Relatório corrigido do experimento](reports/relatorio_experimento_mistura_datasets.md). **123 testes existentes passaram**. Os resultados são retrospectivos; não houve novo treinamento ou validação em placa física.

## Revisão crítica da pesquisa — 07/10/2026

**A pesquisa permite análise experimental; a confiabilidade no cenário real ainda não foi demonstrada.** Corrigimos a classificação do Dryad, que é áudio de celulares, e repetimos a variante afetada do LOSO. Recalculamos precisão, sensibilidade, F1 e intervalos por gravação dos 30 resultados históricos. Também distinguimos normalização com acesso ao lote de teste de inferência em gravações novas. A melhoria é metodológica; não há aprovação do detector.

[Auditoria, resultados e próximos experimentos](reports/auditoria_pesquisa_2026_10_07.md) · [Métricas recalculadas](results/research_audit_2026_10_07/metrics.csv) · [Repetição corrigida](results/loso_corrected_2026_10_07/loso_metrics.csv).

Os resultados anteriores foram preservados. Os novos cálculos usam `research_metrics.py`; os módulos das avaliações congeladas continuam com seus hashes originais. Para conferir o código: `.venv/bin/python -m pytest -q tests`. A reanálise é retrospectiva e não substitui uma coleta independente.

## Reexecução dos estudos com pesos originais — 03/10/2026

Foram executados o `Evaluation.py` original DR-BioL/DAT com o pickle oficial e todas as células do notebook oficial MosquitoSong+ com seus dez H5, WAVs e ruídos. Acurácia reproduzida: **85,26% DR-BioL, 79,39% DAT e 92,78% MosquitoSong+**. São avaliações de pesos dos autores; os treinamentos completos dos artigos não foram refeitos. A auditoria do notebook MosquitoSong+ encontrou gravações e amostras sobrepostas nas partições chamadas treino/teste. A comparação externa separada não comprovou confiabilidade no cenário do usuário. Nenhuma compilação ou alteração de firmware.

[Relatório](reports/reproducao_estudos_originais_2026_10_03.md) · [Notebook executado](reproducao_estudos_originais_2026_10_03.ipynb) · [HTML](reproducao_estudos_originais_2026_10_03.html) · [Reexecução e versões](results/faithful_reproduction_2026_10_03/REPRODUCING.md).

## Contraste entre bases e enxames simulados — 03/10/2026

**Experimental; não incorporado ao Arduino.** Cinco candidatos novos: três redes de 5,65M parâmetros, SVM e ExtraTrees com descritores de múltiplos picos. Escolhido pela validação: `ridge_contrast_swarm`. Teste acústico adquirido depois do congelamento: precisão 72.9%, sensibilidade 10.3%, especificidade 87.2%; Drive 30/107. Consulte os intervalos e as limitações antes de interpretar confiabilidade.

[Relatório](reports/estrategias_contrastivas_enxames_2026_10_03.md) · [Notebook executado](estrategias_contrastivas_enxames_2026_10_03.ipynb) · [HTML](estrategias_contrastivas_enxames_2026_10_03.html) · [Decisão](results/cross_domain_strategies_2026_10_03/reliability_decision.json).

## Confiabilidade binária — nova avaliação de 01/10/2026

**Ainda experimental: o detector não atingiu os critérios de confiabilidade.**
Incorporamos a avaliação por precisão, sensibilidade, F1, especificidade,
matriz de confusão e intervalos por gravação, conforme o artigo indicado pelo
usuário. Acurácia geral acima de 60% não basta para aprovação.

Foram adquiridas 670 gravações para ampliar treino/validação/calibração,
comparadas redes com até 9,65 milhões de parâmetros e modelos clássicos,
e executada uma adaptação com fêmeas de outras espécies e representações PCEN.
Cada rodada foi congelada antes de baixar novos arquivos para teste.
Na última, foram selecionados 168 arquivos, com 151 mantidos pela auditoria:
**precisão 17,5%, sensibilidade 2,8% e F1 Aedes 4,8%** no modelo escolhido
pela validação. No áudio privado, essa adaptação detectou 0/107 janelas;
o modelo anterior de árvores detecta 77/107, mas também erra outros mosquitos.
Uma gravação só de Aedes não mede precisão ou especificidade binária.

[Relatório final da adaptação](reports/confiabilidade_binaria_aedes_adaptacao_2026_10_01.md) ·
[Notebook](confiabilidade_binaria_aedes_adaptacao_2026_10_01.ipynb) ·
[Leitura em HTML](confiabilidade_binaria_aedes_adaptacao_2026_10_01.html) ·
[Decisão e motivos de reprovação](results/sex_balanced_reliability_2026_10_01/reliability_decision.json).

[Reproduções científicas e rodada ampliada](reports/confiabilidade_binaria_aedes_2026_10_01.md) ·
[Notebook ampliado](confiabilidade_binaria_aedes_2026_10_01.ipynb).
Os notebooks apenas leem resultados. Pesos anteriores e firmware de pesquisa
foram preservados; os downloads históricos abaixo não incorporam estes modelos.

```bash
.venv/bin/python -m unittest tests.test_sex_balanced_reliability tests.test_reliability tests.test_reproduction tests.test_corrected_validation
.venv/bin/python evaluate_sex_balanced_reliability.py
.venv/bin/python summarize_sex_balanced_reliability.py
```

## Validação corrigida — 01/10/2026

**Os modelos para computador continuam experimentais.** Corrigimos o vazamento
entre dobras do stacking, a diferença de escala entre treino e inferência, a
normalização dependente do lote e a escolha de limiares no próprio teste.
Também excluímos os controles de ruído que repetiam o treino e preservamos
espécies desconhecidas como não rotuladas.

O stacking caiu de 90,6% para 69,5% de acurácia balanceada no Dryad; o ensemble
ficou em 49,9% nas misturas sintéticas. O teste óptico novo, com calibração
separada, ficou entre 54,5% e 61,25%. Novos áudios acústicos foram selecionados
após o congelamento, mas são apenas negativos: a sensibilidade em Aedes
acústico inédito ainda não foi medida nesta rodada.

[Relatório corrigido](reports/validacao_corrigida_2026_10_01.md) ·
[Notebook de resultados](validacao_corrigida_2026_10_01.ipynb) ·
[Leitura no navegador](validacao_corrigida_2026_10_01.html) ·
[Decisão de confiabilidade](results/corrected_2026_10_01/reliability_decision.json).

O notebook corrigido apenas lê resultados; não treina nem compila. Para
reavaliar os modelos congelados, execute na raiz do projeto:

```bash
.venv/bin/python corrected_validation.py --phase evaluate
.venv/bin/python evaluate_corrected_fresh.py
.venv/bin/python summarize_corrected_validation.py
```

Os checkpoints anteriores foram preservados. As distribuições Arduino abaixo
são históricas; esta rodada não atualizou firmware ou instalador.

Abra [mosquito_wingbeat_estudo.ipynb](mosquito_wingbeat_estudo.ipynb) para executar ou baixe
[mosquito_wingbeat_estudo.html](mosquito_wingbeat_estudo.html) e abra no navegador
para ler o estudo e os gráficos sem Jupyter.

O notebook reúne a leitura de três artigos, referências complementares, download, auditoria,
features de frequência/PSD/MFCC, oito classificadores para 20 espécies, tarefa TinyML
de quatro classes, aparelhos reservados, ruído, abstenção, INT8 e exportação C.
O código do downloader também está incorporado ao notebook, que não depende dos scripts
auxiliares para definir os experimentos iniciais. A extensão executa os scripts incluídos
para contraste/dinâmica, novos classificadores e detecção binária com firmware Arduino.

## Pré-processamento adicional — análise exploratória

Testamos cinco filtros adicionais com pesos e limiares fixos. Nenhum identificou
o enxame privado como Aedes. A redução espectral leve ajudou modestamente em
HumBug, mas piorou outros corpora; filtros de frequência causaram regressões.
O processamento padrão foi mantido.
[Comparação completa](reports/preprocessamento_adicional.md).

## Nova rodada com múltiplos datasets — avaliação offline

Baixamos Figshare, MosquitoSong, UFRGS, BioDCASE Development e Synthetic Swarm.
A CNN foi ajustada com **4.344 janelas adicionais de treino**, incluindo sons
reais de 0,3 s, sem repetir ou preencher o áudio. UFRGS inteira permanece fora
do treino. Há ganhos parciais, mas **a identificação na UFRGS ainda falha**.
Após congelar o modelo, testamos cinco novos Aedes: ambos os modelos identificam
quatro arquivos; a nova versão rejeita mais janelas. No enxame privado, as dez
emissões erradas desapareceram, mas Aedes continua sem identificação.
[Relatório e todos os resultados](reports/retreinamento_multifonte.md) ·
[Decisão de confiabilidade](results/multisource/reliability_decision.json).
Nenhuma atualização de Arduino ou instalador nesta rodada.

## CNN grande para computador — avaliação offline

Treinamos uma CNN residual de **3,33 milhões de parâmetros**, mantendo os dados
anteriores e a ampliação HumBugDB. O acerto sem rejeição subiu de 33,5% para 62,7%
no Dryad e de 74,2% para 89,0% no HumBug (quatro classes nesse teste).
**O novo áudio externo de Aedes não foi reconhecido**; no enxame privado,
foram emitidas identificações incorretas. Ainda não há confiabilidade em campo.
[Relatório completo](reports/modelo_grande_cnn.md) ·
[Decisão e limitações](results/large_cnn/reliability_decision.json).
A adaptação desta CNN ao Arduino aguarda autorização posterior explícita.
O instalador Windows e o firmware publicados permanecem da versão anterior.

## Retreinamento com HumBugDB — ainda não liberado

Treinamos uma nova versão com os dados anteriores e **307 trechos adicionais**.
Ela melhorou no teste HumBug, mas piorou no corpus antigo e não identificou uma
nova gravação de Aedes aegypti do MosquitoSong+, pesquisada após congelar os pesos.
Também não identificou o áudio de enxame fornecido pelo usuário.
**A versão ampliada não foi compilada nem liberada para Arduino.**
[Relatório e comparação](reports/treinamento_ampliado.md) ·
[Protocolo de validação](MODEL_VALIDATION.md) ·
[Decisão de confiabilidade](results/expanded/reliability_decision.json).

## Avaliação experimental do firmware Arduino

O sketch experimental usa o microfone PDM integrado, FFT incremental e modelos treinados para indicar presença e classificar
**20 espécies**. Emite candidata, identificação provisória ou **INCERTO**, com
concordância da mesma espécie em 2/3 janelas. A primeira emissão exige 2,976 s de
observação.

| Avaliação do modelo exportado | Resultado |
|---|---:|
| Escolha forçada nas 7,845 janelas positivas do seu teste | 42.2% geral; 54.0% balanceada |
| Rejeição + confirmação, acerto entre identificações emitidas | 87.9%, 231 identificações |
| Cobertura com confirmação | 3.9% de 5,983 endpoints positivos contíguos |

O acerto maior vem com **muita rejeição**; o aparelho deverá responder incerto com
frequência. Os escores não garantem que uma identificação esteja correta. O padrão
é a dobra 2, primeira que atingiu a meta de calibração; não foi escolhido
pelo teste. O agregado das três dobras usa modelos próprios e teve 39.5% geral /
52.1% balanceada na escolha forçada. Estes são testes do corpus de celulares, sem
anotação de cada voo ou validação no Arduino físico.

Compilou para o alvo com **125,120 bytes de programa** /
**59,664 bytes globais** (exclui pico de pilha/heap).
C++/Python: **7901/7901**
decisões com features iguais e **456/456**
no caminho completo PCM. **Não houve upload ou teste físico**: microfone, distância,
latência, autonomia e alarmes/hora continuam sem medição. A dobra exportada não
tem ruído contíguo elegível para testar a confirmação. Resultados/auditorias:
[results/arduino_species](results/arduino_species/); seção 19 do notebook.

## Arduino: detector de presença da versão anterior

O firmware histórico foi preparado para **Arduino Nano 33 BLE Sense / Sense Rev2**,
com microfone PDM, frontend incremental FFT, rede neural de 16 unidades e confirmação 2/3.
O objetivo é **presença acústica**, separado do reconhecimento das 20 espécies.

| Modo | Recall positivo por janela | Falso positivo no ruído por janela | Recall médio por fonte | Falso positivo médio por fonte |
|---|---:|---:|---:|---:|
| Balanceado 0,5 | 82.7% | 35.2% | 81.0% | 21.8% |
| Calibração FPR 5% | 48.9% | 20.3% | 48.3% | 9.2% |
| Calibração FPR 10% | 72.6% | 29.2% | 70.9% | 15.9% |

São rótulos fracos por arquivo, sem anotação de cada voo. As metas de calibração
não foram garantidas no teste. No modo padrão de 10%, perde cerca de **27,4%** dos
trechos positivos e marca **29,2%** dos trechos de ruído como candidatos antes da
confirmação. A confirmação temporal não demonstrou controle confiável de ruídos
persistentes; o subconjunto contínuo tem somente duas fontes de ruído.

Compilação: **100,832 bytes de programa** e
**59,584 bytes globais**, CLI 1.5.1/core 4.6.0.
O frontend completo C++/Python teve **532/532**
decisões iguais. O Uno R3/Nano clássico não comporta esse firmware.
**Não houve teste numa placa física**: faltam pico de RAM, latência, distância,
autonomia, sensibilidade por evento e falsos alarmes/hora.

Coletar áudio com a placa no local de uso, anotar atividade real e gravar horas sem
mosquito são os próximos passos necessários para reduzir e medir os erros de campo.
Os arquivos de firmware e resultados permanecem em `firmware/` e `results/arduino/`.

## Relatório técnico

Leia o [relatório em PDF](output/pdf/Relatorio_Classificacao_Acustica_Mosquitos.pdf) ou a
[versão em Markdown](reports/relatorio_classificacao_mosquitos.md). O documento reúne métodos,
resultados, comparação com CNNs e artigos, explicabilidade disponível, limitações e próximos passos.
Sua geração recalcula as métricas dos oito classificadores a partir das previsões salvas.
Também verifica os cinco métodos adicionais e as taxas do detector Arduino.

Para regenerar o relatório sem treinar modelos, instale **requirements-report.txt** e execute
**build_report.py**. São necessários fontconfig e DejaVu Sans. A auditoria específica do documento
está em **reports/report_manifest.json**.

## Dados completos baixados

- DOI: 10.5061/dryad.98d7s.
- 20/20 arquivos RAR íntegros: 1,232,964,187 bytes.
- Origem dos arquivos: espelho Zenodo 4964774 com o mesmo DOI; nomes/tamanhos/MD5
  conferidos contra metadados oficiais Dryad. As rotas Dryad recusaram o download automático.
- Arquivos compactados: **data/archives/**; conteúdo completo: **data/raw/**.
- Ruídos reais TinyML: **data/noise/**, com commit e SHA-256 fixados.
- Inventário de 1537 arquivos; 24,405 janelas
  analisadas, incluindo ruído. Benchmark de 20 espécies: 23,877
  janelas em 569 grupos.
- Licença declarada do dataset Dryad/Zenodo: CC0.

O repositório público contém notebook/HTML, código, metadados, tabelas, figuras,
checkpoints pequenos e exportação INT8/C. Os áudios completos, caches, PDFs originais
e a serialização grande de ExtraTrees ficam na pasta local do estudo. Em outra máquina,
o próprio notebook baixa os áudios completos e refaz a análise. A auditoria acima descreve
a execução registrada, não a presença dos áudios em um clone novo.

**prepare_delivery.py** gera o mesmo conteúdo publicável em
**dist/mosquito-wingbeat-notebook.zip**, com verificação de SHA-256.

## Resultados locais

Novos métodos, **mesmas dobras/grupos** da etapa inicial:

| Método | Acurácia balanceada por janela | Macro recall por grupo |
|---|---:|---:|
| ExtraTrees original (referência) | 48.9% | 68.3% |
| SVM RBF | 55.6% | 76.7% |
| ExtraTrees + contraste/dinâmica | 53.8% | 72.8% |
| HistGradientBoosting + contraste/dinâmica | 58.5% | 79.4% |
| MLP compacta + contraste/dinâmica | 51.6% | 72.8% |
| Ensemble fixo (RBF + árvores + boosting) | 59.0% | 77.9% |

O boosting melhorou o macro recall por grupo de **68,3% para 79,4%**. O ensemble
teve a maior acurácia balanceada por janela, **59,0%**; boosting, **58,5%**.
É uma reanálise exploratória do corpus já usado, sem validação externa independente
ou prova de superioridade sobre os artigos. A nova MLP também foi comparada.
O RBF usa 112 características; os modelos com contraste/dinâmica usam 293.

A [importância por permutação](results/arduino/permutation_importance.csv) das famílias
de características do detector avalia sensibilidade preditiva em dados reservados.
Não constitui atribuição causal ou explicação local SHAP/Grad-CAM.

Resultados iniciais preservados:

| Método | Acurácia balanceada por janela | Macro recall por grupo |
|---|---:|---:|
| Majoritária | 4.9% | 5.0% |
| F0: soma de densidades | 32.7% | 42.9% |
| F0: log-verossimilhança | 41.8% | 65.9% |
| PSD + SVM linear | 41.5% | 60.8% |
| MFCC + regressão logística | 47.9% | 64.0% |
| MFCC/PSD/F0 + ExtraTrees | 48.9% | 68.3% |
| CNN1D compacta | 35.8% | 43.0% |
| CNN2D + ruído real no treino | 40.8% | 51.8% |

Essas métricas usam três divisões por grupo/gravação. Não equivalem aos percentuais
publicados pelos artigos. O teste entre aparelhos mostra limites de generalização;
algumas espécies estão concentradas em poucas sessões e há rótulos temporais fracos.
O notebook documenta auditoria, cobertura, intervalos descritivos e limites de interpretação.
Não é uma validação de campo ou de desempenho no microcontrolador.

O classificador linear INT8 usa payload numérico de **892 bytes**,
excluindo frontend/firmware/RAM. A exportação C foi compilada, com
**400/400 previsões iguais ao Python**.
O arquivo C corresponde à dobra 0; não é um modelo revalidado para produção.

## Reproduzir

Ambiente verificado: Python 3.14.7; versões exatas em **results/run_manifest.json**.
GPU usada: AMD Radeon RX 9070 XT. GPU é opcional e os checkpoints aceitam leitura em CPU.
A primeira execução refaz os dados/features; execuções seguintes reutilizam caches
e checkpoints somente quando as assinaturas coincidem. Reserve vários GB em disco.

Dependências de sistema: libarchive, ffmpeg/ffprobe e compilador C (cc).
Para um ambiente novo no Linux:

    git clone https://github.com/Lciarallo/mosquito-wingbeat.git
    cd mosquito-wingbeat
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt jupyterlab
    .venv/bin/python -m ipykernel install --prefix .venv --name mosquito-wingbeat
    JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python -m jupyterlab

Abra o notebook, selecione esse kernel e execute todas as células.
Para executar e exportar HTML pela linha de comando:

    JUPYTER_PATH="$PWD/.venv/share/jupyter" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python execute_notebook.py

**build_notebook.py** reconstrói o notebook a partir do código e da discussão; **execute_notebook.py**
executa e exporta; **prepare_delivery.py** confere e monta a entrega.

Para executar somente a extensão, depois de preparar o acervo no notebook:

    .venv/bin/python improve_device.py
    .venv/bin/python train_arduino.py
    .venv/bin/python verify_arduino.py
    .venv/bin/python probe_arduino_noise.py
    .venv/bin/python train_arduino_species.py
    .venv/bin/python verify_arduino_species.py

Previsões extensas são publicadas em **results/arduino/*.csv.gz** e podem ser lidas
diretamente com pandas.read_csv. Checkpoints grandes em joblib permanecem locais;
o notebook os regenera. Modelos numéricos compactos e headers C++ estão publicados.

## Evidências de execução

- 30 células de código executadas; 0 erros.
- Asserções de separação por grupos e formas de onda idênticas.
- Comparação numérica dos frontends SciPy/Torch.
- 400 previsões C/Python correspondentes.
- 532 decisões do frontend/modelo Arduino C++/Python correspondentes.
- Sketch compilado para Nano 33 BLE; sem teste em placa física.
- Modelo de 20 espécies: 7901 decisões equivalentes e 456 trechos PCM verificados; sketch compilado.
- SHA-256 e configuração dos arquivos/resultados nos manifests.
- Figuras revisadas visualmente.

## Pipeline hierárquico experimental

`train_hierarchical_models.py` calcula métricas de espécie e gênero agregando probabilidades já salvas; não treina uma nova hierarquia. Os 95,43% por gravação são de quatro gêneros e vêm do ensemble calibrado com a dependência metodológica descrita na [revisão de 08/10](reports/revisao_git_classificacao_especies_2026_10_08.md). Os 85,15% com prior regional se restringem às janelas de três espécies selecionadas pelo rótulo verdadeiro.

`leakage_auditor.py` verifica interseções de grupos, caminhos e hashes declarados. Seu certificado não estabelece isolamento da calibração, dos limiares, dos indivíduos ou da linhagem das previsões. Portanto, o status histórico `ZERO_LEAKAGE_VERIFIED` não autoriza uma garantia geral de ausência de vazamento.

`predict_audio.py` usa as MLPs Arduino históricas, filtros harmônicos, persistência temporal e priors opcionais. Esse pipeline tem pesos diferentes do ensemble calibrado do benchmark. `--region brazil` altera o resultado, mas os pesos regionais não têm calibração externa demonstrada. A reexecução no áudio conhecido não confirmou espécie; sua referência de *Ae. aegypti* é informada pelo usuário, sem confirmação independente estabelecida nesta revisão.

## Fontes

- Artigo principal: https://elifesciences.org/articles/27854
- Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.98d7s
- Espelho do mesmo DOI: https://zenodo.org/records/4964774
- TinyML: https://doi.org/10.1145/3524458.3547258
- Dados/código TinyML: https://github.com/Mjrovai/wingbeat-mosquito-tinyml
- Deep learning óptico: https://doi.org/10.23919/EUSIPCO.2018.8553542
- MosquitoSong+: https://doi.org/10.1371/journal.pone.0310121

O diretório [literature/](literature/README.md) registra as fontes e os hashes dos PDFs
consultados; os arquivos originais são mantidos apenas na cópia local.
Wingbeats óptico é outro dataset e não foi baixado nem usado para treinamento neste estudo.

## Licenças das fontes

O dataset Dryad/Zenodo declara CC0. Os ruídos TinyML são obtidos do repositório dos
autores no commit registrado em **data/noise_manifest.json**, respeitando os termos
da fonte. Artigos e outros materiais de terceiros mantêm seus próprios direitos;
o repositório publica a análise e as citações, sem redistribuir os PDFs ou textos integrais.

### Detector binário de Aedes aegypti (experimental, computador)

Nova CNN dedicada a **Aedes aegypti versus não-Aedes**, com estados de rejeição. Não identifica as outras espécies e ainda não é confiável para uso no aparelho: houve falsos positivos relevantes no HumBug e baixa sensibilidade na UFRGS. [Resultados e protocolo](reports/detector_binario_aedes.md), também na seção 25 do notebook.

```bash
.venv/bin/python aedes_binary.py caminho/gravacao.wav --output tmp/teste_aedes
```

O checkpoint local está em `results/aedes_binary/models/aedes_binary.pt` (ignorado pelo Git). Os scripts `train_aedes_binary.py`, `fetch_fresh_aedes_binary.py` e `evaluate_aedes_binary.py` registram treinamento, seleção externa após congelamento e avaliação. Nenhuma compilação/adaptação para Arduino nesta rodada.

### Outras técnicas de ML para o detector binário

Regressão logística, SVM RBF, Extra Trees e gradient boosting foram treinados com características acústicas e comparados à CNN binária. [Resultados, explicabilidade e limites](reports/comparacao_ml_binario_aedes.md), seção 26 do notebook. Ganhos de sensibilidade podem aumentar falsos positivos; todos permanecem experimentais, sem liberação para Arduino.

```bash
.venv/bin/python aedes_classical.py caminho/gravacao.wav --model svm_rbf --output tmp/teste_svm
```

Modelos locais em `results/aedes_classical/models/` (ignorados pelo Git). Sem `--model`, usa a família escolhida somente pela validação, `gradient_boosting`.
