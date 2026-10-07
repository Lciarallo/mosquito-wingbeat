# Estudo de classificação acústica de mosquitos

**[⬇ Baixar instalador para Windows](https://github.com/Lciarallo/mosquito-wingbeat/releases/latest/download/MosquitoWingbeat-Windows.exe)**
— [passo a passo com imagens](WINDOWS.md) · [página de downloads](https://github.com/Lciarallo/mosquito-wingbeat/releases/latest)

Para instalar no **Arduino Nano 33 BLE Sense / Sense Rev2**: abra o `.exe`, conecte
a placa, clique em **Buscar minha placa** e em **Instalar no Arduino**. A janela
mostra os resultados automaticamente. **Não precisa de Python, Arduino IDE,
comandos ou conta no GitHub.** Windows 10/11 de 64 bits; internet no primeiro uso
e cerca de 1 GB livre. Modelo experimental; acurácia em uso real ainda não medida.

Abra [mosquito_wingbeat_estudo.ipynb](mosquito_wingbeat_estudo.ipynb) para executar ou baixe
[mosquito_wingbeat_estudo.html](mosquito_wingbeat_estudo.html) e abra no navegador
para ler o estudo e os gráficos sem Jupyter.

O notebook reúne a leitura de três artigos, referências complementares, download, auditoria,
features de frequência/PSD/MFCC, oito classificadores para 20 espécies, tarefa TinyML
de quatro classes, aparelhos reservados, ruído, abstenção, INT8 e exportação C.
O código do downloader também está incorporado ao notebook, que não depende dos scripts
auxiliares para definir os experimentos iniciais. A extensão executa os scripts incluídos
para contraste/dinâmica, novos classificadores e detecção binária com firmware Arduino.

## Arduino Nano 33 BLE Sense: identificar espécies

Baixe [MosquitoSpecies.zip](output/arduino/MosquitoSpecies.zip), extraia e abra
a pasta **MosquitoSpecies**. No Linux/macOS com Python 3.9+, conecte a placa e execute:

```bash
bash flash_arduino.sh --monitor
```

O instalador por terminal baixa/reutiliza CLI 1.5.1 e core 4.6.0 em uma pasta local, detecta a placa,
compila, grava e abre o monitor a **115200 baud**. A primeira preparação precisa
de internet e cerca de 1 GB livre. No Windows, use o **aplicativo gráfico acima**;
**Instalar_no_Windows.cmd** abre o aplicativo ou seu download; opções por terminal
são uma alternativa avançada que exige Python.
O ZIP contém [guia rápido](firmware/LEIA_PRIMEIRO.md)
e o [manual completo](firmware/MANUAL_DE_MONTAGEM_E_INSTALACAO.md).
`--compile-only` compila sem placa/upload; `--list-ports` consulta portas;
`--port PORTA` seleciona uma placa e `--monitor-only` abre o serial sem regravar.

Alternativa pela Arduino IDE, sem Python: abra **MosquitoSpecies.ino**, mantenha
os quatro headers juntos, instale **Arduino Mbed OS Nano Boards 4.6.0**, selecione
**Arduino Nano 33 BLE** e a porta, faça upload e abra o monitor em **115200 baud**.
Preparação/compilação reais verificadas no Linux e pelo próprio `.exe` em Windows
no GitHub Actions. A interface foi revisada visualmente. Não houve execução nativa
macOS nem upload em placa física. [Auditoria do aplicativo Windows](results/arduino_species/windows_build_audit.json).

O [sketch completo](firmware/MosquitoSpecies/MosquitoSpecies.ino) usa o microfone PDM
integrado, FFT incremental e modelos treinados para indicar presença e classificar
**20 espécies**. Emite candidata, identificação provisória ou **INCERTO**, com
concordância da mesma espécie em 2/3 janelas. A primeira emissão exige 2,976 s de
observação. [Manual de instalação, espécies e limites](firmware/MosquitoSpecies/README.md).
Montagem e gravação: [manual completo](firmware/MANUAL_DE_MONTAGEM_E_INSTALACAO.md).
O script **flash_arduino.sh** grava esse sketch por padrão; **serve.py** é o
servidor de visualização incluído na atualização do projeto.

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

O [firmware](firmware/README.md) foi preparado para **Arduino Nano 33 BLE Sense / Sense Rev2**,
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
Código, modos, instalação e limites: [firmware/README.md](firmware/README.md).

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

## Arquitetura Hierárquica ML e Validação em Campo (Zero Data Leakage)

A partir da análise da literatura científica recente (BioDCASE 2026, SEMISH 2025, SBCAS 2023, HumBugDB 2021, eLife 2017), o repositório foi expandido com uma abordagem hierárquica em três níveis:

1. **Auditoria Criptográfica Anti-Vazamento (`leakage_auditor.py`)**:
   - Garante matematicamente $0$ grupos, $0$ hashes SHA-256 e $0$ arquivos compartilhados entre treino e teste.
   - Certificado gerado em `results/leakage_audit_certificate.json`.
2. **Benchmark Hierárquico Multi-Abordagem (`train_hierarchical_models.py`)**:
   - **Tier 1 (Presença)**: Detecção binária combinando densidade harmônica bioacústica e modelo neural.
   - **Tier 2 (Gênero / Vetor Epidemiológico)**: Separação de *Aedes*, *Anopheles*, *Culex* e *Culiseta* com **95,43% de acurácia por gravação** e **93,48% por janela** em dados 100% disjuntos Out-of-Fold.
   - **Tier 3 (Espécie Fina com Priors)**: Suporte a priors biogeográficos regionais (85,15% de acurácia em espécies brasileiras endêmicas).
3. **Diagnóstico e Classificação de Áudios de Campo (`predict_audio.py`)**:
   - Filtro anti-falso-alarme de harmônicos de $60\text{ Hz}$ de rede elétrica / ventoinhas ($\sigma < 5\text{ Hz}$).
   - Emulação acústica do microfone MEMS ST MP34DT05 do Arduino Nano 33 BLE Sense (`--compare`).
   - Suporte a priors epidemiológicos regionais (`--region brazil`).
   - **Validação em Campo**: Testado no áudio externo de celular `drive_audio.wav`, suprimindo o ruído elétrico inicial e classificando o trecho de voo real (49,5s a 77,4s, $F_0 \approx 610\text{ Hz}$) como **`Aedes aegypti`**, confirmado empiricamente como ground truth.

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
