# Estudo de classificação acústica de mosquitos

Abra [mosquito_wingbeat_estudo.ipynb](mosquito_wingbeat_estudo.ipynb) para executar ou baixe
[mosquito_wingbeat_estudo.html](mosquito_wingbeat_estudo.html) e abra no navegador
para ler o estudo e os gráficos sem Jupyter.

O notebook reúne a leitura de três artigos, referências complementares, download, auditoria,
features de frequência/PSD/MFCC, oito classificadores para 20 espécies, tarefa TinyML
de quatro classes, aparelhos reservados, ruído, abstenção, INT8 e exportação C.
O código do downloader também está incorporado ao notebook, que não depende dos scripts
auxiliares para definir os experimentos.

## Relatório técnico

Leia o [relatório em PDF](output/pdf/Relatorio_Classificacao_Acustica_Mosquitos.pdf) ou a
[versão em Markdown](reports/relatorio_classificacao_mosquitos.md). O documento reúne métodos,
resultados, comparação com CNNs e artigos, explicabilidade disponível, limitações e próximos passos.
Sua geração recalcula as métricas dos oito classificadores a partir das previsões salvas.

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

## Evidências de execução

- 22 células de código executadas; 0 erros.
- Asserções de separação por grupos e formas de onda idênticas.
- Comparação numérica dos frontends SciPy/Torch.
- 400 previsões C/Python correspondentes.
- SHA-256 e configuração dos arquivos/resultados nos manifests.
- Figuras revisadas visualmente.

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
