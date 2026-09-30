"""Create a lightweight research bundle; full raw data remain available locally."""
from pathlib import Path
import hashlib
import json
import zipfile
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
manifest = json.loads((RESULTS / "run_manifest.json").read_text())
execution = json.loads((RESULTS / "notebook_execution.json").read_text())
metrics = pd.read_csv(RESULTS / "species_metrics.csv")
improvements = pd.read_csv(RESULTS / "improvements/species_metrics.csv")
arduino = pd.read_csv(RESULTS / "arduino/metrics.csv")
native_audit = json.loads((RESULTS / "arduino/native_audit.json").read_text())
compile_audit = json.loads((RESULTS / "arduino/compile_audit.json").read_text())
species_protocol = json.loads((RESULTS / "arduino_species/protocol.json").read_text())
species_metrics = pd.read_csv(RESULTS / "arduino_species/metrics.csv")
species_selected = species_metrics.loc[species_metrics.model.eq("Escolha por validação interna")].iloc[0]
species_fold = species_protocol["default_export_fold"]
species_default = pd.read_csv(RESULTS / "arduino_species/fold_metrics.csv")
species_default = species_default.loc[species_default.fold.eq(species_fold)].iloc[0]
species_selective = pd.read_csv(RESULTS / "arduino_species/default_selective_metrics.csv").iloc[-1]
species_native = json.loads((RESULTS / "arduino_species/native_audit.json").read_text())
species_compile = json.loads((RESULTS / "arduino_species/compile_audit.json").read_text())
quantization = json.loads((RESULTS / "int8_audit.json").read_text())
notebook = nbformat.read(ROOT / "mosquito_wingbeat_estudo.ipynb", as_version=4)
nbformat.validate(notebook)
assert execution["error_outputs"] == 0
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
assert not any(o.output_type == "error" for c in notebook.cells for o in c.get("outputs", []))
assert manifest["verified_archives"] == 20
assert quantization["c_python_matching_predictions"] == 400
assert len(metrics) == 8
assert species_native["feature_only_matching_threshold_decisions"]==species_native["feature_only_test_windows"]
assert species_native["complete_pcm_matching_threshold_decisions"]==species_native["complete_pcm_test_windows"]
assert species_compile["compile_succeeded"] and not species_compile["physical_board_tested"]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# The execution audit is written after the notebook finishes; hash it last.
result_files = [p for p in RESULTS.rglob("*") if p.is_file() and p.name != "sha256.json"]
(RESULTS / "sha256.json").write_text(json.dumps(
    {str(p.relative_to(ROOT)): sha256_file(p) for p in result_files}, indent=2))

rows = "\n".join(f"| {row.model} | {row.balanced_accuracy:.1%} | {row.recording_macro_recall:.1%} |"
                 for row in metrics.itertuples())
improved_rows = "\n".join(f"| {row.model} | {row.balanced_accuracy:.1%} | {row.recording_macro_recall:.1%} |"
                         for row in improvements.itertuples())
arduino_rows = "\n".join(f"| {row.mode} | {row.window_recall:.1%} | {row.window_false_positive_rate:.1%} | {row.source_mean_recall:.1%} | {row.source_mean_false_positive_rate:.1%} |"
                        for row in arduino.loc[arduino.model.eq("Escolha por validação interna")].itertuples())
readme = f"""# Estudo de classificação acústica de mosquitos

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

O instalador baixa/reutiliza CLI 1.5.1 e core 4.6.0 em uma pasta local, detecta a placa,
compila, grava e abre o monitor a **115200 baud**. A primeira preparação precisa
de internet e cerca de 1 GB livre. No Windows x86/x64 com Python, abra o atalho
**Instalar_no_Windows.cmd**. O ZIP contém [guia rápido](firmware/LEIA_PRIMEIRO.md)
e o [manual completo](firmware/MANUAL_DE_MONTAGEM_E_INSTALACAO.md).
`--compile-only` compila sem placa/upload; `--list-ports` consulta portas;
`--port PORTA` seleciona uma placa e `--monitor-only` abre o serial sem regravar.

Alternativa pela Arduino IDE, sem Python: abra **MosquitoSpecies.ino**, mantenha
os quatro headers juntos, instale **Arduino Mbed OS Nano Boards 4.6.0**, selecione
**Arduino Nano 33 BLE** e a porta, faça upload e abra o monitor em **115200 baud**.
Preparação/compilação reais verificadas no Linux; não houve execução nativa
Windows/macOS nem upload em placa física.

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
| Escolha forçada nas {int(species_default.positive_windows):,} janelas positivas do seu teste | {species_default.accuracy:.1%} geral; {species_default.balanced_accuracy:.1%} balanceada |
| Rejeição + confirmação, acerto entre identificações emitidas | {species_selective.accepted_positive_class_accuracy:.1%}, {int(species_selective.accepted_positive_windows)} identificações |
| Cobertura com confirmação | {species_selective.positive_coverage:.1%} de {int(species_selective.positive_windows):,} endpoints positivos contíguos |

O acerto maior vem com **muita rejeição**; o aparelho deverá responder incerto com
frequência. Os escores não garantem que uma identificação esteja correta. O padrão
é a dobra {species_fold}, primeira que atingiu a meta de calibração; não foi escolhido
pelo teste. O agregado das três dobras usa modelos próprios e teve {species_selected.accuracy:.1%} geral /
{species_selected.balanced_accuracy:.1%} balanceada na escolha forçada. Estes são testes do corpus de celulares, sem
anotação de cada voo ou validação no Arduino físico.

Compilou para o alvo com **{species_compile['program_storage_bytes']:,} bytes de programa** /
**{species_compile['global_static_memory_bytes']:,} bytes globais** (exclui pico de pilha/heap).
C++/Python: **{species_native['feature_only_matching_threshold_decisions']}/{species_native['feature_only_test_windows']}**
decisões com features iguais e **{species_native['complete_pcm_matching_threshold_decisions']}/{species_native['complete_pcm_test_windows']}**
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
{arduino_rows}

São rótulos fracos por arquivo, sem anotação de cada voo. As metas de calibração
não foram garantidas no teste. No modo padrão de 10%, perde cerca de **27,4%** dos
trechos positivos e marca **29,2%** dos trechos de ruído como candidatos antes da
confirmação. A confirmação temporal não demonstrou controle confiável de ruídos
persistentes; o subconjunto contínuo tem somente duas fontes de ruído.

Compilação: **{compile_audit['program_storage_bytes']:,} bytes de programa** e
**{compile_audit['global_static_memory_bytes']:,} bytes globais**, CLI 1.5.1/core 4.6.0.
O frontend completo C++/Python teve **{native_audit['matching_decisions']}/{native_audit['tested_windows']}**
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
- 20/20 arquivos RAR íntegros: {manifest['download_total_bytes']:,} bytes.
- Origem dos arquivos: espelho Zenodo 4964774 com o mesmo DOI; nomes/tamanhos/MD5
  conferidos contra metadados oficiais Dryad. As rotas Dryad recusaram o download automático.
- Arquivos compactados: **data/archives/**; conteúdo completo: **data/raw/**.
- Ruídos reais TinyML: **data/noise/**, com commit e SHA-256 fixados.
- Inventário de {manifest['inventory_files']} arquivos; {manifest['analyzed_windows']:,} janelas
  analisadas, incluindo ruído. Benchmark de 20 espécies: {manifest['species_windows']:,}
  janelas em {manifest['species_groups']} grupos.
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
{improved_rows}

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
{rows}

Essas métricas usam três divisões por grupo/gravação. Não equivalem aos percentuais
publicados pelos artigos. O teste entre aparelhos mostra limites de generalização;
algumas espécies estão concentradas em poucas sessões e há rótulos temporais fracos.
O notebook documenta auditoria, cobertura, intervalos descritivos e limites de interpretação.
Não é uma validação de campo ou de desempenho no microcontrolador.

O classificador linear INT8 usa payload numérico de **{quantization['payload_bytes']} bytes**,
excluindo frontend/firmware/RAM. A exportação C foi compilada, com
**{quantization['c_python_matching_predictions']}/400 previsões iguais ao Python**.
O arquivo C corresponde à dobra 0; não é um modelo revalidado para produção.

## Reproduzir

Ambiente verificado: Python 3.14.7; versões exatas em **results/run_manifest.json**.
GPU usada: {manifest['gpu']}. GPU é opcional e os checkpoints aceitam leitura em CPU.
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

- {execution['executed_code_cells']} células de código executadas; {execution['error_outputs']} erros.
- Asserções de separação por grupos e formas de onda idênticas.
- Comparação numérica dos frontends SciPy/Torch.
- 400 previsões C/Python correspondentes.
- {native_audit['matching_decisions']} decisões do frontend/modelo Arduino C++/Python correspondentes.
- Sketch compilado para Nano 33 BLE; sem teste em placa física.
- Modelo de 20 espécies: {species_native['feature_only_test_windows']} decisões equivalentes e {species_native['complete_pcm_test_windows']} trechos PCM verificados; sketch compilado.
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
"""
(ROOT / "README.md").write_text(readme)
delivered = [ROOT / name for name in [
    "README.md", "mosquito_wingbeat_estudo.ipynb", "mosquito_wingbeat_estudo.html",
    "requirements.txt", "fetch_data.py", "notebook_code.py", "build_notebook.py",
    "execute_notebook.py", "prepare_delivery.py", ".gitignore", ".gitattributes", "literature/README.md",
    "build_report.py", "requirements-report.txt",
    "improve_device.py", "arduino_frontend.py", "arduino_models.py", "train_arduino.py",
    "verify_arduino.py", "probe_arduino_noise.py",
    "arduino_species_models.py", "train_arduino_species.py", "verify_arduino_species.py",
    "flash_arduino.sh", "install_arduino.py", "Instalar_no_Windows.cmd", "serve.py",
    "tests/test_install_arduino.py",
]]
# Pacote autônomo para IDE ou instalador; sem dataset/ambiente científico.
sketch_bundle = ROOT/"output/arduino/MosquitoSpecies.zip"
sketch_bundle.parent.mkdir(parents=True,exist_ok=True)
sketch_files = [ROOT/"firmware/MosquitoSpecies"/name for name in [
    "MosquitoSpecies.ino","StreamingFeatures.h","PresenceModel.h","SpeciesModel.h","SpeciesDecision.h","README.md"]]
sketch_files += [ROOT/name for name in ["flash_arduino.sh", "install_arduino.py", "Instalar_no_Windows.cmd"]]
sketch_files += [ROOT/"firmware"/name for name in ["LEIA_PRIMEIRO.md", "MANUAL_DE_MONTAGEM_E_INSTALACAO.md"]]
with zipfile.ZipFile(sketch_bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as handle:
    for path in sketch_files:
        handle.write(path,"MosquitoSpecies/"+path.name)
with zipfile.ZipFile(sketch_bundle) as handle:
    assert handle.testzip() is None
delivered.append(sketch_bundle)
delivered += [p for p in (ROOT / "firmware").rglob("*") if p.is_file()]
delivered += sorted((ROOT / "data").glob("*.json"))
delivered += [p for p in (ROOT / "reports").rglob("*") if p.is_file()]
delivered += sorted((ROOT / "output/pdf").glob("*.pdf"))
delivered += [p for p in RESULTS.rglob("*") if p.is_file() and p.suffix != ".joblib"
              and p.name != "delivery_manifest.json"]
delivered = sorted(set(delivered))
assert all(p.exists() for p in delivered)
digests = {str(p.relative_to(ROOT)): sha256_file(p) for p in delivered}
delivery_manifest = ROOT / "delivery_manifest.json"
delivery_manifest.write_text(json.dumps(dict(files=digests, dataset_included_in_zip=False,
                                            original_pdfs_included_in_zip=False,
                                            complete_dataset_available_locally=(ROOT / "data/raw").is_dir()), indent=2))
bundle = ROOT / "dist/mosquito-wingbeat-notebook.zip"
bundle.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
    for path in delivered + [delivery_manifest]:
        handle.write(path, "mosquito-wingbeat/" + str(path.relative_to(ROOT)))
with zipfile.ZipFile(bundle) as handle:
    assert handle.testzip() is None
print(f"Bundle: {len(delivered)+1} arquivos, {bundle.stat().st_size/1e6:.2f} MB")
print(bundle)
