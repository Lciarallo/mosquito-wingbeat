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
quantization = json.loads((RESULTS / "int8_audit.json").read_text())
notebook = nbformat.read(ROOT / "mosquito_wingbeat_estudo.ipynb", as_version=4)
nbformat.validate(notebook)
assert execution["error_outputs"] == 0
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
assert not any(o.output_type == "error" for c in notebook.cells for o in c.get("outputs", []))
assert manifest["verified_archives"] == 20
assert quantization["c_python_matching_predictions"] == 400
assert len(metrics) == 8

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
readme = f"""# Estudo de classificação acústica de mosquitos

Abra [mosquito_wingbeat_estudo.ipynb](mosquito_wingbeat_estudo.ipynb) para executar ou baixe
[mosquito_wingbeat_estudo.html](mosquito_wingbeat_estudo.html) e abra no navegador
para ler o estudo e os gráficos sem Jupyter.

O notebook reúne a leitura de três artigos, referências complementares, download, auditoria,
features de frequência/PSD/MFCC, oito classificadores para 20 espécies, tarefa TinyML
de quatro classes, aparelhos reservados, ruído, abstenção, INT8 e exportação C.
O código do downloader também está incorporado ao notebook, que não depende dos scripts
auxiliares para definir os experimentos.

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

## Evidências de execução

- {execution['executed_code_cells']} células de código executadas; {execution['error_outputs']} erros.
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
"""
(ROOT / "README.md").write_text(readme)
delivered = [ROOT / name for name in [
    "README.md", "mosquito_wingbeat_estudo.ipynb", "mosquito_wingbeat_estudo.html",
    "requirements.txt", "fetch_data.py", "notebook_code.py", "build_notebook.py",
    "execute_notebook.py", "prepare_delivery.py", ".gitignore", "literature/README.md",
]]
delivered += sorted((ROOT / "data").glob("*.json"))
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
