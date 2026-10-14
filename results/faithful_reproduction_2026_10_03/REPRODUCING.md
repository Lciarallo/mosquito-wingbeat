# Reexecutar avaliações originais

Esta rodada avalia pesos treinados pelos autores. Não refaz treinos nem a implantação em Arduino. As alterações de compatibilidade e as diferenças de ambiente estão nos manifestos e no relatório.

## Código e dados

Os diretórios precisam conter os checkouts oficiais nas revisões abaixo. Em uma instalação nova:

```bash
git clone https://github.com/Yuanbo2020/DR-BioL tmp/reproductions_2026_10_01/dr_biol
git -C tmp/reproductions_2026_10_01/dr_biol checkout 1e4fea5146e40bdea3e9b3f0e36a54e6e63571e8
git clone https://github.com/akaraspt/mosquitosongp tmp/reproductions_2026_10_01/mosquitosongp
git -C tmp/reproductions_2026_10_01/mosquitosongp checkout 0e9187c454d7ce720cbb4925664554d1d44797c5
.venv/bin/python fetch_faithful_dr_biol.py
```

O download DR-BioL verifica o MD5 do ZIP publicado no Zenodo 17287037 antes de extrair o pickle. O diretório DR-BioL já contém os dois checkpoints oficiais; seus hashes estão em `dr_biol/execution.json`.

Para MosquitoSong+, obtenha os arquivos públicos indicados pelo README dos autores. Se os 1.324 WAVs já estiverem em `data/external/additional/mosquitosong`, mantenha essa pasta. Os arquivos podem ser baixados em uma instalação nova desta forma:

```bash
gdown --folder 'https://drive.google.com/drive/folders/1TfLHtHpVAfus6msjiDjgdeQHT9m3gnX7' -O data/external/faithful_2026_10_03/mosquitosong_models
gdown --folder 'https://drive.google.com/drive/folders/1jvRfJ4X3UKtyV8regcHRgZLuzwPQWW6N' -O data/external/faithful_2026_10_03/mosquitosong_data
mkdir -p data/external/additional
ln -s ../faithful_2026_10_03/mosquitosong_data/mosquito_recordings data/external/additional/mosquitosong
```

Os H5 vão em `mosquitosong_models/`; os 12 ruídos em `mosquitosong_data/environmental_noise_recordings/`. Os hashes de todos os H5, WAVs e ruídos usados estão em `mosquitosong/execution.json`. A ordem dos modelos foi registrada nas métricas: cada fold usou seu arquivo correspondente.

## Ambiente separado do TensorFlow

TensorFlow 2.4 não pode ser instalado no Python 3.14 usado pelo restante do projeto. Mantive o ambiente principal intacto:

```bash
uv python install 3.8
uv venv --python 3.8 tmp/faithful_reproduction_2026_10_03/env_tf24
uv pip install --python tmp/faithful_reproduction_2026_10_03/env_tf24/bin/python -r results/faithful_reproduction_2026_10_03/mosquitosong_environment.txt
```

O README original sugere Python 3.7, TensorFlow 2.4 e GPU NVIDIA. Aqui a avaliação MosquitoSong+ foi em Python 3.8.20, TensorFlow 2.4.0 e CPU. Sem lockfile dos autores, as versões originais das outras bibliotecas não podem ser confirmadas. O arquivo de versões descreve exatamente o ambiente usado nesta reexecução.

## Avaliações originais

```bash
MPLBACKEND=Agg .venv/bin/python run_author_dr_biol.py
TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 MPLBACKEND=Agg tmp/faithful_reproduction_2026_10_03/env_tf24/bin/python run_author_mosquitosong.py
tmp/faithful_reproduction_2026_10_03/env_tf24/bin/python audit_author_mosquitosong.py
```

DR-BioL usa as versões do ambiente principal registradas em `dr_biol/execution.json`. O script executa o `Evaluation.py` original; para DAT muda somente a seleção do nome do modelo. O adaptador sklearn aceita o argumento posicional antigo sem mudar a fórmula. Cinco avaliações do mesmo checkpoint com dropout não são cinco treinos. A figura ROC/PR e as confusões do código original são salvas em diretórios separados em `tmp/faithful_reproduction_2026_10_03/dr_author_work/`.

O runner MosquitoSong+ executa as células originais via Python, com a mesma ordem e sem mudar seus textos; o notebook não tem magics de IPython. O valor de expressões isoladas não é renderizado implicitamente nesse runner. As contas, CSVs e previsões são originais. O registro de `classification_report` apenas captura os argumentos e retorna o resultado da função original. O notebook executado e os seis CSVs ficam em `mosquitosong/`.

## Diagnóstico externo adicional

Essa etapa é separada do protocolo do artigo. Requer os 32 áudios já identificados no manifesto anterior e o áudio privado, que não é distribuído. `common_test_metadata.json` registra os mesmos 641 intervalos e os 107 intervalos privados, com a qualidade calculada antes da inferência. Para refazer essa metadata, em uma máquina com os mesmos arquivos:

```python
import json
from pathlib import Path
from corrected_validation import audio_windows

root = Path('.')
plan = json.loads((root/'results/cross_domain_strategies_2026_10_03/fresh_selection.json').read_text())
records = []
for item in plan['files']:
    waves, q, size = audio_windows(root/item['path'])
    records.append({**item, 'windows': len(waves), 'size_16k': size,
                    'quality_ok': q.tolist(), 'scope': 'external32_retrospective'})
waves, q, size = audio_windows(root/'tmp/private-audio-test/download.audio')
records.append(dict(path='tmp/private-audio-test/download.audio', group='private_drive',
                    reference_aedes=True, reference_species='Aedes aegypti (user declaration)',
                    windows=len(waves), size_16k=size, quality_ok=q.tolist(), scope='drive_positive_only'))
(root/'results/faithful_reproduction_2026_10_03/common_test_metadata.json').write_text(json.dumps(records, indent=2))
```

O protocolo externo fica em `external_protocol.json`: janelas nativas de 300 ms e comparação separada em 992 ms com três sub-janelas reais; dez modelos sem seleção do melhor; nenhum ajuste de ganho/limiar pelo teste. A média entre os dez modelos é um ensemble derivado adicional, distinto do estudo. Os modelos anteriores não foram reexecutados nem ajustados nesta comparação: suas previsões/métricas congeladas nos mesmos intervalos foram reaproveitadas. Pesos anteriores e o áudio privado não são necessários para reexecutar a avaliação oficial dos artigos.

```bash
TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 tmp/faithful_reproduction_2026_10_03/env_tf24/bin/python evaluate_author_mosquitosong_external.py
.venv/bin/python build_faithful_reproduction_artifacts.py
.venv/bin/python -m pytest -q tests/test_faithful_reproduction.py
```

Nenhum comando desta sequência compila, reduz ou instala um modelo no Arduino.
