# Publicação integral do estado de pesquisa - 14/10/2026

Este snapshot reúne as alterações locais de código, documentação, notebooks, apresentações, relatórios e resultados. As fontes recebidas da v5 são experimentais; esta publicação não cria uma rodada v5 treinada nem declara o modelo liberado.

## Artefatos grandes e restauração

Os pesos, bundles, arrays e tabelas grandes ficam nos anexos da release de pesquisa `research-snapshot-2026-10-14` do repositório `Lciarallo/mosquito-wingbeat`. A release é marcada como prévia, sem substituir o instalador Windows mais recente.

O manifesto em `reports/github_publication_2026_10_14/artifact_manifest.json` relaciona 303 arquivos em 8 pacotes TAR, com tamanho e SHA-256 de cada arquivo e pacote. Os pacotes preservam os caminhos relativos ao repositório e os bytes originais, inclusive de resultados fracos. Foram incluídos modelos anteriormente ignorados pelo Git. Não reescrevemos manifestos, limiares, pesos ou relatórios históricos para publicá-los.

Depois de clonar o projeto e instalar suas dependências, restaure os anexos com GitHub CLI autenticado:

```bash
python3 scripts/restore_publication_artifacts.py
python3 scripts/restore_publication_artifacts.py --verify
```

O script confere os hashes dos pacotes e arquivos, recusa caminhos fora do repositório e não sobrescreve arquivos existentes divergentes. Armazena exclusões dos binários em `.git/info/exclude`, preservando o `.gitignore` protegido por snapshots científicos. Há testes para caminhos inseguros e preservação de arquivos divergentes. O download usa arquivos temporários e requer espaço para um pacote mais seus arquivos restaurados.

## Código externo

`external/cdmsc_code` é um submódulo do projeto original `aptemvs/Cross-Domain-Mosquito-Species-Classification`, fixado no commit `0aa089136ff1cc029d72ef21d9ada5ace9bb9b1b`. Use `git submodule update --init --recursive` para obter as fontes. O auxiliar local de download está em `external/local_support/cdmsc_download.py`. Corpora e embeddings baixáveis mantêm suas origens e não são duplicados no histórico Git.

## Validação e limites

A suíte anterior desta publicação teve 215 aprovados e uma falha conhecida em `test_cepstrum_is_scale_normalized_and_retains_shape` da v5. O registro final, incluindo os novos testes do restaurador, está em `reports/github_publication_2026_10_14/pytest.txt`. A preservação é conferida por `verify_freeze` e `verify_history`; resultado em `reports/github_publication_2026_10_14/validation.json`. A sintaxe dos módulos Python também foi conferida. Não houve novo treinamento, compilação de firmware ou alteração do instalador.

As métricas publicadas são as observadas, sem apagar falhas nem transformar persistência temporal em confirmação entomológica. As rodadas v1-v4 e a auditoria dos scripts recebidos do Drive mantêm suas limitações. A identificação automática de espécies continua experimental.

## Fora do repositório público

O áudio privado `data/raw/drive_test/drive_audio.wav`, corpora brutos, ambientes Python, caches, credenciais e renders temporários não foram publicados. Os resultados derivados e metadados de avaliação permanecem disponíveis. Os arquivos privados e os pesos originais continuam intactos localmente e nas entregas já verificadas no Drive. A entrega pública de modelos não contém áudios privados.
