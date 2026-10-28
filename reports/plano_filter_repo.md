# Plano de limpeza do histórico com `git filter-repo`

Proposto em 22/10/2026. **Não executado.** Depende de confirmação explícita do autor, porque reescreve o histórico e exige force-push.

## Diagnóstico

| Medida | Valor |
|---|---|
| `.git` antes de `git gc` | 620 MB, todos objetos soltos |
| `.git` depois de `git gc --prune=never` | 562 MB (`size-pack` 561 MB) |
| Blobs no histórico | 403 MB |
| `results/` no histórico | 267 MB |
| `output/` no histórico | 118 MB |
| Blobs restantes sem `results/`, `output/` e `drive_audio.wav` | cerca de 15 MB |

Nenhuma ponta de branch versiona `results/`, `output/` ou `drive_audio.wav`. Esses caminhos existem só em commits antigos. Exemplos: `output/notebook_transfer_delivery_2026_10_14/notebook_transfer_completo_2026_10_14.zip` (30 MB) e várias versões de `results/ternary_nested_2026_10_14/predictions_oof.csv` e de `results/ternary_detection_2026_10_14/predictions_oof.csv`.

**Dado privado no histórico público.** `drive_audio.wav` é o áudio privado do Drive, excluído do estudo em `reports/exclusao_audio_drive_2026_10_15.md`. Ele entrou em `de59a7e` (06/10/2026) e saiu em `15f8a21` (15/10/2026), mas continua no histórico de `origin/main`, e o repositório é público. `results/*/private_audio_summary.json` e `results/corrected_2026_10_01/private_audio_aggregate.csv` também estão no histórico.

## Impacto

| Alvo | Efeito |
|---|---|
| `main` (28 commits), `detector-v3` (33), `reanalise-v2` (31, só local) | Todos os SHAs mudam a partir do primeiro commit, `8e04b88` (29/09/2026). As árvores das pontas ficam idênticas. |
| Remoto `origin` (público, 0 forks) | Exige `git push --force-with-lease` de `main` e `detector-v3`. Outros clones precisam ser refeitos. |
| PR aberto `detector-v3` → `main` | O GitHub mantém `refs/pull/N/head` apontando para o histórico antigo. Depois do force-push, o PR passa a mostrar os novos commits, mas a referência antiga só sai com pedido ao GitHub Support. |
| Verificação por hash | Não muda. Protocolos e manifestos (`reanalysis/protocol_*.json`, `*_freeze.json`, `data/metadata/*_reference.json`) registram o SHA-256 do **conteúdo** dos arquivos. Os arquivos citados em `results/` ficam no diretório de trabalho local, que o `filter-repo` não toca. Nenhum arquivo versionado cita SHA completo de commit. |
| Referências curtas a commits em relatórios | `reports/analise_de_lacunas_continuacao_v3.md` cita `be13d6a`, e `reports/reanalise_v2.md` cita `3d5a1b8`. Os relatórios não serão editados. O mapa antigo → novo gerado pelo `filter-repo` vai para `reports/mapa_commits_filter_repo.csv`. |
| Stash `codex-preserve-local-before-git-update-2026-10-08` | É convertido em branch antes da reescrita, para não se perder. |

## Comandos (executar só após confirmação)

```bash
set -euo pipefail
REPO=~/Documentos/Git_projects/mosquito-wingbeat
BK=~/backups/mosquito-wingbeat-pre-filter-$(date +%Y%m%d).git
WORK=/tmp/mosquito-wingbeat-filter

# 0. Pré-condições: árvore limpa e branches sincronizados com origin
cd "$REPO"
git status --porcelain | grep -q . && { echo "árvore suja"; exit 1; }
git fetch origin

# 1. Preservar o stash como branch e fazer backup completo (todas as refs)
git branch backup/stash-2026-10-08 'stash@{0}'
git clone --mirror "$REPO" "$BK"
git -C "$BK" count-objects -vH

# 2. Instalar a ferramenta
python3 -m pip install --user git-filter-repo

# 3. Reescrever num clone novo, nunca no diretório de trabalho
git clone --no-local "$REPO" "$WORK"
cd "$WORK"
for b in main reanalise-v2 backup/stash-2026-10-08; do git branch --track "$b" "origin/$b" 2>/dev/null || true; done
git filter-repo --invert-paths \
  --path results/ \
  --path output/ \
  --path drive_audio.wav
cp .git/filter-repo/commit-map "$REPO/reports/mapa_commits_filter_repo.csv"

# 4. Verificar: árvores das pontas idênticas, testes e tamanho
for b in main detector-v3 reanalise-v2; do
  test "$(git -C "$REPO" rev-parse "$b^{tree}")" = "$(git rev-parse "$b^{tree}")" && echo "$b: árvore idêntica"
done
git count-objects -vH
git log --all --oneline -- results output drive_audio.wav | wc -l   # deve ser 0

# 5. Publicar (force-push com lease)
git remote add origin https://github.com/Lciarallo/mosquito-wingbeat.git
git fetch origin
git push --force-with-lease=main:origin/main --force-with-lease=detector-v3:origin/detector-v3 origin main detector-v3

# 6. Alinhar o diretório de trabalho (arquivos ignorados em results/ e data/ não são tocados)
cd "$REPO"
git fetch "$WORK" main:main-rewritten detector-v3:detector-v3-rewritten
git checkout detector-v3 && git reset --hard detector-v3-rewritten
git branch -f main main-rewritten
git branch -D main-rewritten detector-v3-rewritten
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

Depois disso, abrir um pedido ao GitHub Support para remover as visualizações em cache e as referências de PR ao histórico antigo, citando `drive_audio.wav` e o commit `de59a7e`.

## Reversão

Até o passo 5, nada sai da máquina, e o backup `--mirror` restaura todas as refs. Depois do force-push, a reversão é `git push --mirror` a partir do backup, que reintroduz o histórico antigo, inclusive o áudio privado.
