# Limpeza conservadora da pasta mosquito-wingbeat no Drive

Data UTC: 2026-10-14T18:19:31.937553+00:00. Ação autorizada pelo usuário: retirar apenas arquivos realmente inúteis.

Foram enviados à lixeira quatro diretórios da pasta mosquito-wingbeat, preservando a recuperação pelo Google Drive. Não houve exclusão permanente nem esvaziamento da lixeira.

| Diretório | Arquivos | Bytes | Motivo |
|---|---:|---:|---|
| __pycache__/ | 115 | 2149084 | Cache compilado regenerável a partir dos testes/código fonte. |
| tests/__pycache__/ | 30 | 390103 | Cache compilado regenerável a partir dos testes/código fonte. |
| .pytest_cache/ | 5 | 21014 | Cache operacional do pytest, sem manifesto ou resultado científico. |
| .venv/ | 252 | 11654853 | Cópia incompleta de ambiente virtual, sem interpretador Python, sem utilidade como instalação executável; requisitos e instruções permanecem. |

Total: 402 arquivos, 14215054 bytes. Arquivos na lixeira ainda podem consumir quota até seu esvaziamento; não afirmamos espaço liberado.

Foram preservados `.git`, `data`, `results`, `reports`, `scripts`, `firmware`, `.arduino-tools`, instaladores, literatura, documentos acadêmicos, apresentações novas e antigas, código de todas as rodadas e requisitos de instalação. A pasta `tmp` foi preservada porque contém versões anteriores de scripts, áudios e logs úteis à rastreabilidade. Logs, nomes “backup” ou resultados fracos não foram tratados automaticamente como inúteis.

A revisão foi dirigida aos diretórios comprovadamente descartáveis. Não é uma classificação integral de todos os arquivos do Google Drive do usuário. Antes de qualquer remoção, as listagens dos caches e da instalação virtual foram inspecionadas; toda a árvore de `.venv` foi inventariada. O descritor `pyvenv.cfg` recebido foi guardado localmente para rastreabilidade. A instalação local `.venv` não foi removida.

As quatro pastas foram verificadas pela API com `trashed=true`. A listagem da raiz passou de 210 para 207 itens, exatamente as três pastas esperadas; `tests` passou de 21 para 20, apenas seu cache. `verify_freeze` da v4 e `verify_history` passaram, mantendo os 1.239 arquivos históricos locais intocados. Os logs e manifestos completos desta operação estão em `output/drive_cleanup_2026_10_14_1815/`.

Nas próximas sincronizações, excluir `.venv/**`, `__pycache__/**`, `**/__pycache__/**`, `.pytest_cache/**` e arquivos `*.pyc`; os requisitos e fontes devem continuar incluídos. Este registro novo não modifica os manifestos, relatórios ou documentação histórica protegidos.
