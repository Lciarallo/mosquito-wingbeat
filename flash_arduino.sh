#!/usr/bin/env bash
# Funciona na raiz do repositório e dentro do ZIP MosquitoSpecies.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for TASK_PYTHON in python3 python; do
    if command -v "$TASK_PYTHON" >/dev/null 2>&1 &&
       "$TASK_PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
        exec "$TASK_PYTHON" "$TASK_SCRIPT_DIR/install_arduino.py" "$@"
    fi
done
echo "Python 3.9 ou mais recente não encontrado. Instale Python ou use a Arduino IDE. Consulte LEIA_PRIMEIRO.md / o manual de instalação." >&2
exit 1
