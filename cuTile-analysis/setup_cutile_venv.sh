#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${CUTILE_VENV:-${ROOT_DIR}/.venv-cutile}"

python -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements-cutile.txt"

echo "Created cuTile venv at ${VENV_DIR}"
echo "Run: source ${ROOT_DIR}/cutile_env.sh"
