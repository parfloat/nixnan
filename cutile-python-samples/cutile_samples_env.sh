#!/usr/bin/env bash

_sample_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_nixnan_root="$(cd "${_sample_root}/.." && pwd)"

source "${_nixnan_root}/cuTile-analysis/cutile_env.sh"

export CUTILE_TORCH_SITE_PACKAGES="${CUTILE_TORCH_SITE_PACKAGES:-/home/ganesh/repos/BackendBenchExamples/.venv/lib/python3.10/site-packages}"

if [[ ! -f "${CUTILE_TORCH_SITE_PACKAGES}/torch/__init__.py" ]]; then
  echo "Missing torch package at CUTILE_TORCH_SITE_PACKAGES=${CUTILE_TORCH_SITE_PACKAGES}" >&2
  return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="${CUTILE_VENV}/lib/python3.10/site-packages:${CUTILE_TORCH_SITE_PACKAGES}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "cuTile sample environment prepared."
  echo "To persist it in your shell, run:"
  echo "  source ${_sample_root}/cutile_samples_env.sh"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "CUTILE_TORCH_SITE_PACKAGES=${CUTILE_TORCH_SITE_PACKAGES}"
fi
