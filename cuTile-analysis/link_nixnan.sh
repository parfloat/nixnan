#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIXNAN_TOOL_DIR="${NIXNAN_TOOL_DIR:-../nvbit_release/tools/nixnan}"

if [[ ! -d "${ROOT_DIR}/${NIXNAN_TOOL_DIR}" ]]; then
  echo "ERROR: NIXNAN_TOOL_DIR does not exist: ${ROOT_DIR}/${NIXNAN_TOOL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/${NIXNAN_TOOL_DIR}/nixnan.so" ]]; then
  echo "ERROR: nixnan.so not found under: ${ROOT_DIR}/${NIXNAN_TOOL_DIR}" >&2
  exit 1
fi

ln -sfn "${NIXNAN_TOOL_DIR}" "${ROOT_DIR}/borrowed-nixnan"
ln -sfn "borrowed-nixnan/nixnan.so" "${ROOT_DIR}/nixnan.so"

echo "borrowed-nixnan -> ${NIXNAN_TOOL_DIR}"
echo "nixnan.so -> borrowed-nixnan/nixnan.so"
