#!/usr/bin/env bash
# reproduce.sh — build NixNan from this repo and run this issue's reproducer
# under instrumentation. Writes fresh nnlog / stdout next to the bundled
# originals so you can compare.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISSUE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NIXNAN_ROOT="$(cd "${ISSUE_DIR}/../.." && pwd)"
DATA_DIR="${ISSUE_DIR}/data"

echo "==> issue dir:        ${ISSUE_DIR}"
echo "==> NixNan repo root: ${NIXNAN_ROOT}"
echo

# 1. Build NixNan if not already built
NIXNAN_SO="${NIXNAN_ROOT}/nvbit_release/tools/nixnan/nixnan.so"
if [[ ! -f "${NIXNAN_SO}" ]]; then
  echo "==> NixNan not built yet — running 'make' (one-time, ~2 minutes)..."
  ( cd "${NIXNAN_ROOT}" && make ) || {
    echo "ERROR: NixNan build failed. See ${NIXNAN_ROOT}/Tutorial.md for"
    echo "       prerequisites (CUDA toolkit, GCC, Make)."
    exit 1
  }
fi

if [[ ! -f "${NIXNAN_SO}" ]]; then
  echo "ERROR: ${NIXNAN_SO} still missing after build."
  exit 1
fi
echo "==> NixNan .so: ${NIXNAN_SO}"
echo

# 2. Verify torch + CUDA available
echo "==> Verifying PyTorch + CUDA..."
python3 -c "import torch; print('torch', torch.__version__, ' cuda_avail=', torch.cuda.is_available()); assert torch.cuda.is_available()" || {
  echo
  echo "ERROR: PyTorch CUDA is not available."
  echo "       If 'nvidia-smi' itself fails with 'Driver/library version mismatch',"
  echo "       see ${NIXNAN_ROOT}/Tutorial.md for a user-local libcuda shim."
  exit 1
}
echo

# 3. Run the bundled repro under NixNan
cd "${DATA_DIR}"
echo "==> Running ${DATA_DIR}/repro.py under NixNan instrumentation..."
echo

LD_PRELOAD="${NIXNAN_SO}" \
  TOOL_VERBOSE=0 \
  LINE_INFO=1 \
  PRINT_ILL_INSTR=1 \
  INSTR_MEM=1 \
  HISTOGRAM=1 \
  ENABLE_FUN_DETAIL=1 \
  SAMPLING=1 \
  BIN_SPEC_FILE="${DATA_DIR}/bin_spec.json" \
  LOGFILE="${DATA_DIR}/nixnan.nnlog.fresh" \
  python3 repro.py > "${DATA_DIR}/stdout.nnlog.fresh" 2>&1
rc=$?

echo "==> Repro exited with code ${rc}"
echo "==> Fresh logs written:"
echo "    ${DATA_DIR}/nixnan.nnlog.fresh    (NixNan trace)"
echo "    ${DATA_DIR}/stdout.nnlog.fresh    (Python output)"
echo "==> Compare with the bundled originals:"
echo "    ${DATA_DIR}/nixnan.nnlog"
echo "    ${DATA_DIR}/stdout.nnlog"
echo
echo "==> Our analysis of this issue:"
echo "    ${DATA_DIR}/issueFeedback.md"
