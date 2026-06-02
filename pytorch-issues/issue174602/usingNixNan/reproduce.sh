#!/usr/bin/env bash
# reproduce.sh - find or install NixNan, then run this issue's reproducer
# under instrumentation. Writes fresh nnlog / stdout next to the bundled
# originals so you can compare.
set -uo pipefail

NIXNAN_REPO_URL="https://github.com/parfloat/nixnan.git"
NIXNAN_TUTORIAL_URL="https://github.com/parfloat/nixnan/blob/main/Tutorial.md"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISSUE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_NIXNAN_ROOT="$(cd "${ISSUE_DIR}/../.." && pwd)"
DATA_DIR="${ISSUE_DIR}/data"

if [[ -f "${PARENT_NIXNAN_ROOT}/Makefile" && -d "${PARENT_NIXNAN_ROOT}/nvbit_release/tools/nixnan" ]]; then
  DEFAULT_NIXNAN_INSTALL_DIR="${PARENT_NIXNAN_ROOT}"
else
  DEFAULT_NIXNAN_INSTALL_DIR="${HOME}/nixnan"
fi
NIXNAN_INSTALL_DIR="${NIXNAN_INSTALL_DIR:-${DEFAULT_NIXNAN_INSTALL_DIR}}"

abs_file() {
  local path="$1"

  (cd "$(dirname "${path}")" && printf '%s/%s\n' "$(pwd)" "$(basename "${path}")")
}

nixnan_so_for_root() {
  local root="$1"

  [[ -z "${root}" ]] && return 1
  if [[ -f "${root}/nvbit_release/tools/nixnan/nixnan.so" ]]; then
    abs_file "${root}/nvbit_release/tools/nixnan/nixnan.so"
    return 0
  fi
  if [[ -f "${root}/nixnan.so" ]]; then
    abs_file "${root}/nixnan.so"
    return 0
  fi
  return 1
}

find_nixnan_so() {
  local root so

  if [[ -n "${NIXNAN_SO:-}" ]]; then
    if [[ -f "${NIXNAN_SO}" ]]; then
      abs_file "${NIXNAN_SO}"
      return 0
    fi
    echo "ERROR: NIXNAN_SO is set but does not point to a file: ${NIXNAN_SO}" >&2
    return 2
  fi

  for root in \
    "${NIXNAN_ROOT:-}" \
    "${NIXNAN_HOME:-}" \
    "${PARENT_NIXNAN_ROOT}" \
    "${NIXNAN_INSTALL_DIR}" \
    "${HOME}/nixnan" \
    "${HOME}/repos/parfloat-work/nixnan"; do
    so="$(nixnan_so_for_root "${root}")" || true
    if [[ -n "${so}" ]]; then
      printf '%s\n' "${so}"
      return 0
    fi
  done

  return 1
}

root_from_nixnan_so() {
  local so="$1"
  case "${so}" in
    */nvbit_release/tools/nixnan/nixnan.so)
      (cd "$(dirname "${so}")/../../.." && pwd)
      ;;
    *)
      (cd "$(dirname "${so}")" && pwd)
      ;;
  esac
}

build_nixnan() {
  local root="$1"

  echo "==> Building NixNan in ${root}..."
  (cd "${root}" && make) || {
    echo
    echo "ERROR: NixNan build failed. See the upstream tutorial for prerequisites:"
    echo "       ${NIXNAN_TUTORIAL_URL}"
    exit 1
  }
}

install_nixnan() {
  local answer

  echo "==> NixNan was not found."
  echo "    Looked for nixnan.so via NIXNAN_SO, NIXNAN_ROOT, NIXNAN_HOME,"
  echo "    the parent checkout, and ${NIXNAN_INSTALL_DIR}."
  echo
  echo "    Tutorial: ${NIXNAN_TUTORIAL_URL}"
  echo "    This script can clone/build NixNan from ${NIXNAN_REPO_URL}."
  echo "    Install/build directory: ${NIXNAN_INSTALL_DIR}"
  echo

  if [[ ! -t 0 ]]; then
    echo "ERROR: Cannot prompt for installation because stdin is not a terminal."
    echo "       Install NixNan manually, or rerun with NIXNAN_SO=/path/to/nixnan.so."
    exit 1
  fi

  read -r -p "Install and build NixNan now? [y/N] " answer
  case "${answer}" in
    y|Y|yes|YES)
      ;;
    *)
      echo "ERROR: NixNan is required to run this reproducer."
      echo "       Install it manually or set NIXNAN_SO=/path/to/nixnan.so."
      exit 1
      ;;
  esac

  if [[ -d "${NIXNAN_INSTALL_DIR}/.git" || -f "${NIXNAN_INSTALL_DIR}/Makefile" ]]; then
    echo "==> Using existing NixNan checkout at ${NIXNAN_INSTALL_DIR}."
  elif [[ -e "${NIXNAN_INSTALL_DIR}" ]]; then
    echo "ERROR: ${NIXNAN_INSTALL_DIR} exists but does not look like a NixNan checkout."
    echo "       Set NIXNAN_INSTALL_DIR to another location, or set NIXNAN_SO directly."
    exit 1
  else
    command -v git >/dev/null 2>&1 || {
      echo "ERROR: git is required to clone ${NIXNAN_REPO_URL}."
      exit 1
    }
    git clone "${NIXNAN_REPO_URL}" "${NIXNAN_INSTALL_DIR}" || exit 1
  fi

  build_nixnan "${NIXNAN_INSTALL_DIR}"
}

NIXNAN_SO="$(find_nixnan_so)"
find_rc=$?
if [[ ${find_rc} -eq 2 ]]; then
  exit 1
fi

if [[ -z "${NIXNAN_SO}" ]]; then
  install_nixnan
  NIXNAN_SO="$(find_nixnan_so)"
fi

if [[ -z "${NIXNAN_SO}" || ! -f "${NIXNAN_SO}" ]]; then
  echo "ERROR: nixnan.so is still missing after setup."
  echo "       Set NIXNAN_SO=/path/to/nixnan.so and rerun this script."
  exit 1
fi

NIXNAN_ROOT="$(root_from_nixnan_so "${NIXNAN_SO}")"

echo "==> issue dir:        ${ISSUE_DIR}"
echo "==> NixNan root:      ${NIXNAN_ROOT}"
echo "==> NixNan .so:       ${NIXNAN_SO}"
echo

# 1. Verify torch + CUDA available
echo "==> Verifying PyTorch + CUDA..."
python3 -c "import torch; print('torch', torch.__version__, ' cuda_avail=', torch.cuda.is_available()); assert torch.cuda.is_available()" || {
  echo
  echo "ERROR: PyTorch CUDA is not available."
  echo "       If 'nvidia-smi' itself fails with 'Driver/library version mismatch',"
  echo "       see the NixNan tutorial for a user-local libcuda shim:"
  echo "       ${NIXNAN_TUTORIAL_URL}"
  exit 1
}
echo

# 2. Run the bundled repro under NixNan
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
