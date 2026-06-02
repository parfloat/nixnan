#!/usr/bin/env bash
# reproduce.sh - find or install NixNan, verify runtime prerequisites, then run
# this issue's reproducer under instrumentation. Writes fresh nnlog / stdout
# next to the bundled originals so you can compare.
set -uo pipefail

NIXNAN_REPO_URL="https://github.com/parfloat/nixnan.git"
NIXNAN_TUTORIAL_URL="https://github.com/parfloat/nixnan/blob/main/Tutorial.md"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

prepend_ld_library_path() {
  local dir="$1"

  [[ -d "${dir}" ]] || return 1
  case ":${LD_LIBRARY_PATH:-}:" in
    *:"${dir}":*) ;;
    *) export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

candidate_nvidia_lib_dirs() {
  local dir seen=":"

  for dir in \
    "${NIXNAN_NVIDIA_LIBDIR:-}" \
    "${NVIDIA_LIBDIR:-}" \
    "${CUDA_DRIVER_LIBDIR:-}" \
    "${HOME}"/opt/nv*/usr/lib/x86_64-linux-gnu \
    "${HOME}"/opt/nvidia*/usr/lib/x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu \
    /lib/x86_64-linux-gnu; do
    [[ -d "${dir}" ]] || continue
    [[ -f "${dir}/libcuda.so" || -f "${dir}/libcuda.so.1" ]] || continue
    case "${seen}" in
      *:"${dir}":*) continue ;;
    esac
    seen="${seen}${dir}:"
    printf '%s\n' "${dir}"
  done
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

require_build_prereqs() {
  local root="$1"
  local missing=0

  for cmd in make; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "ERROR: '${cmd}' is required to build NixNan but was not found in PATH."
      missing=1
    fi
  done

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc is required to build NixNan but was not found in PATH."
    echo "       Install the CUDA toolkit or add its bin directory to PATH."
    missing=1
  fi

  if [[ ! -f "${root}/nvbit-Linux-x86_64-1.8.tar.bz2" && ! -d "${root}/nvbit_release_x86_64" ]]; then
    for cmd in wget tar; do
      if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: '${cmd}' is required to fetch/unpack NVBit for the first build."
        missing=1
      fi
    done
  fi

  if [[ ${missing} -ne 0 ]]; then
    echo "       See ${NIXNAN_TUTORIAL_URL} for setup notes."
    exit 1
  fi
}

build_nixnan() {
  local root="$1"

  require_build_prereqs "${root}"
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

torch_cuda_probe() {
  "${PYTHON_BIN}" - <<'PY_TORCH_PROBE'
import sys
try:
    import torch
except Exception as exc:
    print(f"ERROR: failed to import torch: {exc}")
    sys.exit(10)

print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}  cuda_avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(11)
print(f"device_count={torch.cuda.device_count()}")
try:
    print(f"device0={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"device0=<unavailable: {exc}>")
PY_TORCH_PROBE
}

ensure_torch_cuda() {
  local probe_out nvsmi_out dir old_ld

  echo "==> Verifying Python + PyTorch + CUDA..."
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}"
    echo "       Set PYTHON_BIN=/path/to/python and rerun this script."
    exit 1
  fi
  echo "==> Python: $(command -v "${PYTHON_BIN}")"

  if probe_out="$(torch_cuda_probe 2>&1)"; then
    printf '%s\n' "${probe_out}"
    echo
    return 0
  fi

  echo "==> PyTorch CUDA was not available with the current environment."
  printf '%s\n' "${probe_out}"
  echo

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvsmi_out="$(nvidia-smi 2>&1 || true)"
    if [[ -n "${nvsmi_out}" ]]; then
      echo "==> nvidia-smi status:"
      printf '%s\n' "${nvsmi_out}" | head -n 12
      echo
    fi
  else
    echo "==> nvidia-smi was not found; continuing with PyTorch CUDA probes."
    echo
  fi

  echo "==> Searching for a usable NVIDIA driver library directory..."
  while IFS= read -r dir; do
    old_ld="${LD_LIBRARY_PATH:-}"
    prepend_ld_library_path "${dir}" || continue
    echo "    trying ${dir}"
    if probe_out="$(torch_cuda_probe 2>&1)"; then
      echo "==> Using NVIDIA library directory: ${dir}"
      printf '%s\n' "${probe_out}"
      echo
      return 0
    fi
    export LD_LIBRARY_PATH="${old_ld}"
  done < <(candidate_nvidia_lib_dirs)

  echo "ERROR: PyTorch CUDA is not available."
  echo "       If nvidia-smi reports a driver/library mismatch, set one of:"
  echo "         NIXNAN_NVIDIA_LIBDIR=/path/to/dir/with/libcuda.so.1"
  echo "         NVIDIA_LIBDIR=/path/to/dir/with/libcuda.so.1"
  echo "       Then rerun this script. See also: ${NIXNAN_TUTORIAL_URL}"
  exit 1
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

ensure_torch_cuda

cd "${DATA_DIR}"
echo "==> Running ${DATA_DIR}/repro.py under NixNan instrumentation..."
echo

FRESH_NIXNAN_LOG="${LOGFILE:-${DATA_DIR}/nixnan.nnlog.fresh}"
FRESH_STDOUT_LOG="${STDOUT_LOGFILE:-${DATA_DIR}/stdout.nnlog.fresh}"
RUN_LD_PRELOAD="${NIXNAN_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"

LD_PRELOAD="${RUN_LD_PRELOAD}" \
  TOOL_VERBOSE="${TOOL_VERBOSE:-0}" \
  LINE_INFO="${LINE_INFO:-1}" \
  PRINT_ILL_INSTR="${PRINT_ILL_INSTR:-1}" \
  INSTR_MEM="${INSTR_MEM:-1}" \
  HISTOGRAM="${HISTOGRAM:-1}" \
  ENABLE_FUN_DETAIL="${ENABLE_FUN_DETAIL:-1}" \
  SAMPLING="${SAMPLING:-1}" \
  BIN_SPEC_FILE="${BIN_SPEC_FILE:-${DATA_DIR}/bin_spec.json}" \
  LOGFILE="${FRESH_NIXNAN_LOG}" \
  "${PYTHON_BIN}" repro.py > "${FRESH_STDOUT_LOG}" 2>&1
rc=$?

echo "==> Repro exited with code ${rc}"
echo "==> Fresh logs written:"
echo "    ${FRESH_NIXNAN_LOG}    (NixNan trace)"
echo "    ${FRESH_STDOUT_LOG}    (Python output)"
echo "==> Compare with the bundled originals:"
echo "    ${DATA_DIR}/nixnan.nnlog"
echo "    ${DATA_DIR}/stdout.nnlog"
echo
echo "==> Our analysis of this issue:"
echo "    ${DATA_DIR}/issueFeedback.md"
