#!/usr/bin/env bash

_cutile_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUTILE_VENV="${CUTILE_VENV:-${_cutile_root}/.venv-cutile}"
export CUTILE_SHIM_DIR="${CUTILE_SHIM_DIR:-${_cutile_root}/.cutile-libs}"
export NV580_LIB_DIR="${NV580_LIB_DIR:-/home/ganesh/opt/nv580.126/usr/lib/x86_64-linux-gnu}"
export CUDA_TOOLKIT_LIB_DIR="${CUDA_TOOLKIT_LIB_DIR:-/usr/local/cuda-13.2/targets/x86_64-linux/lib}"

mkdir -p "${CUTILE_SHIM_DIR}"

ln -sf "${NV580_LIB_DIR}/libcuda.so" "${CUTILE_SHIM_DIR}/libcuda.so"
ln -sf "${NV580_LIB_DIR}/libcuda.so.1" "${CUTILE_SHIM_DIR}/libcuda.so.1"
ln -sf "${NV580_LIB_DIR}/libnvidia-ml.so.1" "${CUTILE_SHIM_DIR}/libnvidia-ml.so.1"
ln -sf "${NV580_LIB_DIR}/libnvidia-ptxjitcompiler.so.1" "${CUTILE_SHIM_DIR}/libnvidia-ptxjitcompiler.so.1"
ln -sf "${NV580_LIB_DIR}/libnvidia-ptxjitcompiler.so.580.126.09" "${CUTILE_SHIM_DIR}/libnvidia-ptxjitcompiler.so"

if [[ -z "${NVIDIA_GPUCOMP_LIB:-}" ]]; then
  if [[ -f "${NV580_LIB_DIR}/libnvidia-gpucomp.so.580.126.09" ]]; then
    NVIDIA_GPUCOMP_LIB="${NV580_LIB_DIR}/libnvidia-gpucomp.so.580.126.09"
  else
    NVIDIA_GPUCOMP_LIB="/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.580.159.03"
  fi
fi

if [[ -f "${NVIDIA_GPUCOMP_LIB}" ]]; then
  ln -sf "${NVIDIA_GPUCOMP_LIB}" "${CUTILE_SHIM_DIR}/libnvidia-gpucomp.so.580.126.09"
fi

export LD_LIBRARY_PATH="${CUTILE_SHIM_DIR}:${NV580_LIB_DIR}:${CUDA_TOOLKIT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export PATH="${CUTILE_VENV}/bin:${PATH}"
export PYTHON_BIN="${PYTHON_BIN:-${CUTILE_VENV}/bin/python}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "cuTile runtime environment prepared."
  echo "To persist it in your shell, run:"
  echo "  source ${_cutile_root}/cutile_env.sh"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
fi
