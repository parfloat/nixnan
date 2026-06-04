#!/usr/bin/env bash
# Sweep the cuTile Python samples under NixNan to look for FP exceptions
# (NaN, Inf, divide-by-zero, denormals).
#
# Layout per sample (mirrors pytorch-issues/issueXYZ/data/):
#   nixnan_samples/<sample_id>/data/{stdout.nnlog, nixnan.nnlog, run.env}
# Shared:
#   nixnan_samples/bin_spec.json    - exponent-bin spec (COUNT bucket size)
#   nixnan_samples/summary.tsv      - one row per sample
#
# Heavy samples use SAMPLING (Nth-kernel-launch instrumentation). FFT runs
# unsampled because the known div-by-zero is the motivation for this sweep.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIXNAN_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/nixnan_samples}"
BIN_SPEC_FILE="${BIN_SPEC_FILE:-${DATA_ROOT}/bin_spec.json}"
SUMMARY_FILE="${SUMMARY_FILE:-${DATA_ROOT}/summary.tsv}"
NIXNAN_SO="${NIXNAN_SO:-${NIXNAN_ROOT}/cuTile-analysis/nixnan.so}"
COUNT="${COUNT:-128}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-0}"

# shellcheck source=/dev/null
source "${ROOT_DIR}/cutile_samples_env.sh"

mkdir -p "${DATA_ROOT}"

if [[ ! -f "${BIN_SPEC_FILE}" ]]; then
  cat > "${BIN_SPEC_FILE}" <<JSON
{
  "_comment": "Per-binade exponent histogram for cuTile Python sample runs. count=${COUNT}. NaN and +/-Inf are reported by NixNan exception summaries, not normal exponent bins.",
  "count": ${COUNT},
  "bf16": [[-126, 127]],
  "f16": [[-14, 15]],
  "f32": [[-126, 127]],
  "f64": [[-1022, 1023]]
}
JSON
fi

if [[ ! -f "${NIXNAN_SO}" ]]; then
  echo "ERROR: NixNan preload library is missing: ${NIXNAN_SO}" >&2
  exit 1
fi

sample_id_for() {
  local sample="$1"
  case "${sample}" in
    samples/quickstart/VectorAdd_quickstart.py) echo "VectorAdd_quickstart" ;;
    samples/quickstart/VA_ovflo_nixnan.py)      echo "VA_ovflo_nixnan" ;;
    samples/*.py)
      local base="${sample##*/}"
      echo "${base%.py}"
      ;;
    *)
      printf '%s' "${sample}" | tr '/.:-' '____'
      ;;
  esac
}

# Per-sample SAMPLING factor. 0 = no sampling (instrument every kernel launch).
# N>0 = instrument every Nth repeat of a recurring kernel.
sampling_for() {
  case "$1" in
    samples/AttentionFMHA.py)                 echo 500 ;;
    samples/AllGatherMatmul.py)               echo 10  ;;
    samples/BatchMatMul.py)                   echo 10  ;;
    samples/MatMul.py)                        echo 10  ;;
    samples/MoE.py)                           echo 10  ;;
    samples/LayerNorm.py)                     echo 5   ;;
    samples/Transpose.py)                     echo 5   ;;
    samples/VectorAddition.py)                echo 5   ;;
    samples/FFT.py)                           echo 0   ;;
    samples/quickstart/*)                     echo 0   ;;
    *)                                        echo 0   ;;
  esac
}

sample_args() {
  local sample="$1"
  if [[ "${RUN_CORRECTNESS}" == "1" ]]; then
    case "${sample}" in
      samples/quickstart/*) ;;
      *) printf '%s\n' "--correctness-check" ;;
    esac
  fi
}

SAMPLES=(
  "samples/AllGatherMatmul.py"
  "samples/AttentionFMHA.py"
  "samples/BatchMatMul.py"
  "samples/FFT.py"
  "samples/LayerNorm.py"
  "samples/MatMul.py"
  "samples/MoE.py"
  "samples/Transpose.py"
  "samples/VectorAddition.py"
  "samples/quickstart/VectorAdd_quickstart.py"
  "samples/quickstart/VA_ovflo_nixnan.py"
)

printf 'sample_id\tsample\tsampling\texit_code\tnixnan_log_size\tstdout_log\tnixnan_log\n' > "${SUMMARY_FILE}"

for sample in "${SAMPLES[@]}"; do
  sample_id="$(sample_id_for "${sample}")"
  case_dir="${DATA_ROOT}/${sample_id}/data"
  stdout_log="${case_dir}/stdout.nnlog"
  nixnan_log="${case_dir}/nixnan.nnlog"
  env_log="${case_dir}/run.env"
  sampling_n="$(sampling_for "${sample}")"
  rc=0

  mkdir -p "${case_dir}"

  cmd=("${PYTHON_BIN}" "${ROOT_DIR}/${sample}")
  while IFS= read -r arg; do
    [[ -n "${arg}" ]] && cmd+=("${arg}")
  done < <(sample_args "${sample}")

  {
    printf 'sample_id=%s\n' "${sample_id}"
    printf 'sample=%s\n' "${sample}"
    printf 'run_correctness=%s\n' "${RUN_CORRECTNESS}"
    printf 'count=%s\n' "${COUNT}"
    printf 'sampling=%s\n' "${sampling_n}"
    printf 'histogram=1\n'
    printf 'bin_spec_file=%s\n' "${BIN_SPEC_FILE}"
    printf 'nixnan_so=%s\n' "${NIXNAN_SO}"
    printf 'python_bin=%s\n' "${PYTHON_BIN}"
    printf 'cutile_torch_site_packages=%s\n' "${CUTILE_TORCH_SITE_PACKAGES}"
    printf 'command='
    printf '%q' "${cmd[0]}"
    for arg in "${cmd[@]:1}"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  } > "${env_log}"

  printf '\n=== %s (SAMPLING=%s) ===\n' "${sample_id}" "${sampling_n}"
  set +e
  (
    cd "${case_dir}"
    LD_PRELOAD="${NIXNAN_SO}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      TOOL_VERBOSE="${TOOL_VERBOSE:-0}" \
      LINE_INFO="${LINE_INFO:-0}" \
      PRINT_ILL_INSTR="${PRINT_ILL_INSTR:-1}" \
      INSTR_MEM="${INSTR_MEM:-1}" \
      HISTOGRAM="${HISTOGRAM:-1}" \
      ENABLE_FUN_DETAIL="${ENABLE_FUN_DETAIL:-0}" \
      SAMPLING="${sampling_n}" \
      COUNT="${COUNT}" \
      BIN_SPEC_FILE="${BIN_SPEC_FILE}" \
      LOGFILE="${nixnan_log}" \
      PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}" \
      "${cmd[@]}" > "${stdout_log}" 2>&1
  )
  rc=$?
  set -e

  nnlog_size="$(stat -c%s "${nixnan_log}" 2>/dev/null || echo 0)"
  printf 'exit_code=%s\n' "${rc}" >> "${env_log}"
  printf 'nixnan_log_size=%s\n' "${nnlog_size}" >> "${env_log}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${sample_id}" "${sample}" "${sampling_n}" "${rc}" "${nnlog_size}" \
    "${stdout_log}" "${nixnan_log}" >> "${SUMMARY_FILE}"
  printf 'exit_code=%s  nixnan_log=%s bytes\n' "${rc}" "${nnlog_size}"
done

echo
echo "NixNan cuTile sample sweep complete."
echo "Data root: ${DATA_ROOT}"
echo "Summary:   ${SUMMARY_FILE}"
