#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/nixnan_sweep/data}"
BIN_SPEC_FILE="${BIN_SPEC_FILE:-${DATA_ROOT}/bin_spec.json}"
SUMMARY_FILE="${SUMMARY_FILE:-${DATA_ROOT}/summary.tsv}"

if [[ -f "${ROOT_DIR}/cutile_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/cutile_env.sh"
fi

"${ROOT_DIR}/link_nixnan.sh" >/dev/null

PYTHON_BIN="${PYTHON_BIN:-python}"
NIXNAN_SO="${NIXNAN_SO:-${ROOT_DIR}/nixnan.so}"

MATRIX_SIZES="${MATRIX_SIZES:-4}"
MATRIX_KINDS="${MATRIX_KINDS:-hpl-ai conditioned}"
CONDITION_NUMBERS="${CONDITION_NUMBERS:-10}"
MAX_ITER="${MAX_ITER:-50}"
SEED="${SEED:-1}"
GMRES="${GMRES:-1}"
COMPARE_CPU="${COMPARE_CPU:-1}"
BACKEND="${BACKEND:-cutile}"
TENSOR_CORE_PROBE="${TENSOR_CORE_PROBE:-1}"
TENSOR_CORE_SIZE="${TENSOR_CORE_SIZE:-64}"
VERIFY_TENSOR_CORES="${VERIFY_TENSOR_CORES:-1}"

# Each token is input,factor,solve,refinement,residual.
SOLVER_PRECISION_CONFIGS="${SOLVER_PRECISION_CONFIGS:-float64,float32,float32,float64,float64 float64,float16,float16,float64,float64}"

mkdir -p "${DATA_ROOT}"

if [[ ! -f "${BIN_SPEC_FILE}" ]]; then
  if [[ -f "${ROOT_DIR}/../pytorch-issues/issue181806/data/bin_spec.json" ]]; then
    cp "${ROOT_DIR}/../pytorch-issues/issue181806/data/bin_spec.json" "${BIN_SPEC_FILE}"
  else
    cat > "${BIN_SPEC_FILE}" <<'JSON'
{
  "count": 1024,
  "bf16": [],
  "f16": [[-14, 15]],
  "f32": [[-126, 127]],
  "f64": [[-1022, 1023]]
}
JSON
  fi
fi

if ! [[ -f "${NIXNAN_SO}" ]]; then
  echo "ERROR: NixNan preload library is missing: ${NIXNAN_SO}" >&2
  exit 1
fi

verify_tensor_cores() {
  local verify_dir="${DATA_ROOT}/tensor_core_verify"
  local stdout_log="${verify_dir}/stdout.nnlog"
  local nixnan_log="${verify_dir}/nixnan.verbose.nnlog"

  mkdir -p "${verify_dir}"

  (
    cd "${verify_dir}"
    LD_PRELOAD="${NIXNAN_SO}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      TOOL_VERBOSE=1 \
      LINE_INFO="${LINE_INFO:-0}" \
      PRINT_ILL_INSTR="${PRINT_ILL_INSTR:-1}" \
      INSTR_MEM="${INSTR_MEM:-1}" \
      HISTOGRAM="${HISTOGRAM:-1}" \
      ENABLE_FUN_DETAIL=1 \
      SAMPLING=0 \
      BIN_SPEC_FILE="${BIN_SPEC_FILE}" \
      LOGFILE="${nixnan_log}" \
      PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" -c "from hpl_ai.cutile_hpl_ai import run_tensor_core_probe; checksum = run_tensor_core_probe(${TENSOR_CORE_SIZE}); print('tensor_core_probe=true'); print('tensor_core_operation=wmma_mma_sync_fp16_fp16_accumulate_fp32'); print('tensor_core_size=${TENSOR_CORE_SIZE}'); print(f'tensor_core_checksum={checksum:.6e}')" \
      > "${stdout_log}" 2>&1
  )

  if ! grep -q "HMMA" "${nixnan_log}"; then
    echo "ERROR: tensor-core verification did not find HMMA in ${nixnan_log}" >&2
    exit 1
  fi
}

if [[ "${TENSOR_CORE_PROBE}" == "1" && "${VERIFY_TENSOR_CORES}" == "1" ]]; then
  echo "==> Verifying tensor-core HMMA instrumentation..."
  verify_tensor_cores
fi

printf 'case_id\tn\tmatrix_kind\tcondition_number\tinput_precision\tfactor_precision\tsolve_precision\trefinement_precision\tresidual_precision\texit_code\tstdout_log\tnixnan_log\n' > "${SUMMARY_FILE}"

sanitize() {
  printf '%s' "$1" | tr ',=./:' '_____'
}

run_case() {
  local n="$1"
  local matrix_kind="$2"
  local condition_number="$3"
  local input_precision="$4"
  local factor_precision="$5"
  local solve_precision="$6"
  local refinement_precision="$7"
  local residual_precision="$8"

  local condition_label="na"
  if [[ "${matrix_kind}" == "conditioned" ]]; then
    condition_label="$(sanitize "${condition_number}")"
  fi

  local precision_label
  precision_label="$(sanitize "in_${input_precision},factor_${factor_precision},solve_${solve_precision},refine_${refinement_precision},residual_${residual_precision}")"

  local case_id="n${n}_${matrix_kind}_cond${condition_label}_${precision_label}"
  local case_dir="${DATA_ROOT}/${case_id}"
  local stdout_log="${case_dir}/stdout.nnlog"
  local nixnan_log="${case_dir}/nixnan.nnlog"
  local env_log="${case_dir}/run.env"
  local rc=0

  mkdir -p "${case_dir}"

  local cmd=(
    "${PYTHON_BIN}"
    "${ROOT_DIR}/hpl_ai/cutile_hpl_ai.py"
    --backend "${BACKEND}"
    --n "${n}"
    --matrix-kind "${matrix_kind}"
    --max-iter "${MAX_ITER}"
    --seed "${SEED}"
    --input-precision "${input_precision}"
    --factor-precision "${factor_precision}"
    --solve-precision "${solve_precision}"
    --refinement-precision "${refinement_precision}"
    --residual-precision "${residual_precision}"
  )

  if [[ "${matrix_kind}" == "conditioned" ]]; then
    cmd+=(--condition-number "${condition_number}")
  fi

  if [[ "${GMRES}" == "0" ]]; then
    cmd+=(--no-gmres)
  fi

  if [[ "${COMPARE_CPU}" == "1" && "${BACKEND}" == "cutile" ]]; then
    cmd+=(--compare-cpu)
  fi

  if [[ "${TENSOR_CORE_PROBE}" == "1" ]]; then
    cmd+=(--tensor-core-probe --tensor-core-size "${TENSOR_CORE_SIZE}")
  fi

  {
    printf 'case_id=%s\n' "${case_id}"
    printf 'n=%s\n' "${n}"
    printf 'matrix_kind=%s\n' "${matrix_kind}"
    printf 'condition_number=%s\n' "${condition_number}"
    printf 'input_precision=%s\n' "${input_precision}"
    printf 'factor_precision=%s\n' "${factor_precision}"
    printf 'solve_precision=%s\n' "${solve_precision}"
    printf 'refinement_precision=%s\n' "${refinement_precision}"
    printf 'residual_precision=%s\n' "${residual_precision}"
    printf 'backend=%s\n' "${BACKEND}"
    printf 'gmres=%s\n' "${GMRES}"
    printf 'compare_cpu=%s\n' "${COMPARE_CPU}"
    printf 'tensor_core_probe=%s\n' "${TENSOR_CORE_PROBE}"
    printf 'tensor_core_size=%s\n' "${TENSOR_CORE_SIZE}"
    printf 'nixnan_so=%s\n' "${NIXNAN_SO}"
    printf 'bin_spec_file=%s\n' "${BIN_SPEC_FILE}"
    printf 'sampling=0\n'
    printf 'histogram_count=1024\n'
    printf 'command='
    printf '%q' "${cmd[0]}"
    local arg
    for arg in "${cmd[@]:1}"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  } > "${env_log}"

  printf '\n=== %s ===\n' "${case_id}"
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
      SAMPLING=0 \
      BIN_SPEC_FILE="${BIN_SPEC_FILE}" \
      LOGFILE="${nixnan_log}" \
      "${cmd[@]}" > "${stdout_log}" 2>&1
  )
  rc=$?
  set -e

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_id}" "${n}" "${matrix_kind}" "${condition_number}" \
    "${input_precision}" "${factor_precision}" "${solve_precision}" \
    "${refinement_precision}" "${residual_precision}" "${rc}" \
    "${stdout_log}" "${nixnan_log}" >> "${SUMMARY_FILE}"

  printf 'exit_code=%s\n' "${rc}" >> "${env_log}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "case failed: ${case_id}" >&2
    return "${rc}"
  fi
}

for n in ${MATRIX_SIZES}; do
  for matrix_kind in ${MATRIX_KINDS}; do
    if [[ "${matrix_kind}" == "hpl-ai" ]]; then
      condition_values="n/a"
    elif [[ "${matrix_kind}" == "conditioned" ]]; then
      condition_values="${CONDITION_NUMBERS}"
    else
      echo "Unsupported MATRIX_KINDS entry: ${matrix_kind}" >&2
      exit 2
    fi

    for condition_number in ${condition_values}; do
      for precision_config in ${SOLVER_PRECISION_CONFIGS}; do
        IFS=',' read -r input_precision factor_precision solve_precision refinement_precision residual_precision <<< "${precision_config}"
        run_case "${n}" "${matrix_kind}" "${condition_number}" \
          "${input_precision}" "${factor_precision}" "${solve_precision}" \
          "${refinement_precision}" "${residual_precision}"
      done
    done
  done
done

echo
echo "NixNan cuTile sweep complete."
echo "Data root: ${DATA_ROOT}"
echo "Summary:   ${SUMMARY_FILE}"
