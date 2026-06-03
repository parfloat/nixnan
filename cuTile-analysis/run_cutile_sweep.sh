#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${ROOT_DIR}/cutile_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/cutile_env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND="${BACKEND:-cutile}"
MATRIX_SIZES="${MATRIX_SIZES:-16 32 64}"
MATRIX_KINDS="${MATRIX_KINDS:-hpl-ai conditioned}"
CONDITION_NUMBERS="${CONDITION_NUMBERS:-10 1000 100000}"
INPUT_PRECISIONS="${INPUT_PRECISIONS:-float64}"
FACTOR_PRECISIONS="${FACTOR_PRECISIONS:-float32}"
SOLVE_PRECISIONS="${SOLVE_PRECISIONS:-float32}"
REFINEMENT_PRECISIONS="${REFINEMENT_PRECISIONS:-float64}"
RESIDUAL_PRECISIONS="${RESIDUAL_PRECISIONS:-float64}"
MAX_ITER="${MAX_ITER:-50}"
SEED="${SEED:-1}"
GMRES="${GMRES:-1}"
COMPARE_CPU="${COMPARE_CPU:-0}"

if [[ "${BACKEND}" != "cpu" && "${BACKEND}" != "cutile" ]]; then
  echo "BACKEND must be cpu or cutile, got: ${BACKEND}" >&2
  exit 2
fi

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
      for input_precision in ${INPUT_PRECISIONS}; do
        for factor_precision in ${FACTOR_PRECISIONS}; do
          for solve_precision in ${SOLVE_PRECISIONS}; do
            for refinement_precision in ${REFINEMENT_PRECISIONS}; do
              for residual_precision in ${RESIDUAL_PRECISIONS}; do
                cmd=(
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

                printf '\n=== n=%s matrix=%s cond=%s input=%s factor=%s solve=%s refine=%s residual=%s backend=%s gmres=%s ===\n' \
                  "${n}" "${matrix_kind}" "${condition_number}" "${input_precision}" \
                  "${factor_precision}" "${solve_precision}" "${refinement_precision}" \
                  "${residual_precision}" "${BACKEND}" "${GMRES}"
                "${cmd[@]}"
              done
            done
          done
        done
      done
    done
  done
done
