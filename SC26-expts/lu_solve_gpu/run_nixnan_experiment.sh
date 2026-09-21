#!/usr/bin/env bash
# run_nixnan_experiment.sh
#
# Runs lu_solve_gpu.cu (a CUDA LU-factorization solver, ported from the
# FPChecker tutorial's tutorial/example_1/lu_solve.cpp) under nixnan
# (~/repos/nixnan/nixnan.so), reproducing the tutorial's "bad matrix ->
# floating-point exceptions" story on the GPU, plus two nixnan-specific
# instrumentation configurations:
#
#   Stage 1 (histogram):  exp-binade gathering with BIN_SPEC_FILE=bin_spec.json
#                          (HISTOGRAM=1), for both matrix.csv (well-conditioned)
#                          and bad_matrix.csv (ill-conditioned).
#   Stage 2 (sampling):   an exponential SAMPLING sweep on bad_matrix.csv:
#                          SAMPLING=1, 2, 4, 8, each held for 16 runs ("snaps"),
#                          then reset back to 1.
#
# All outputs are saved under logs/ in suitably named files; nothing is run
# without LD_PRELOAD=nixnan.so except the one-time uninstrumented baseline
# in Stage 0.
#
# --- Why Stage 2 is 16 separate process launches per level, not one run ---
# nixnan reads SAMPLING once at process init (see ~/repos/nixnan/src/nixnan.cu,
# GET_VAR_INT(sampling, "SAMPLING", ...)); there is no supported way to change
# it mid-run. So "every 1 for 16 snaps, then every 2 for 16 snaps, ..." is
# implemented here as 16 separate whole-program runs per level -- each run is
# one "snap". lu_solve_gpu is deterministic (fixed CSV input, no randomness),
# so the 16 snaps at a given level are expected to produce byte-identical
# output; what this stage demonstrates/records is the SAMPLING schedule and
# harness itself (skip-pattern across the ~19 repeated elimination-kernel
# invocations for this 20x20 matrix), not run-to-run variation.
#
# --- Note on BIN_SPEC_FILE's "doublings" field ---
# nixnan's docs (README.md, Tutorial.md) describe a "doublings" key for
# adaptive threshold scaling of the report count. As of this checkout,
# src/fp-histogram.cu never reads that key -- only "count" is used (see
# gen_bin_spec.py for details). bin_spec.json still sets it, for schema
# forward-compatibility, but it is currently a no-op; this is why Stage 2's
# exponential schedule is implemented in this script instead of relying on
# BIN_SPEC_FILE's doublings.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NIXNAN_SO="${NIXNAN_SO:-../../nixnan.so}"
EXECUTABLE="./lu_solve_gpu"
BIN_SPEC_FILE="./bin_spec.json"
LOG_DIR="./logs"

# Stage 2 schedule: SAMPLING levels, each held for SNAPS_PER_LEVEL runs
# ("every 1: 16 snaps, every 2: 16 snaps, every 4: 16 snaps, every 8: 16
# snaps -- then back"), repeated CYCLES times (default 1 pass through the
# schedule; set CYCLES=2+ to also see it reset and repeat).
SAMPLING_LEVELS=(1 2 4 8)
SNAPS_PER_LEVEL=16
CYCLES="${CYCLES:-1}"
SAMPLING_MATRIX="${SAMPLING_MATRIX:-bad_matrix.csv}"

echo "=========================================="
echo "lu_solve_gpu under nixnan"
echo "=========================================="
echo "NIXNAN_SO=$NIXNAN_SO"
[ -f "$NIXNAN_SO" ] || { echo "ERROR: $NIXNAN_SO not found. Build nixnan first (cd ../.. && make)."; exit 1; }

# ----------------------------------------------------------------------------
# Stage 0: build
# ----------------------------------------------------------------------------
echo ""
echo "[Stage 0] Building $EXECUTABLE ..."
make clean >/dev/null 2>&1 || true
make
[ -x "$EXECUTABLE" ] || { echo "ERROR: build failed"; exit 1; }
echo "Build OK."

mkdir -p "$LOG_DIR/baseline" "$LOG_DIR/histogram" "$LOG_DIR/sampling"

# ----------------------------------------------------------------------------
# Stage 0b: uninstrumented baseline (sanity check, no LD_PRELOAD)
# ----------------------------------------------------------------------------
echo ""
echo "[Stage 0b] Uninstrumented baseline runs (sanity check) ..."
for mat in matrix bad_matrix; do
    logfile="$LOG_DIR/baseline/${mat}.log"
    "$EXECUTABLE" "${mat}.csv" > "$logfile" 2>&1 || true
    echo "  $mat.csv -> $logfile"
done
echo "  (expect: matrix.csv residual ~1e-16; bad_matrix.csv residual/solution = nan)"

# ----------------------------------------------------------------------------
# Stage 1: exp-binade histogram gathering (HISTOGRAM=1 + BIN_SPEC_FILE)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[Stage 1] Exp-binade histogram gathering"
echo "=========================================="
echo "  BIN_SPEC_FILE=$BIN_SPEC_FILE (f64 exponent range [-1022,1023] in buckets of 4;"
echo "  f16/bf16/f32 left empty -- lu_solve_gpu only computes in double precision)"
[ -f "$BIN_SPEC_FILE" ] || { echo "ERROR: $BIN_SPEC_FILE not found. Run ./gen_bin_spec.py first."; exit 1; }

for mat in matrix bad_matrix; do
    logfile="$LOG_DIR/histogram/${mat}.log"
    echo "  Running HISTOGRAM=1 BIN_SPEC_FILE=$BIN_SPEC_FILE ./lu_solve_gpu ${mat}.csv -> $logfile"
    HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC_FILE" LOGFILE="$logfile" \
        LD_PRELOAD="$NIXNAN_SO" "$EXECUTABLE" "${mat}.csv" > "$LOG_DIR/histogram/${mat}.stdout" 2>&1 || true
    n_bins=$(grep -c "bin has reached threshold" "$logfile" || true)
    n_nan=$(grep -c "NaN:" "$logfile" || true)
    echo "    -> $n_bins binade-threshold reports"
done

# ----------------------------------------------------------------------------
# Stage 2: exponential SAMPLING sweep (1 -> 2 -> 4 -> 8 -> reset)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[Stage 2] Exponential SAMPLING sweep on $SAMPLING_MATRIX"
echo "=========================================="
echo "  Schedule: ${SAMPLING_LEVELS[*]} (each held for $SNAPS_PER_LEVEL snaps), $CYCLES cycle(s)"

for cycle in $(seq 1 "$CYCLES"); do
    cycle_dir="$LOG_DIR/sampling/cycle_$(printf '%02d' "$cycle")"
    for level in "${SAMPLING_LEVELS[@]}"; do
        level_dir="$cycle_dir/sampling_$level"
        mkdir -p "$level_dir"
        echo "  cycle $cycle, SAMPLING=$level: $SNAPS_PER_LEVEL snaps -> $level_dir/"
        for snap in $(seq 1 "$SNAPS_PER_LEVEL"); do
            snap_name="snap_$(printf '%02d' "$snap").log"
            SAMPLING="$level" LOGFILE="$level_dir/$snap_name" \
                LD_PRELOAD="$NIXNAN_SO" "$EXECUTABLE" "$SAMPLING_MATRIX" \
                > "$level_dir/snap_$(printf '%02d' "$snap").stdout" 2>&1 || true
        done
    done
    echo "  cycle $cycle complete. Resetting SAMPLING back to ${SAMPLING_LEVELS[0]}."
done

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
echo "Baseline logs:   $LOG_DIR/baseline/"
echo "Histogram logs:  $LOG_DIR/histogram/  (bin_spec.json binade reports + final exception summary)"
echo "Sampling logs:   $LOG_DIR/sampling/cycle_NN/sampling_<level>/snap_NN.log"
echo ""
echo "Quick check -- exception summary counts per histogram run:"
for mat in matrix bad_matrix; do
    echo "  --- $mat.csv ---"
    grep -A0 "NaN:\|Infinity:\|Division by 0:" "$LOG_DIR/histogram/${mat}.log" | grep -v " 0 (0 repeats)" || echo "    (no exceptions)"
done
