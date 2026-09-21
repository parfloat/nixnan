#!/usr/bin/env bash
# run_nixnan_experiment.sh
#
# Runs reaction_diffusion_gpu.cu (a CUDA port of the FPChecker tutorial's
# tutorial/example_2/reaction_diffusion.cpp: a 1D reaction-diffusion PDE,
# du/dt = D*d2u/dx2 + lambda*u, solved explicitly with L=1, T=4, N=101,
# D=0.01, lambda=25, M=80000 steps) under nixnan (~/repos/nixnan/nixnan.so),
# comparing the FP64 and FP32 builds exactly as the CPU tutorial's Part A /
# Part B do -- except both builds exist as separate binaries here
# (reaction_diffusion_gpu_fp64 / _fp32) instead of hand-editing a typedef.
#
# Stages:
#   0  Build + uninstrumented baseline sanity runs (both precisions).
#   1  Exp-binade histogram gathering (HISTOGRAM=1 + BIN_SPEC_FILE), both
#      precisions -- see gen_bin_spec.py / bin_spec.json.
#   2  Exponential SAMPLING sweep on the FP32 build (the interesting
#      exception-producing case): SAMPLING=1,2,4,8, 16 runs ("snaps") per
#      level, then reset back to 1 -- same schedule and same caveat as
#      ../lu_solve_gpu/run_nixnan_experiment.sh (SAMPLING is a no-op in
#      this nixnan build; see that experiment's README/report).
#   3  "Interesting exception settings": two nixnan features not exercised
#      in the lu_solve_gpu experiment --
#        3a PRINT_ILL_INSTR=1 + MAX_ERRORS=20: source-line- and
#           SASS-instruction-attributed early-exit diagnostic, for both
#           precisions (FP32 is expected to hit the cap early; FP64 is
#           expected to run to completion without ever reaching it).
#        3b INSTR_MEM=1: memory-instrumentation, to show NaN/Inf values
#           actually being written to GPU global memory (the u_next
#           buffer), not just appearing transiently in registers.
#
# All outputs are saved under logs/ in suitably named files.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NIXNAN_SO="${NIXNAN_SO:-../../nixnan.so}"
FP64_EXE="./reaction_diffusion_gpu_fp64"
FP32_EXE="./reaction_diffusion_gpu_fp32"
BIN_SPEC_FILE="./bin_spec.json"
LOG_DIR="./logs"

# Stage 2 schedule (see ../lu_solve_gpu/run_nixnan_experiment.sh for the
# rationale: nixnan reads SAMPLING once at process start, so each "snap" is
# a separate whole-program launch).
SAMPLING_LEVELS=(1 2 4 8)
SNAPS_PER_LEVEL=16
CYCLES="${CYCLES:-1}"
SAMPLING_EXE="${SAMPLING_EXE:-$FP32_EXE}"

echo "=========================================="
echo "reaction_diffusion_gpu under nixnan"
echo "=========================================="
echo "NIXNAN_SO=$NIXNAN_SO"
[ -f "$NIXNAN_SO" ] || { echo "ERROR: $NIXNAN_SO not found. Build nixnan first (cd ../.. && make)."; exit 1; }

# ----------------------------------------------------------------------------
# Stage 0: build + uninstrumented baseline
# ----------------------------------------------------------------------------
echo ""
echo "[Stage 0] Building $FP64_EXE and $FP32_EXE ..."
make clean >/dev/null 2>&1 || true
make
[ -x "$FP64_EXE" ] && [ -x "$FP32_EXE" ] || { echo "ERROR: build failed"; exit 1; }
echo "Build OK."

mkdir -p "$LOG_DIR/baseline" "$LOG_DIR/histogram" "$LOG_DIR/sampling" "$LOG_DIR/diagnostic" "$LOG_DIR/memory"

echo ""
echo "[Stage 0b] Uninstrumented baseline runs (sanity check) ..."
for prec in fp64 fp32; do
    exe="./reaction_diffusion_gpu_${prec}"
    logfile="$LOG_DIR/baseline/${prec}.log"
    "$exe" > "$logfile" 2>&1 || true
    echo "  $prec -> $logfile"
done
echo "  (expect: fp64 completes with max ~1.7e43 at t=4; fp32's naive host-side"
echo "   max-reduction silently reports 0.0 once Inf/NaN enters the array --"
echo "   see the report for why, and Stage 3 for nixnan catching it anyway)"

# ----------------------------------------------------------------------------
# Stage 1: exp-binade histogram gathering
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[Stage 1] Exp-binade histogram gathering"
echo "=========================================="
[ -f "$BIN_SPEC_FILE" ] || { echo "ERROR: $BIN_SPEC_FILE not found. Run ./gen_bin_spec.py first."; exit 1; }
echo "  BIN_SPEC_FILE=$BIN_SPEC_FILE (f32 range [-126,127] + f64 range [-1022,1023],"
echo "  both in buckets of 4; f16/bf16 left empty -- this program never uses them)"

for prec in fp64 fp32; do
    exe="./reaction_diffusion_gpu_${prec}"
    logfile="$LOG_DIR/histogram/${prec}.log"
    echo "  Running HISTOGRAM=1 BIN_SPEC_FILE=$BIN_SPEC_FILE $exe -> $logfile"
    HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC_FILE" LOGFILE="$logfile" \
        LD_PRELOAD="$NIXNAN_SO" "$exe" > "$LOG_DIR/histogram/${prec}.stdout" 2>&1 || true
    n_bins=$(grep -c "bin has reached threshold" "$logfile" || true)
    echo "    -> $n_bins binade-threshold reports"
done

# ----------------------------------------------------------------------------
# Stage 2: exponential SAMPLING sweep
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[Stage 2] Exponential SAMPLING sweep on $SAMPLING_EXE"
echo "=========================================="
echo "  Schedule: ${SAMPLING_LEVELS[*]} (each held for $SNAPS_PER_LEVEL snaps), $CYCLES cycle(s)"

for cycle in $(seq 1 "$CYCLES"); do
    cycle_dir="$LOG_DIR/sampling/cycle_$(printf '%02d' "$cycle")"
    for level in "${SAMPLING_LEVELS[@]}"; do
        level_dir="$cycle_dir/sampling_$level"
        mkdir -p "$level_dir"
        echo "  cycle $cycle, SAMPLING=$level: $SNAPS_PER_LEVEL snaps -> $level_dir/"
        for snap in $(seq 1 "$SNAPS_PER_LEVEL"); do
            snap_pad=$(printf '%02d' "$snap")
            SAMPLING="$level" LOGFILE="$level_dir/snap_$snap_pad.log" \
                LD_PRELOAD="$NIXNAN_SO" "$SAMPLING_EXE" \
                > "$level_dir/snap_$snap_pad.stdout" 2>&1 || true
        done
    done
    echo "  cycle $cycle complete. Resetting SAMPLING back to ${SAMPLING_LEVELS[0]}."
done

# ----------------------------------------------------------------------------
# Stage 3: interesting exception settings
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[Stage 3] Interesting exception settings"
echo "=========================================="

echo "  [3a] PRINT_ILL_INSTR=1 MAX_ERRORS=20 (source-line + SASS attributed early-exit diagnostic)"
for prec in fp64 fp32; do
    exe="./reaction_diffusion_gpu_${prec}"
    logfile="$LOG_DIR/diagnostic/${prec}.log"
    PRINT_ILL_INSTR=1 MAX_ERRORS=20 LOGFILE="$logfile" \
        LD_PRELOAD="$NIXNAN_SO" "$exe" > "$LOG_DIR/diagnostic/${prec}.stdout" 2>&1 || true
    n_err=$(grep -c "#nixnan: error" "$logfile" || true)
    echo "    $prec -> $logfile ($n_err error lines)"
done

echo "  [3b] INSTR_MEM=1 (memory-instrumentation: NaN/Inf flowing into GPU global memory)"
for prec in fp64 fp32; do
    exe="./reaction_diffusion_gpu_${prec}"
    logfile="$LOG_DIR/memory/${prec}.log"
    INSTR_MEM=1 LOGFILE="$logfile" \
        LD_PRELOAD="$NIXNAN_SO" "$exe" > "$LOG_DIR/memory/${prec}.stdout" 2>&1 || true
    echo "    $prec -> $logfile"
done

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
echo "Baseline logs:    $LOG_DIR/baseline/"
echo "Histogram logs:   $LOG_DIR/histogram/"
echo "Sampling logs:    $LOG_DIR/sampling/cycle_NN/sampling_<level>/snap_NN.log"
echo "Diagnostic logs:  $LOG_DIR/diagnostic/  (PRINT_ILL_INSTR + MAX_ERRORS)"
echo "Memory logs:      $LOG_DIR/memory/      (INSTR_MEM)"
echo ""
echo "Quick check -- exception summary counts per histogram run:"
for prec in fp64 fp32; do
    echo "  --- $prec ---"
    grep -A0 "NaN:\|Infinity:\|Division by 0:" "$LOG_DIR/histogram/${prec}.log" | grep -v " 0 (0 repeats)" || echo "    (no exceptions)"
done
