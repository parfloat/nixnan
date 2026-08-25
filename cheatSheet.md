# NixNan Cheat Sheet: rd_nixnan.cu Demonstration

A progressive guide to demonstrating NixNan features using the reaction-diffusion FTCS example with FP16, BF16, FP32, and FP64 kernels.

---

## Prerequisites

```bash
# Ensure binary is built with FP64 support
ls -lh ./rd_nixnan

# Verify nixnan.so is available
ls -lh ./nixnan.so

# Check current directory
pwd
```

---

## Part 1: Basic NixNan Features

### 1.1: Run without instrumentation (baseline)

Shows the raw output without any exception detection:

```bash
./rd_nixnan
```

**Expected output:**
- Four simulations: FP16, BF16, FP32, FP64
- FP16 first non-finite at step 300
- BF16, FP32, and FP64 first non-finite at step 1900
- Final summary report with exception counts

### 1.2: Basic exception detection with NixNan

Detect all floating-point exceptions across all precisions:

```bash
LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Expected output:**
- Same simulation results as baseline
- Summary at end showing NaN/Infinity counts per precision (FP16, BF16, FP32, FP64)

### 1.3: Save output to log file

Capture all output to a file for analysis:

```bash
LOGFILE=./basic_run.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Check the log:**
```bash
tail -50 ./basic_run.log
```

### 1.4: Enable simple histogram (global exponent ranges)

See overall min/max exponents observed across all formats:

```bash
HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Expected output at end:**
```
#nixnan: --- FP exponent ranges --- 
#nixnan: Exponent range for f16: [zero, inf]
#nixnan: Exponent range for bf16: [zero, inf]
#nixnan: Exponent range for f32: [zero, inf]
#nixnan: Exponent range for f64: [zero, inf]
```

---

## Part 2: Binade-Targeted Monitoring with spec.json

### 2.1: Auto-generate template spec.json

First run will create a template and exit:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Expected output:**
```
Created template bin specification file at ./spec.json
Exiting now. Please edit the file to specify which exponent ranges to report.
```

### 2.2: Create custom spec.json for overflow monitoring (all precisions)

Replace the template with a spec designed to monitor overflow ranges across all precision formats:

```bash
cat > spec.json << 'EOF'
{
    "count": 256,
    "doublings": 7,
    "bf16": [[120, 127]],
    "f16":  [[13, 15]],
    "f32":  [[120, 127]],
    "f64":  [[1015, 1023]]
}
EOF
```

**Verify it was created:**
```bash
cat spec.json
```

**What this spec does:**
- `count: 256` - Report every 256 exceptions in these ranges
- `doublings: 7` - Enable adaptive threshold doubling (256→512→1024→...→32768→reset)
- `f16 [[13,15]]` - Monitor FP16 exponents 13-15 (overflow range, max ≈65504)
- `bf16 [[120,127]]` - Monitor BF16 exponents 120-127 (overflow range, max ≈3.4e38)
- `f32 [[120,127]]` - Monitor FP32 exponents 120-127 (overflow range, max ≈3.4e38)
- `f64 [[1015,1023]]` - Monitor FP64 exponents 1015-1023 (overflow range, max ≈1.8e308)

### 2.3: Run with binade monitoring

Now run the simulation with the spec:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Expected output (sample):**
```
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=256
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=512
...
#nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=256
#nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=512
... (multiple doubling cycles)
#nixnan: f64 bin has reached threshold: kernel=rd_step_fp64 range=[1015,1023] count=256
... (FP64 will also hit thresholds)
```

### 2.4: Save binade output to log

Capture binade monitoring to a file:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./binade_analysis.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Examine just the binade hits:**
```bash
grep "bin has reached threshold" ./binade_analysis.log | head -20
```

**Count total binade threshold hits:**
```bash
grep "bin has reached threshold" ./binade_analysis.log | wc -l
```

### 2.5: Compare all precision overflow behavior

See how different precisions handle the same simulation:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./precision_compare.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Extract binade hits by precision:**
```bash
echo "FP16 hits:"
grep "f16 bin has reached threshold" ./precision_compare.log | wc -l

echo "BF16 hits:"
grep "bf16 bin has reached threshold" ./precision_compare.log | wc -l

echo "FP32 hits:"
grep "f32 bin has reached threshold" ./precision_compare.log | wc -l

echo "FP64 hits:"
grep "f64 bin has reached threshold" ./precision_compare.log | wc -l
```

---

## Part 3: Exponential/Adaptive Sampling

### 3.1: Understand adaptive threshold doubling

Threshold progression with `doublings: 7`:

```bash
echo "Threshold progression:"
python3 << 'PYEOF'
count = 256
for i in range(8):
    print(f"  Doubling {i}: count={count}")
    count *= 2
PYEOF
```

**Output shows:**
```
Threshold progression:
  Doubling 0: count=256
  Doubling 1: count=512
  Doubling 2: count=1024
  Doubling 3: count=2048
  Doubling 4: count=4096
  Doubling 5: count=8192
  Doubling 6: count=16384
  Doubling 7: count=32768
```

### 3.2: Run with aggressive doubling (observe scaling quickly)

Create a spec with fewer doublings to see resets sooner:

```bash
cat > spec_aggressive.json << 'EOF'
{
    "count": 256,
    "doublings": 3,
    "bf16": [[120, 127]],
    "f16":  [[13, 15]],
    "f32":  [[120, 127]],
    "f64":  [[1015, 1023]]
}
EOF
```

**Run simulation:**
```bash
BIN_SPEC_FILE=./spec_aggressive.json HISTOGRAM=1 LOGFILE=./aggressive_doubling.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Count resets (when count goes back to 256):**
```bash
grep "bf16.*count=256" ./aggressive_doubling.log | wc -l
```

### 3.3: Run with conservative doubling (capture fine-grained changes)

Create a spec with many doublings to see more intermediate steps:

```bash
cat > spec_conservative.json << 'EOF'
{
    "count": 128,
    "doublings": 10,
    "bf16": [[120, 127]],
    "f16":  [[13, 15]],
    "f32":  [[120, 127]],
    "f64":  [[1015, 1023]]
}
EOF
```

**Run simulation:**
```bash
BIN_SPEC_FILE=./spec_conservative.json HISTOGRAM=1 LOGFILE=./conservative_doubling.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Show all FP64 threshold milestones:**
```bash
grep "f64.*rd_step_fp64" ./conservative_doubling.log | head -25
```

### 3.4: Kernel invocation sampling (reduce overhead)

Enable sampling to instrument only every 2nd kernel invocation:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LOGFILE=./sampled_k2.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Compare threshold hit counts with/without sampling:**
```bash
echo "Without sampling:"
grep "bin has reached threshold" ./binade_analysis.log 2>/dev/null | wc -l

echo "With SAMPLING=2:"
grep "bin has reached threshold" ./sampled_k2.log 2>/dev/null | wc -l
```

### 3.5: Higher sampling factor

Instrument only every 4th kernel invocation (coarser sampling):

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=4 LOGFILE=./sampled_k4.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Compare three sampling modes:**
```bash
echo "No sampling (SAMPLING=0):"
grep "bin has reached threshold" ./binade_analysis.log 2>/dev/null | wc -l

echo "SAMPLING=2:"
grep "bin has reached threshold" ./sampled_k2.log 2>/dev/null | wc -l

echo "SAMPLING=4:"
grep "bin has reached threshold" ./sampled_k4.log 2>/dev/null | wc -l
```

### 3.6: Combine adaptive doubling + sampling + logging

Full-featured run with all options enabled:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LOGFILE=./full_analysis.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Show execution summary with all four precisions:**
```bash
echo "=== Simulation summary ==="
grep "first non-finite at step" ./full_analysis.log

echo -e "\n=== Binade hits by precision ==="
echo "FP16: $(grep 'f16 bin' ./full_analysis.log | wc -l)"
echo "BF16: $(grep 'bf16 bin' ./full_analysis.log | wc -l)"
echo "FP32: $(grep 'f32 bin' ./full_analysis.log | wc -l)"
echo "FP64: $(grep 'f64 bin' ./full_analysis.log | wc -l)"

echo -e "\n=== Exception summary ==="
grep "Operations ---" -A 5 ./full_analysis.log | tail -30
```

---

## Part 4: Analysis and Comparison Commands

### 4.1: Create comparison specs

**For underflow monitoring (all precisions):**
```bash
cat > spec_underflow.json << 'EOF'
{
    "count": 256,
    "doublings": 7,
    "bf16": [[-126, -115]],
    "f16":  [[-14, -10]],
    "f32":  [[-126, -100]],
    "f64":  [[-1022, -900]]
}
EOF
```

**For full range monitoring (all precisions):**
```bash
cat > spec_full_range.json << 'EOF'
{
    "count": 512,
    "doublings": 5,
    "bf16": [[0, 127], [120, 127]],
    "f16":  [[0, 15], [13, 15]],
    "f32":  [[0, 127], [120, 127]],
    "f64":  [[0, 1023], [1015, 1023]]
}
EOF
```

### 4.2: Quick precision comparison (all four formats)

Run analysis across all precisions:

```bash
echo "=== FP16 Overflow (spec.json) ==="
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan 2>&1 | \
  grep -E "f16|first non-finite"

echo -e "\n=== BF16 Overflow (spec.json) ==="
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan 2>&1 | \
  grep -E "bf16|first non-finite"

echo -e "\n=== FP32 Overflow (spec.json) ==="
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan 2>&1 | \
  grep -E "f32|first non-finite"

echo -e "\n=== FP64 Overflow (spec.json) ==="
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan 2>&1 | \
  grep -E "f64|first non-finite"
```

### 4.3: Show exception progression across all precisions

Extract exception counts for all formats:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./exception_summary.log LD_PRELOAD=./nixnan.so ./rd_nixnan

echo "Exception summary by precision:"
echo "================================"
echo ""
echo "FP16:"
grep "FP16 Operations" -A 5 ./exception_summary.log | grep -E "NaN|Infinity"
echo ""
echo "BF16:"
grep "BF16 Operations" -A 5 ./exception_summary.log | grep -E "NaN|Infinity"
echo ""
echo "FP32:"
grep "FP32 Operations" -A 5 ./exception_summary.log | grep -E "NaN|Infinity"
echo ""
echo "FP64:"
grep "FP64 Operations" -A 5 ./exception_summary.log | grep -E "NaN|Infinity"
```

### 4.4: Binade hits breakdown for all precisions

Extract binade hits for each format:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./all_precisions.log LD_PRELOAD=./nixnan.so ./rd_nixnan

echo "Binade hits by precision:"
echo "========================="
echo "FP16 hits (range [13,15]):"
grep "f16 bin has reached threshold" ./all_precisions.log | wc -l
echo ""
echo "BF16 hits (range [120,127]):"
grep "bf16 bin has reached threshold" ./all_precisions.log | wc -l
echo ""
echo "FP32 hits (range [120,127]):"
grep "f32 bin has reached threshold" ./all_precisions.log | wc -l
echo ""
echo "FP64 hits (range [1015,1023]):"
grep "f64 bin has reached threshold" ./all_precisions.log | wc -l
```

### 4.5: Detailed threshold progression for FP64

Show how FP64 thresholds progress through doublings:

```bash
echo "=== FP64 Threshold Progression ==="
grep "f64.*rd_step_fp64.*range=\[1015,1023\]" ./all_precisions.log | head -20
```

---

## Cleanup Commands

### Remove analysis files

```bash
rm -f binade_analysis.log basic_run.log precision_compare.log \
      aggressive_doubling.log conservative_doubling.log \
      sampled_k2.log sampled_k4.log full_analysis.log exception_summary.log \
      all_precisions.log
```

### Clean spec files

```bash
rm -f spec_aggressive.json spec_conservative.json spec_underflow.json spec_full_range.json
```

### Keep only main spec.json

```bash
# Keep spec.json for future use
ls -l spec.json
```

---

## Quick Reference: Commands by Use Case

### Want to see overflow monitoring for all precisions?
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan
```

### Want to see adaptive doubling in action?
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./adaptive.log LD_PRELOAD=./nixnan.so ./rd_nixnan
grep "bin has reached threshold" ./adaptive.log
```

### Want to compare sampling strategies?
```bash
# No sampling
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./s0.log LD_PRELOAD=./nixnan.so ./rd_nixnan
# Sample every 2nd
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LOGFILE=./s2.log LD_PRELOAD=./nixnan.so ./rd_nixnan
# Compare counts
echo "SAMPLING=0: $(grep 'bin has reached' ./s0.log | wc -l)"
echo "SAMPLING=2: $(grep 'bin has reached' ./s2.log | wc -l)"
```

### Want to see all four precisions in one run?
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./all_formats.log LD_PRELOAD=./nixnan.so ./rd_nixnan
echo "Format breakdown:"
echo "FP16: $(grep 'f16 bin' ./all_formats.log | wc -l) hits"
echo "BF16: $(grep 'bf16 bin' ./all_formats.log | wc -l) hits"
echo "FP32: $(grep 'f32 bin' ./all_formats.log | wc -l) hits"
echo "FP64: $(grep 'f64 bin' ./all_formats.log | wc -l) hits"
```

### Want to see kernel-specific exceptions?
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./kernels.log LD_PRELOAD=./nixnan.so ./rd_nixnan
echo "FP16 kernel hits:"
grep "rd_step_fp16" ./kernels.log | wc -l
echo "BF16 kernel hits:"
grep "rd_step_bf16" ./kernels.log | wc -l
echo "FP32 kernel hits:"
grep "rd_step_fp32" ./kernels.log | wc -l
echo "FP64 kernel hits:"
grep "rd_step_fp64" ./kernels.log | wc -l
```

---

## Notes for Live Demonstration

1. **First demo**: Run `./rd_nixnan` (no instrumentation) to show baseline behavior across all four precisions
2. **Second demo**: Run with `LD_PRELOAD=./nixnan.so` to show exception detection
3. **Third demo**: Run with `BIN_SPEC_FILE=./spec.json HISTOGRAM=1` to show binade monitoring with FP64 support
4. **Fourth demo**: Show adaptive doubling by examining log output
5. **Fifth demo**: Compare sampling strategies side-by-side
6. **Sixth demo**: Break down binade hits by all four precision formats
7. **Conclude**: Show exception summary from final run with FP16, BF16, FP32, and FP64

Each command is copy-paste ready and will run against rd_nixnan.cu with FP64 kernel included.

---

**Last Updated**: 2026-08-25  
**rd_nixnan.cu version**: 4 kernels (FP16, BF16, FP32, FP64)  
**Tested with**: rd_nixnan binary with FP64 support  
**Tool**: NixNan with adaptive threshold doubling support  
**Format Support**: FP16, BF16, FP32, FP64 (all four precisions)
