#!/bin/bash
##===----------------------------------------------------------------------===##
## simpleExample.sh - Demonstrate nixnan FP histogram instrumentation
##
## This script builds and runs testManyFMTs with nixnan instrumentation,
## showing how to collect floating-point operation histograms across
## multiple precisions (FP16, FP32, FP64) and binades.
##
## Usage: ./simpleExample.sh
##===----------------------------------------------------------------------===##

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NIXNAN_SO="../../nixnan.so"

echo ""
echo -e "${BOLD}${CYAN}============================================================${NC}"
echo -e "${BOLD}${CYAN}    nixnan FP Histogram - Simple Example${NC}"
echo -e "${BOLD}${CYAN}============================================================${NC}"
echo ""

echo -e "${YELLOW}What is nixnan?${NC}"
echo ""
echo "nixnan is an NVBit-based GPU binary instrumentation tool that"
echo "monitors floating-point operations on NVIDIA GPUs. It can:"
echo ""
echo "  * Track FP16, FP32, and FP64 operations"
echo "  * Detect special values (NaN, Inf, denormals)"
echo "  * Generate histograms of operand/result magnitudes by binade"
echo "  * Help identify numerical stability issues in GPU code"
echo ""

echo -e "${YELLOW}What does this example do?${NC}"
echo ""
echo "testManyFMTs generates 200 random values in range [0.0001, 16.0]"
echo "for each floating-point format (FP16, FP32, FP64) and performs"
echo "arithmetic operations (add, mul, fma, sub) to exercise the GPU's"
echo "floating-point units across multiple binades (~17 binades)."
echo ""
echo "When run with nixnan:"
echo "  - Without HISTOGRAM: Detects special values (NaN, Inf, etc.)"
echo "  - With HISTOGRAM=1:  Also collects binade distribution histograms"
echo ""

echo -e "${BOLD}${CYAN}------------------------------------------------------------${NC}"
echo -e "${BOLD}Step 1: Building nixnan.so and testManyFMTs${NC}"
echo -e "${BOLD}${CYAN}------------------------------------------------------------${NC}"
echo ""
echo -e "${BLUE}Running: make -f MakeTest${NC}"
echo ""

make -f MakeTest

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""

echo -e "${BOLD}${CYAN}------------------------------------------------------------${NC}"
echo -e "${BOLD}Step 2: Run Configuration${NC}"
echo -e "${BOLD}${CYAN}------------------------------------------------------------${NC}"
echo ""
echo "How would you like to run the test?"
echo ""
echo "  1) With HISTOGRAM=1 (collect binade histograms)"
echo "  2) Without HISTOGRAM (detect special values only)"
echo "  3) Without nixnan (baseline run)"
echo "  4) Run all three for comparison"
echo ""

read -p "Enter choice [1-4]: " choice

run_baseline() {
    echo ""
    echo -e "${BOLD}${CYAN}--- Running WITHOUT nixnan (baseline) ---${NC}"
    echo -e "${BLUE}Command: ./testManyFMTs${NC}"
    echo ""
    ./testManyFMTs
}

run_no_histogram() {
    echo ""
    echo -e "${BOLD}${CYAN}--- Running WITH nixnan (no histogram) ---${NC}"
    echo -e "${BLUE}Command: LD_PRELOAD=$NIXNAN_SO ./testManyFMTs${NC}"
    echo ""
    LD_PRELOAD="$NIXNAN_SO" ./testManyFMTs
}

run_histogram() {
    echo ""
    echo -e "${BOLD}${CYAN}--- Running WITH nixnan + HISTOGRAM=1 ---${NC}"
    echo -e "${BLUE}Command: HISTOGRAM=1 LD_PRELOAD=$NIXNAN_SO ./testManyFMTs${NC}"
    echo ""
    HISTOGRAM=1 LD_PRELOAD="$NIXNAN_SO" ./testManyFMTs
}

case $choice in
    1)
        run_histogram
        ;;
    2)
        run_no_histogram
        ;;
    3)
        run_baseline
        ;;
    4)
        run_baseline
        echo ""
        echo -e "${YELLOW}Press Enter to continue with nixnan (no histogram)...${NC}"
        read
        run_no_histogram
        echo ""
        echo -e "${YELLOW}Press Enter to continue with nixnan + HISTOGRAM=1...${NC}"
        read
        run_histogram
        ;;
    *)
        echo "Invalid choice. Running with HISTOGRAM=1 by default."
        run_histogram
        ;;
esac

echo ""
echo -e "${BOLD}${CYAN}============================================================${NC}"
echo -e "${BOLD}${CYAN}    Summary${NC}"
echo -e "${BOLD}${CYAN}============================================================${NC}"
echo ""
echo "nixnan instrumentation modes:"
echo ""
echo "  ${BOLD}LD_PRELOAD=nixnan.so${NC}"
echo "    - Instruments all GPU FP operations"
echo "    - Reports special values (NaN, Inf, denormals) if detected"
echo ""
echo "  ${BOLD}HISTOGRAM=1 LD_PRELOAD=nixnan.so${NC}"
echo "    - Same as above, plus:"
echo "    - Collects histograms of operand/result magnitudes"
echo "    - Shows distribution across binades (powers of 2)"
echo ""
echo "For more options, see the nixnan documentation."
echo ""
echo -e "${GREEN}${BOLD}Done!${NC}"
echo ""
