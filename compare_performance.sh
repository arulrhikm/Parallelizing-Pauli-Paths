#!/bin/bash

# Performance Comparison Script for Pauli Simulator
# Compares CPU vs GPU execution times for each test

echo "========================================"
echo "  PAULI SIMULATOR PERFORMANCE COMPARISON"
echo "========================================"
echo ""

# Check if pauli_sim exists
if [ ! -f "./pauli_sim" ]; then
    echo "Error: pauli_sim executable not found!"
    echo "Please run 'make all' first to build the simulator."
    exit 1
fi

# Determine which test to run
TEST_NUM=""
if [ "$1" != "" ]; then
    TEST_NUM="-t $1"
    echo "Running test #$1"
else
    echo "Running all tests"
fi
echo ""

# Run CPU version
echo "----------------------------------------"
echo "Running CPU version..."
echo "----------------------------------------"
CPU_START=$(date +%s.%N)
./pauli_sim -c $TEST_NUM > /tmp/cpu_output.txt 2>&1
CPU_END=$(date +%s.%N)
CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)

# Run GPU version
echo ""
echo "----------------------------------------"
echo "Running GPU version..."
echo "----------------------------------------"
GPU_START=$(date +%s.%N)
./pauli_sim $TEST_NUM > /tmp/gpu_output.txt 2>&1
GPU_END=$(date +%s.%N)
GPU_TIME=$(echo "$GPU_END - $GPU_START" | bc)

# Calculate overall speedup
SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)

# Display overall results
echo ""
echo "========================================"
echo "  OVERALL RESULTS"
echo "========================================"
echo ""
printf "Total CPU Time:    %.3f seconds\n" $CPU_TIME
printf "Total GPU Time:    %.3f seconds\n" $GPU_TIME
echo ""
printf "Overall Speedup:   %.2fx\n" $SPEEDUP
echo ""

# Show if GPU is faster or slower
if (( $(echo "$SPEEDUP > 1" | bc -l) )); then
    echo "✓ GPU is faster overall!"
elif (( $(echo "$SPEEDUP < 1" | bc -l) )); then
    echo "⚠ GPU is slower overall (overhead may dominate for small workloads)"
else
    echo "≈ Similar performance"
fi

# Parse and compare individual test timings
echo ""
echo "========================================"
echo "  PER-TEST BREAKDOWN"
echo "========================================"
echo ""

# Extract test timings from CPU output
CPU_TESTS=$(grep -E "Test #[0-9]+" /tmp/cpu_output.txt | grep -oP "Test #\K[0-9]+")
CPU_TIMES=$(grep -E "Computation time:" /tmp/cpu_output.txt | grep -oP "[0-9]+\.[0-9]+" || echo "")

# Extract test timings from GPU output
GPU_TESTS=$(grep -E "Test #[0-9]+" /tmp/gpu_output.txt | grep -oP "Test #\K[0-9]+")
GPU_TIMES=$(grep -E "Computation time:" /tmp/gpu_output.txt | grep -oP "[0-9]+\.[0-9]+" || echo "")

# Check if we have timing data
if [ -z "$CPU_TIMES" ] || [ -z "$GPU_TIMES" ]; then
    echo "Note: Individual test timings not found in output."
    echo "The simulator may not be printing 'Computation time:' for each test."
else
    # Convert to arrays
    CPU_TIMES_ARR=($CPU_TIMES)
    GPU_TIMES_ARR=($GPU_TIMES)
    CPU_TESTS_ARR=($CPU_TESTS)
    
    # Print header
    printf "%-10s %12s %12s %12s\n" "Test" "CPU (ms)" "GPU (ms)" "Speedup"
    printf "%-10s %12s %12s %12s\n" "----" "--------" "--------" "-------"
    
    # Compare each test
    for i in "${!CPU_TIMES_ARR[@]}"; do
        cpu_ms=${CPU_TIMES_ARR[$i]}
        gpu_ms=${GPU_TIMES_ARR[$i]}
        test_num=${CPU_TESTS_ARR[$i]}
        
        if [ -n "$gpu_ms" ] && (( $(echo "$gpu_ms > 0" | bc -l) )); then
            speedup=$(echo "scale=2; $cpu_ms / $gpu_ms" | bc)
            printf "%-10s %12.3f %12.3f %12.2fx\n" "#$test_num" "$cpu_ms" "$gpu_ms" "$speedup"
        fi
    done
fi

echo ""
echo "Full outputs saved to:"
echo "  CPU: /tmp/cpu_output.txt"
echo "  GPU: /tmp/gpu_output.txt"
echo ""
