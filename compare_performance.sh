#!/bin/bash

# Performance Comparison Script for Pauli Simulator
# Compares CPU vs GPU execution times

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

# Calculate speedup
SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)

# Display results
echo ""
echo "========================================"
echo "  RESULTS"
echo "========================================"
echo ""
printf "CPU Time:    %.3f seconds\n" $CPU_TIME
printf "GPU Time:    %.3f seconds\n" $GPU_TIME
echo ""
printf "Speedup:     %.2fx\n" $SPEEDUP
echo ""

# Show if GPU is faster or slower
if (( $(echo "$SPEEDUP > 1" | bc -l) )); then
    echo "✓ GPU is faster!"
elif (( $(echo "$SPEEDUP < 1" | bc -l) )); then
    echo "⚠ GPU is slower (overhead may dominate for small workloads)"
else
    echo "≈ Similar performance"
fi

echo ""
echo "Full outputs saved to:"
echo "  CPU: /tmp/cpu_output.txt"
echo "  GPU: /tmp/gpu_output.txt"
echo ""
