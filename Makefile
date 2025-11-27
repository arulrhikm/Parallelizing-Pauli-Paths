# Makefile for Pauli Simulator with CUDA
CXX = g++
NVCC = nvcc
TARGET = pauli_sim

# CUDA paths and flags
CUDA_PATH ?= /usr/local/cuda-11.7
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart
NVCCFLAGS = -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/g++-11
CXXFLAGS = -std=c++17 -Wall -O2 -I.

# Source directories and files
SRC_DIR = src
OBJ_DIR = build

# All source files
CU_SOURCES = $(SRC_DIR)/pauli_gpu.cu
CPP_SOURCES = $(SRC_DIR)/pauli.cpp $(SRC_DIR)/main.cpp $(SRC_DIR)/tests.cpp
HEADERS = $(SRC_DIR)/pauli.h $(SRC_DIR)/puali_gpu.h

# Object files
CU_OBJS = $(OBJ_DIR)/pauli_gpu.o
CPP_OBJS = $(OBJ_DIR)/pauli.o $(OBJ_DIR)/main.o $(OBJ_DIR)/tests.o
OBJECTS = $(CU_OBJS) $(CPP_OBJS)

all: dirs $(TARGET)

dirs:
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

# Compile CUDA files
$(OBJ_DIR)/pauli_gpu.o: $(SRC_DIR)/pauli_gpu.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -I. -I$(SRC_DIR) -c $< -o $@

# Compile C++ files
$(OBJ_DIR)/pauli.o: $(SRC_DIR)/pauli.cpp $(SRC_DIR)/pauli.h
	$(CXX) $(CXXFLAGS) -I. -I$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I. -I$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/tests.o: $(SRC_DIR)/tests.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -I. -I$(SRC_DIR) -c $< -o $@

# Test specific targets
test: $(TARGET)
	./$(TARGET) --test  # or whatever your test command is

run: $(TARGET)
	./$(TARGET)

# Quick compilation test for CUDA only
compile-test: dirs
	$(NVCC) $(NVCCFLAGS) -I. -I$(SRC_DIR) -c $(SRC_DIR)/pauli_gpu.cu -o $(OBJ_DIR)/pauli_gpu_compile_test.o
	@echo "CUDA compilation successful!"

# Check file existence
check-files:
	@echo "Checking for source files..."
	@ls -la $(SRC_DIR)/pauli_gpu.cu 2>/dev/null && echo "✓ pauli_gpu.cu found" || echo "✗ pauli_gpu.cu missing"
	@ls -la $(SRC_DIR)/pauli.h 2>/dev/null && echo "✓ pauli.h found" || echo "✗ pauli.h missing"
	@ls -la $(SRC_DIR)/puali_gpu.h 2>/dev/null && echo "✓ puali_gpu.h found" || echo "✗ puali_gpu.h missing"
	@ls -la $(SRC_DIR)/main.cpp 2>/dev/null && echo "✓ main.cpp found" || echo "✗ main.cpp missing"
	@ls -la $(SRC_DIR)/tests.cpp 2>/dev/null && echo "✓ tests.cpp found" || echo "✗ tests.cpp missing"
	@ls -la $(SRC_DIR)/pauli.cpp 2>/dev/null && echo "✓ pauli.cpp found" || echo "✗ pauli.cpp missing"

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean dirs compile-test check-files test run
