# Try to find compatible GCC version for CUDA
# CUDA 11.7 supports up to GCC 11
GCC_11 := $(shell which g++-11 2>/dev/null)
GCC_10 := $(shell which g++-10 2>/dev/null)
GCC_9 := $(shell which g++-9 2>/dev/null)

ifneq ($(GCC_11),)
    CXX = g++-11
    HOST_COMPILER = -ccbin g++-11
else ifneq ($(GCC_10),)
    CXX = g++-10
    HOST_COMPILER = -ccbin g++-10
else ifneq ($(GCC_9),)
    CXX = g++-9
    HOST_COMPILER = -ccbin g++-9
else
    CXX = g++
    HOST_COMPILER =
endif

NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -O2
NVCCFLAGS = -std=c++17 -O2 $(HOST_COMPILER)

OBJ_DIR = build
OBJ_DIR_CPU = build_cpu

# Source files in current directory (since Makefile is in src/)
SOURCES = pauli.cpp tests.cpp main.cpp
OBJECTS_GPU = $(OBJ_DIR)/pauli.o $(OBJ_DIR)/tests.o $(OBJ_DIR)/main.o
OBJECTS_CPU = $(OBJ_DIR_CPU)/pauli.o $(OBJ_DIR_CPU)/tests.o $(OBJ_DIR_CPU)/main.o

# GPU sources
GPU_SOURCES = pauli_gpu.cu
GPU_OBJECTS = $(OBJ_DIR)/pauli_gpu.o

# Output in parent directory
TARGET_GPU = ../pauli_propagation_gpu.exe
TARGET_CPU = ../pauli_propagation_cpu.exe

# Build both by default
all: gpu cpu

# GPU version
gpu: $(OBJECTS_GPU) $(GPU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET_GPU) $^
	@echo "✓ GPU version built: $(TARGET_GPU)"

# CPU-only version (no CUDA)
cpu: $(OBJECTS_CPU)
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -o $(TARGET_CPU) $^
	@echo "✓ CPU version built: $(TARGET_CPU)"

# Compile C++ files for GPU build
$(OBJ_DIR)/%.o: %.cpp
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C++ files for CPU build
$(OBJ_DIR_CPU)/%.o: %.cpp
	mkdir -p $(OBJ_DIR_CPU)
	$(CXX) $(CXXFLAGS) -DCPU_ONLY -c $< -o $@

# Compile CUDA files
$(OBJ_DIR)/%.o: %.cu
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(OBJ_DIR_CPU) $(TARGET_GPU) $(TARGET_CPU)

.PHONY: all clean gpu cpu
