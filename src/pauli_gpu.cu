/* 
THINGS TO DO:
GET GATE QUBITS

*/

// PauliSimulatorGPU.cu
#include "pauli.h"
#include "pauli_gpu.h"
#include "gates.cu_inl"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <cstring>

#define THREADS_PER_BLOCK 256
#define MAX_PAULI_WORDS_PER_BLOCK THREADS_PER_BLOCK
#define MAX_QUBITS 10

// This stores the global constants
struct GlobalConstants
{
    int num_qubits;
    int num_words;
    int num_gates;
    Pauli *pauli_words;
    cuDoubleComplex *coeffs;
    GateType *gate_types;
    int *gate_qubits;
    double *result;
};

__constant__ GlobalConstants cuPauliPropConst;

__device__ __inline__ int
pauliWordWeight(Pauli *pauli_word)
{
    int w = 0;
    for (int i = 0; i < cuPauliPropConst.num_qubits; ++i)
    {
        if (pauli_word[i] != I)
            ++w;
    }
    return w;
}

__device__ __inline__ void
cleanupAndTruncate(int max_weight, Pauli *pauli_word, cuDoubleComplex &phase)
{   
    //for now duplicates just remain seperate
    if (cuCabs(phase) <= 1e-10) {
        phase = make_cuDoubleComplex(0.0, 0.0);
    } else if (pauliWordWeight(pauli_word) > max_weight)
    {
        phase = make_cuDoubleComplex(0.0, 0.0);
    }
}

__device__ __inline__ bool
countExpectation(Pauli *word)
{
    for (int i = 0; i < cuPauliPropConst.num_qubits; i++)
    {
        if (word[i] == X || word[i] == Y)
            return false;
    }
    return true;
}

__device__ __inline__ double2 
computeExpecation() {
    cuDoubleComplex sum = make_cuDoubleComplex(0.0,0.0);
    for (int word_idx = 0; word_idx < cuPauliPropConst.num_words; ++word_idx)
    {
        if (countExpectation(&cuPauliPropConst.pauli_words[cuPauliPropConst.num_qubits * word_idx]))
        {
            cuDoubleComplex coeff = cuPauliPropConst.coeffs[word_idx];
            sum = cuCadd(sum, cuCmul(coeff, coeff));
        }
    }
    return sum;
}

__device__ __inline__ char
pauliToString(Pauli p)
{
    switch (p) {
        case I:
            return 'I';
        case X:
            return 'X';
        case Y:
            return 'Y';
        case Z:
            return 'Z';
        default:
            return '?';
    }
}


// Kernel implementation
__global__ void pauli_propagation_kernel(int max_weight)
{
    // __shared__ Pauli pauli_words[MAX_PAULI_WORDS * cuPauliPropConst.num_qubits];
    // __shared__ cuDoubleComplex coeffs[MAX_PAULI_WORDS];
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_qubits = cuPauliPropConst.num_qubits;

    if (word_idx >= cuPauliPropConst.num_words)
    {
        return;
    }

    // Each thread gets its own local copy of the Pauli word and coefficient
    Pauli *pauli_word = new Pauli[num_qubits];
    for (int i = 0; i < num_qubits; i++) {
        pauli_word[i] = cuPauliPropConst.pauli_words[word_idx * num_qubits + i];
    }
    cuDoubleComplex phase = cuPauliPropConst.coeffs[word_idx];

    for (int gate_idx = cuPauliPropConst.num_gates - 1; gate_idx >= 0; --gate_idx)
    {
        cuDoubleComplex old_phase = phase;
        const GateType gate_type = cuPauliPropConst.gate_types[gate_idx];
        int2 gate_qubits;
        gate_qubits.x = cuPauliPropConst.gate_qubits[2 * gate_idx];
        gate_qubits.y = cuPauliPropConst.gate_qubits[2 * gate_idx + 1];
        // Apply gate conjugation
        apply_gate_device(gate_type, gate_qubits, pauli_word, phase);
        phase = cuCmul(old_phase, phase);
        // cleanup + truncation
        cleanupAndTruncate(max_weight, pauli_word, phase);
    }
    for (int i = 0; i < num_qubits; i++)
    {
        cuPauliPropConst.pauli_words[word_idx * num_qubits + i] = pauli_word[i];
    }
    cuPauliPropConst.coeffs[word_idx] = phase;

    if (word_idx == 0)
    {
        double2 result = computeExpecation();
        cuPauliPropConst.result[0] = result.x;
        cuPauliPropConst.result[1] = result.y;
    }
}

/**
 * HOST CODE
 * 
 */
PauliSimulatorGPU::PauliSimulatorGPU(int num_qubits,
                                     const std::map<PauliWord, Complex> &init,
                                     const std::vector<Gate> &circuit)
    : num_qubits(num_qubits), d_pauli_words(nullptr), d_coeffs(nullptr), 
    d_gate_types(nullptr), d_gate_qubits(nullptr), d_result(nullptr)
{
    // Flatten initial observable to device
    allocatePauliWords(init);
    allocateGates(circuit);

    cudaMalloc(&d_result, 2 * sizeof(double));

    GlobalConstants params;
    params.num_qubits = num_qubits;
    params.num_words = num_words;
    params.num_gates = num_gates;
    params.pauli_words = d_pauli_words;
    params.coeffs = (cuDoubleComplex *) d_coeffs;
    params.gate_types = d_gate_types;
    params.gate_qubits = d_gate_qubits;
    params.result = d_result;
    cudaMemcpyToSymbol(cuPauliPropConst, &params, sizeof(GlobalConstants));
}

PauliSimulatorGPU::~PauliSimulatorGPU()
{
    cleanup();
}

void PauliSimulatorGPU::cleanup()
{
    if (d_pauli_words) {
        cudaFree(d_pauli_words);
        d_pauli_words = nullptr;
    }
    if (d_coeffs) {
        cudaFree(d_coeffs);
        d_coeffs = nullptr;
    }
    if (d_gate_types) {
        cudaFree(d_gate_types);
        d_gate_types = nullptr;
    }
    if (d_gate_qubits) {
        cudaFree(d_gate_qubits);
        d_gate_qubits = nullptr;
    }
    if (d_result) {
        cudaFree(d_result);
        d_result = nullptr;
    }
    
    cudaDeviceSynchronize();
}

void PauliSimulatorGPU::allocatePauliWords(const std::map<PauliWord, Complex> &obs)
{
    num_words = obs.size();
    if (num_words == 0) {
        d_pauli_words = nullptr;
        d_coeffs = nullptr;
        return;
    }
    
    // Allocate device memory for Pauli words (1 byte per qubit per word)
    std::cout << "NUM QUBITS: " << num_qubits << "\n";
    size_t pauli_words_size = num_words * num_qubits * sizeof(Pauli);
    cudaMalloc(&d_pauli_words, pauli_words_size);
    
    // Allocate device memory for coefficients (2 doubles per word: real and imag)
    size_t coeffs_size = num_words * 2 * sizeof(double);
    cudaMalloc(&d_coeffs, coeffs_size);
    
    // Create host buffers for flattened data
    std::vector<Pauli> h_pauli_words(num_words * num_qubits);
    std::vector<double> h_coeffs(num_words * 2);
    
    // Flatten the observable map
    int word_idx = 0;
    for (const auto &[pw, coeff] : obs) {
        // Copy Pauli operators for this word
        for (int qubit = 0; qubit < num_qubits; ++qubit) {
            h_pauli_words[word_idx * num_qubits + qubit] = pw.ops[qubit];
        }
        
        // Copy coefficient (real and imaginary parts)
        h_coeffs[word_idx * 2] = coeff.real();     // real part
        h_coeffs[word_idx * 2 + 1] = coeff.imag(); // imaginary part
        
        word_idx++;
    }
    
    // Copy data to device
    cudaMemcpy(d_pauli_words, h_pauli_words.data(), pauli_words_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, h_coeffs.data(), coeffs_size, cudaMemcpyHostToDevice);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in flattenToDevice: " << cudaGetErrorString(err) << std::endl;
        cleanup();
    }
}

void PauliSimulatorGPU::allocateGates(const std::vector<Gate> &circuit) {
    // Allocate and copy gates to device
    if (!circuit.empty())
    {
        num_gates = circuit.size();
        size_t gate_types_size = num_gates * sizeof(GateType);
        size_t gate_qubits_size = num_gates * sizeof(int) * 2;
        GateType *gateTypes = new GateType[num_gates];
        int *gateQubits = new int[num_gates * 2];

        for (int i = 0; i < num_gates; ++i)
        {
            gateTypes[i] = circuit[i].type;
            gateQubits[2 * i] = circuit[i].qubits[0];
            gateQubits[2 * i + 1] = circuit[i].qubits.size() > 1 ? circuit[i].qubits[1] : 0;
        }

        cudaMalloc(&d_gate_types, gate_types_size);
        cudaMalloc(&d_gate_qubits, gate_qubits_size);
        cudaMemcpy(d_gate_types, gateTypes, gate_types_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate_qubits, gateQubits, gate_qubits_size, cudaMemcpyHostToDevice);
        delete[] gateTypes;
        delete [] gateQubits;

        // Check for any CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error during initialization: " << cudaGetErrorString(err) << std::endl;
            cleanup();
        }
    }
}

Complex PauliSimulatorGPU::runPropagation(int max_weight)
{
    if (num_words == 0)
    {
        return Complex(0.0, 0.0);
    }

    // Configure kernel launch parameters
    int blockSize = THREADS_PER_BLOCK;                       // Threads per block
    int numBlocks = (num_words + blockSize - 1) / blockSize; // Ceiling division

    std::cout << "Launching GPU kernel with " << numBlocks << " blocks, "
              << blockSize << " threads per block" << std::endl;
    std::cout << "Processing " << num_words << " Pauli words" << std::endl;

    // Launch the kernel
    pauli_propagation_kernel<<<numBlocks, blockSize>>>(max_weight);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return Complex(0.0, 0.0);
    }

    // Synchronize to wait for kernel completion
    cudaDeviceSynchronize();

    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return Complex(0.0, 0.0);
    }

    std::cout << "GPU kernel completed successfully" << std::endl;

    // TODO: You'll need to add code here to:
    // 1. Copy results back from device to host
    // 2. Sum up the expectation values from all Pauli words
    // 3. Implement weight-based truncation logic
    double result[2];
    cudaMemcpy(result, d_result, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    // For now, return a placeholder
    return Complex(result[0], result[1]);
}