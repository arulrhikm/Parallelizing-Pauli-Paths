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

    // This stores the global constants
struct GlobalConstants
{
    int num_qubits;
    int num_words;
    int num_gates;
    Pauli *pauli_words;
    cuDoubleComplex *coeffs;
    GateType *gates;
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
cleanupAndTruncate(int max_weight, Pauli *pauli_word, cuDoubleComplex *phase)
{
    if (cuCabs(*phase) <= 1e-10) {
        *phase = make_cuDoubleComplex(0.0, 0.0);
    } else if (pauliWordWeight(pauli_word) > max_weight)
    {
        *phase = make_cuDoubleComplex(0.0, 0.0);
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

    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_qubits = cuPauliPropConst.num_qubits;

    if (word_idx >= cuPauliPropConst.num_words)
    {
        return;
    }

    // Each thread gets its own local copy of the Pauli word and coefficient
    Pauli *pauli_word = &(cuPauliPropConst.pauli_words[word_idx * num_qubits]);
    cuDoubleComplex *phase = &cuPauliPropConst.coeffs[word_idx];

    //printf("Start\n%c(%f,%f)\n", pauliToString(pauli_word[0]), phase.x, phase.y);
    // Apply gates in reverse order (Heisenberg picture)
    for (int gate_idx = cuPauliPropConst.num_gates - 1; gate_idx >= 0; --gate_idx)
    {
        const GateType gate = cuPauliPropConst.gates[gate_idx];
        const int2 changeQubit{0, 1};

        // Apply gate conjugation
        apply_gate_device(gate, changeQubit, pauli_word, phase);

        //printf("APPLY\n%c(%f,%f)\n", pauliToString(pauli_word[0]), phase.x, phase.y);
        //cleanup + truncation
        cleanupAndTruncate(max_weight, pauli_word, phase);
    }
    //printf("Trunc\n%c(%f,%f)\n", pauliToString(pauli_word[0]), phase.x, phase.y);

    //cuPauliPropConst.coeffs[word_idx] = phase;
    if (word_idx == 0) {
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
    : num_qubits(num_qubits), d_pauli_words(nullptr), d_coeffs(nullptr), d_gates(nullptr), d_result(nullptr)
{
    // Flatten initial observable to device
    flattenMapToDevice(init);

    cudaMalloc(&d_result, 2 * sizeof(double));
    
    // Allocate and copy gates to device
    if (!circuit.empty()) {
        num_gates = circuit.size();
        size_t gates_size = num_gates * sizeof(GateType);
        cudaMalloc(&d_gates, gates_size);
        //TODO: BIG TROUBLE HERE
        GateType *gateTypes = new GateType[num_gates];
        for (int i = 0; i < num_gates; ++i) {
            gateTypes[i] = circuit[i].type;
        }
        cudaMemcpy(d_gates, gateTypes, gates_size, cudaMemcpyHostToDevice);
        delete[] gateTypes;

        // Check for any CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during initialization: " << cudaGetErrorString(err) << std::endl;
            cleanup();
        }
    }

    GlobalConstants params;
    params.num_qubits = num_qubits;
    params.num_words = num_words;
    params.num_gates = num_gates;
    params.pauli_words = d_pauli_words;
    params.coeffs = (cuDoubleComplex *) d_coeffs;
    params.gates = d_gates;
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
    if (d_gates) {
        cudaFree(d_gates);
        d_gates = nullptr;
    }
    
    cudaDeviceSynchronize();
}

void PauliSimulatorGPU::flattenMapToDevice(const std::map<PauliWord, Complex> &obs)
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

Complex PauliSimulatorGPU::runPropagation(int max_weight)
{
    if (num_words == 0)
    {
        return Complex(0.0, 0.0);
    }

    // Configure kernel launch parameters
    int blockSize = 256;                                     // Threads per block
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