// PauliSimulatorGPU.cu
#include "puali.h"
#include "puali_gpu.h"
#include "gates.cu_inl"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
    // This stores the global constants
    struct GlobalConstants
{
    int num_qubits;
    int num_words;
    uint8_t *d_pauli_words;
    double *d_coeffs;
    uint8_t *d_gates;
    int *d_num_qubits;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuPauliPropConst;

// Kernel implementation
__global__ void pauli_propagation_kernel()
{

    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_qubits = cuPauliPropConst.num_qubits;

    if (word_idx >= cuPauliPropConst.num_words)
    {
        return;
    }

    // Each thread gets its own local copy of the Pauli word and coefficient
    Pauli *local_pauli_word = &(cuPauliPropConst.pauli_words[word_idx * num_qubits]);
    double2 local_phase = *(double2 *)cuPauliPropConst.coeffs[2 * word_idx];

    // Apply gates in reverse order (Heisenberg picture)
    for (int gate_idx = num_gates - 1; gate_idx >= 0; --gate_idx)
    {
        const GateType gate = cuPauliPropConst.gates[gate_idx];
        const int2 changeQubit{0, 1};

        // Apply gate conjugation
        apply_gate_device(gateType, changeQubit, local_pauli_word, &local_phase);
    }

    // Compute expectation value for this term
    /* double term_expectation = compute_expectation_device(local_pauli_word, local_phase_real, local_phase_imag, num_qubits); */

/*     // Apply coefficient to expectation
    double real_contribution = coeff_real * term_expectation;
    double imag_contribution = coeff_imag * term_expectation;

    // Atomic add to final expectation (real and imaginary parts)
    atomicAdd(&final_expectation[0], real_contribution);
    atomicAdd(&final_expectation[1], imag_contribution); */

    // Clean up local memory
    delete[] local_pauli_word;
}

/**
 * HOST CODE
 * 
 */
PauliSimulatorGPU::PauliSimulatorGPU(int num_qubits, 
                                     const std::map<PauliWord, Complex> &init,
                                     const std::vector<GateType> &circuit)
    : num_qubits(num_qubits), d_pauli_words(nullptr), d_coeffs(nullptr), d_gates(nullptr)
{
    // Flatten initial observable to device
    flattenMapToDevice(init);
    
    // Allocate and copy gates to device
    if (!circuit.empty()) {
        size_t gates_size = circuit.size() * sizeof(GateType);
        cudaMalloc(&d_gates, gates_size);
        cudaMemcpy(d_gates, circuit.data(), gates_size, cudaMemcpyHostToDevice);
        
        // Check for any CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during initialization: " << cudaGetErrorString(err) << std::endl;
            cleanup();
        }
    }

    GlobalConstants params;
    params.num_qubits = num_qubits;
    params.num_words = init.size();
    params.d_pauli_words = d_pauli_words;
    params.d_coeffs = d_coeffs;
    params.d_gates = d_gates;
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
    size_t pauli_words_size = num_words * num_qubits * sizeof(Pauli);
    cudaMalloc(&d_pauli_words, pauli_words_size);
    
    // Allocate device memory for coefficients (2 doubles per word: real and imag)
    size_t coeffs_size = num_words * 2 * sizeof(double);
    cudaMalloc(&d_coeffs, coeffs_size);
    
    // Create host buffers for flattened data
    std::vector<uint8_t> h_pauli_words(num_words * num_qubits);
    std::vector<double> h_coeffs(num_words * 2);
    
    // Flatten the observable map
    int word_idx = 0;
    for (const auto &[pw, coeff] : obs) {
        // Copy Pauli operators for this word
        for (int qubit = 0; qubit < num_qubits; ++qubit) {
            h_pauli_words[word_idx * num_qubits + qubit] = static_cast<uint8_t>(pw.ops[qubit]);
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
    // TODO: Implement the actual GPU propagation kernel
    // This would involve:
    // 1. Launching CUDA kernels to apply gate conjugations
    // 2. Managing memory for intermediate results
    // 3. Handling weight-based truncation on GPU
    // 4. Computing final expectation value
    
    std::cout << "GPU propagation not yet implemented - returning 0+0i" << std::endl;
    std::cout << "Parameters: num_qubits=" << num_qubits << ", max_weight=" << max_weight << std::endl;
    
    // For now, return a placeholder
    return Complex(0.0, 0.0);
}