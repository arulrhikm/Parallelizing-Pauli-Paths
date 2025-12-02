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

#define THREADS_PER_BLOCK 128
#define MAX_PAULI_WORDS (THREADS_PER_BLOCK * 10)
#define MAX_QUBITS 10
#define SHARED_BYTES_PER_BLOCK (MAX_PAULI_WORDS * MAX_QUBITS)
#define SCAN_BLOCK_DIM THREADS_PER_BLOCK
#include "exclusiveScan.cu_inl"

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
    double *gate_angles;
    double *result;
};

__constant__ GlobalConstants cuPauliPropConst;

__device__ __inline__ char
printPauliWords(int local_id, int num_qubits, int num_words, Pauli *pauli_words, cuDoubleComplex *coeffs)
{
    __syncthreads();
    if (local_id == 0) {
        printf("============\n");
        for (int i = 0; i < num_words; i++) {
            int g_i = i * num_qubits;
            printPauliWord(num_qubits, &pauli_words[g_i], coeffs[i]);
        }
        printf("============\n");
    }
    __syncthreads();
}

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

__device__ __inline__ bool
keep(int max_weight, Pauli *pauli_word, cuDoubleComplex &phase) {
    // for now duplicates just remain seperate
    return cuCabs(phase) > 1e-10 && pauliWordWeight(pauli_word) <= max_weight;
}

// len(flags) = SCAN_BLOCK_DIM
__device__ __inline__ void
createFlags(int local_id, uint16_t *flags, Pauli *pauli_words, cuDoubleComplex *coeffs, int max_weight, int start_idx)
{
    for (int i = local_id; i < SCAN_BLOCK_DIM - 1; i += THREADS_PER_BLOCK)
    {
        if (i + start_idx < MAX_PAULI_WORDS) {
            int g_idx = (i + start_idx) * cuPauliPropConst.num_qubits;
            flags[i] = (uint16_t) keep(max_weight, &pauli_words[g_idx], coeffs[i + start_idx]);
        } else {
            flags[i] = 0;
        }

    }
    // flags[SCAN_BLOCK_DIM - 1] = 0;
    // if (local_id == 0) {
    //     for (int i = 0; i < SCAN_BLOCK_DIM; i++)
    //         printf("flags[%d] = %d\n", i, flags[i]);
    // }
}


__device__ __inline__ int
organizeIdxs(int local_id, uint16_t *old_idxs, int old_start_idx, uint16_t *prefixSumOutput)
{
    for (int i = local_id; i < SCAN_BLOCK_DIM - 1; i += THREADS_PER_BLOCK)
    {
        if (prefixSumOutput[i] != prefixSumOutput[i + 1])
        {
            old_idxs[prefixSumOutput[i]] = old_start_idx + i;
        }
    }
    return prefixSumOutput[SCAN_BLOCK_DIM - 1];
}

__device__ __inline__ void
loadSharedMemeory(int local_id, int num_qubits, int new_start, uint16_t *old_idxs, int length, 
                  Pauli *pauli_words, cuDoubleComplex *coeffs)
{
    for (int i = local_id; i < length; i += THREADS_PER_BLOCK)
    {
        int new_idx = i + new_start;
        int old_idx = old_idxs[i];
        for (int j = 0; j < num_qubits; j++)
        {
            int g_new = new_idx * num_qubits;
            int g_old = old_idx * num_qubits;
            pauli_words[g_new + j] = pauli_words[g_old + j];
        }
        coeffs[new_idx] = coeffs[old_idx];
        if (old_idx != new_idx) {
            coeffs[old_idx] = make_cuDoubleComplex(0.0, 0.0);
        }   
        
    }
}

// REQUIREMENTS:
//  - Input array must have power-of-two length.
//  - Number of threads in the thread block must be the size of the array!
//  - SCAN_BLOCK_DIM is both the number of threads in the block (must be power of 2)
//         and the number of elements that will be scanned.
//          You should define this in your cudaRenderer.cu file,
//          based on your implementation.
//  - The parameter sScratch should be a pointer to an array with 2*SCAN_BLOCK_DIM elements
//  - The 3 arrays should be in shared memory.
__device__ __inline__ int
cleanup(int local_id, int num_qubits, int max_weight, Pauli *pauli_words, cuDoubleComplex *phases,
        uint16_t *prefixSumInput, uint16_t *prefixSumOutput, uint16_t *prefixSumScratch)
{
    __syncthreads();
    int num_words = 0;
    for (int seen_words = 0; seen_words < MAX_PAULI_WORDS; seen_words += SCAN_BLOCK_DIM - 1)
    {
        createFlags(local_id, prefixSumInput, pauli_words, phases, max_weight, seen_words);
        __syncthreads();
        sharedMemExclusiveScan(local_id, prefixSumInput, prefixSumOutput, prefixSumScratch, SCAN_BLOCK_DIM);
        __syncthreads();
        int newWords = organizeIdxs(local_id, prefixSumInput, seen_words, prefixSumOutput);
        // if (local_id == 0 && newWords > 0) {
        //     printf("HURRAY!!\n");
        // }
        __syncthreads();
        loadSharedMemeory(local_id, num_qubits, num_words, prefixSumInput, newWords, pauli_words, phases);
        __syncthreads();
        num_words += newWords;
    }
    return num_words;
}

__device__ __inline__ bool
countExpectation(Pauli *word, int num_qubits)
{
    for (int i = 0; i < num_qubits; i++)
    {
        if (word[i] == X || word[i] == Y)
            return false;
    }
    return true;
}

__device__ __inline__ double2 
computeExpecation(Pauli *pauli_words, cuDoubleComplex *coeffs, int num_words, int num_qubits) {
    cuDoubleComplex sum = make_cuDoubleComplex(0.0,0.0);
    for (int word_idx = 0; word_idx < num_words; ++word_idx)
    {
        //printf("coeff: (%f, %f)\n", coeffs[word_idx].x, coeffs[word_idx].y);
        if (countExpectation(&pauli_words[num_qubits * word_idx], num_qubits))
        {
            cuDoubleComplex coeff = coeffs[word_idx];
            sum = cuCadd(sum, coeff);
        }
    }
    return sum;
}


// Kernel implementation
__global__ void pauli_propagation_kernel(int max_weight)
{
    __shared__ Pauli pauli_words[SHARED_BYTES_PER_BLOCK];
    __shared__ cuDoubleComplex coeffs[MAX_PAULI_WORDS];
    __shared__ uint16_t prefixSumInput[SCAN_BLOCK_DIM];
    __shared__ uint16_t prefixSumOutput[SCAN_BLOCK_DIM];
    __shared__ uint16_t prefixSumScratch[SCAN_BLOCK_DIM * 2];

    int local_id = threadIdx.x;
    int num_qubits = cuPauliPropConst.num_qubits;
    int num_words = cuPauliPropConst.num_words;
    // must have half the number of bytes allowed incase every gate duplicates
    int max_words = MAX_PAULI_WORDS / 2;

    // Each thread gets its own local copy of the Pauli word and coefficient
    Pauli *extra_pauli_word = new Pauli[num_qubits];
    for (int word_idx = local_id; word_idx < MAX_PAULI_WORDS; word_idx++) {
        int g_word_idx = word_idx * num_qubits;
        for (int i = 0; i < num_qubits; i++)
        {
            pauli_words[g_word_idx + i] = cuPauliPropConst.pauli_words[g_word_idx + i];
        }
        coeffs[word_idx] = cuPauliPropConst.coeffs[word_idx];
    }
    __syncthreads();
    //printPauliWords(local_id, num_qubits, 2, pauli_words, coeffs);

    for (int gate_idx = cuPauliPropConst.num_gates - 1; gate_idx >= 0; --gate_idx)
    {
        const GateType gate_type = cuPauliPropConst.gate_types[gate_idx];
        int2 gate_qubits;
        gate_qubits.x = cuPauliPropConst.gate_qubits[2 * gate_idx];
        gate_qubits.y = cuPauliPropConst.gate_qubits[2 * gate_idx + 1];
        double angle = cuPauliPropConst.gate_angles[gate_idx];
        for (int i = local_id; i < num_words; i += THREADS_PER_BLOCK)
        {
            int g_i = i * num_qubits;
            int extra_i = i + max_words;
            int g_extra_i = extra_i * num_qubits;
            apply_gate_device(num_qubits, gate_type, gate_qubits, angle,
                              &pauli_words[g_i], coeffs[i],
                              &pauli_words[g_extra_i], coeffs[extra_i]);
        }

        num_words = cleanup(local_id, num_qubits, max_weight, pauli_words, coeffs,
                            prefixSumInput, prefixSumOutput, prefixSumScratch);


        //printPauliWords(local_id, num_qubits, 2, pauli_words, coeffs);
        if (local_id == 0 && num_words > max_words) {
            printf("STOP EVERYTHING\n");
            gate_idx = 0;
        }

        //printPauliWords(local_id, num_qubits, 2, pauli_words, coeffs);
    }

    if (local_id == 0)
    {
        double2 result = computeExpecation(pauli_words, coeffs, num_words, num_qubits);
        cuPauliPropConst.result[0] = result.x;
        cuPauliPropConst.result[1] = result.y;
    }
    // cuPauliPropConst.result[0] = 0.0;
    // cuPauliPropConst.result[1] = 0.0;
}

/**
 * HOST CODE
 * 
 */
PauliSimulatorGPU::PauliSimulatorGPU(int num_qubits,
                                     const std::map<PauliWord, Complex> &init,
                                     const std::vector<Gate> &circuit)
    : num_qubits(num_qubits), d_pauli_words(nullptr), d_coeffs(nullptr), 
    d_gate_types(nullptr), d_gate_qubits(nullptr), d_gate_angles(nullptr),
    d_result(nullptr)
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
    params.gate_angles = d_gate_angles;
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
    if (d_gate_angles) {
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
    if (num_words > MAX_PAULI_WORDS / 2) {
        std::cerr << "Can only have " << MAX_PAULI_WORDS / 2 
        << "pauli words" << std::endl;
    }
    
    // Allocate device memory for Pauli words (1 byte per qubit per word)
    size_t pauli_words_size = MAX_PAULI_WORDS * num_qubits * sizeof(Pauli);
    cudaMalloc(&d_pauli_words, pauli_words_size);
    
    // Allocate device memory for coefficients (2 doubles per word: real and imag)
    size_t coeffs_size = MAX_PAULI_WORDS * 2 * sizeof(double);
    cudaMalloc(&d_coeffs, coeffs_size);
    
    // Create host buffers for flattened data
    std::vector<Pauli> h_pauli_words(MAX_PAULI_WORDS * num_qubits);
    std::vector<double> h_coeffs(MAX_PAULI_WORDS * 2);

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
    for (; word_idx < MAX_PAULI_WORDS; ++word_idx)
    {
        for (int qubit = 0; qubit < num_qubits; ++qubit)
        {
            h_pauli_words[word_idx * num_qubits + qubit] = I;
        }
        h_coeffs[word_idx * 2] = 0.0;     // real part
        h_coeffs[word_idx * 2 + 1] = 0.0; // imaginary part
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
    if (!circuit.empty()) {
        num_gates = circuit.size();
        size_t gate_types_size = num_gates * sizeof(GateType);
        size_t gate_qubits_size = num_gates * sizeof(int) * 2;
        size_t gate_angles_size = num_gates * sizeof(double);
        GateType *gateTypes = new GateType[num_gates];
        int *gateQubits = new int[num_gates * 2];
        double *gateAngles = new double[num_gates];

        for (int i = 0; i < num_gates; ++i)
        {
            gateTypes[i] = circuit[i].type;
            gateQubits[2 * i] = circuit[i].qubits[0];
            gateQubits[2 * i + 1] = circuit[i].qubits.size() > 1 ? circuit[i].qubits[1] : 0;
            gateAngles[i] = circuit[i].angle;
        }

        cudaMalloc(&d_gate_types, gate_types_size);
        cudaMalloc(&d_gate_qubits, gate_qubits_size);
        cudaMalloc(&d_gate_angles, gate_angles_size);
        cudaMemcpy(d_gate_types, gateTypes, gate_types_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate_qubits, gateQubits, gate_qubits_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate_angles, gateAngles, gate_angles_size, cudaMemcpyHostToDevice);
        delete[] gateTypes;
        delete[] gateQubits;
        delete[] gateAngles;

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
    int numBlocks = 1; // Ceiling division
    if ((num_words + blockSize - 1) / blockSize > numBlocks)
    {
        std::cerr << "can only launch with 1 thread block" << std::endl;
        return Complex(0.0, 0.0);
    }

    // std::cout << "Launching GPU kernel with " << numBlocks << " blocks, "
    //           << blockSize << " threads per block" << std::endl;
    // std::cout << "Processing " << num_words << " Pauli words" << std::endl;

    // Launch the kernel
    pauli_propagation_kernel<<<numBlocks, blockSize>>>(max_weight);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "\033[31m" << "Kernel launch failed: " << cudaGetErrorString(err) << "\033[0m" << std::endl;
        return Complex(0.0, 0.0);
    }

    // Synchronize to wait for kernel completion
    cudaDeviceSynchronize();

    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "\033[31m" << "Kernel execution failed: " << cudaGetErrorString(err) << "\033[0m" << std::endl;
        return Complex(0.0, 0.0);
    }

    // std::cout << "GPU kernel completed successfully"<< std::endl;

    // TODO: You'll need to add code here to:
    // 1. Copy results back from device to host
    // 2. Sum up the expectation values from all Pauli words
    // 3. Implement weight-based truncation logic
    double result[2];
    cudaMemcpy(result, d_result, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    // For now, return a placeholder
    return Complex(result[0], result[1]);
}