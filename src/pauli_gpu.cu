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
#define MAX_BLOCKS 400
#define MAX_PAULI_WORDS (THREADS_PER_BLOCK * 2)
#define MAX_QUBITS 10
#define SHARED_BYTES_PER_BLOCK (MAX_PAULI_WORDS * MAX_QUBITS)
#define SCAN_BLOCK_DIM THREADS_PER_BLOCK
#include "exclusiveScan.cu_inl"

using namespace std;

// This stores the global constants
struct GlobalConstants
{
    int num_qubits;
    int num_words;
    int num_gates;
    int *gate_idx;
    GateType *gate_types;
    int *gate_qubits;
    double *gate_angles;
    Pauli *pauli_words;
    cuDoubleComplex *coeffs;
    int *prev_words;
    int *prev_blocks;
    double *result;
};

__constant__ GlobalConstants cuConstants;

__device__ __inline__ char
printPauliWords(int local_id, int num_qubits, int num_words, Pauli *pauli_words, cuDoubleComplex *coeffs)
{
    __syncthreads();
    if (local_id == 0)
    {
        printf("============\n");
        for (int i = 0; i < num_words; i++)
        {
            int pi = i * num_qubits;
            printPauliWord(num_qubits, &pauli_words[pi], coeffs[i]);
        }
        printf("============\n");
    }
    __syncthreads();
}

__device__ __inline__ int
pauliWordWeight(Pauli *pauli_word)
{
    int w = 0;
    for (int i = 0; i < cuConstants.num_qubits; ++i)
    {
        if (pauli_word[i] != I)
            ++w;
    }
    return w;
}

__device__ __inline__ bool
keep(int max_weight, Pauli *pauli_word, cuDoubleComplex &phase)
{
    // for now duplicates just remain seperate
    return cuCabs(phase) > 1e-10 && pauliWordWeight(pauli_word) <= max_weight;
}

// len(flags) = SCAN_BLOCK_DIM
__device__ __inline__ void
createFlags(int local_id, uint16_t *flags, Pauli *pauli_words, cuDoubleComplex *coeffs, int max_weight, int start_idx)
{
    for (int i = local_id; i < SCAN_BLOCK_DIM - 1; i += THREADS_PER_BLOCK)
    {
        if (i + start_idx < MAX_PAULI_WORDS)
        {
            int pidx = (i + start_idx) * cuConstants.num_qubits;
            flags[i] = (uint16_t)keep(max_weight, &pauli_words[pidx], coeffs[i + start_idx]);
        }
        else
        {
            flags[i] = 0;
        }
    }
    // if (blockIdx.x == 0 && local_id == 0)
    // {
    //     printf("start_idx = %d\n", start_idx);
    //     printf("FLAGS - [");
    //     for (int i = 0; i < SCAN_BLOCK_DIM; i++)
    //     {
    //         printf("%d", flags[i]);
    //         if (i != SCAN_BLOCK_DIM - 1) {
    //             printf(", ");
    //         }
    //     }
    //     printf("]\n");
    // }
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
    int new_idx = local_id + new_start;
    int old_idx = old_idxs[local_id];
    cuDoubleComplex oldPhase;
    if (local_id < length)
    {
        oldPhase = coeffs[old_idx];
        coeffs[old_idx] = make_cuDoubleComplex(0.0, 0.0);;
    }
    __syncthreads();
    if (local_id < length)
    {
        for (int q = 0; q < num_qubits; q++)
        {
            int pnew = new_idx * num_qubits;
            int pold = old_idx * num_qubits;
            pauli_words[pnew + q] = pauli_words[pold + q];
        }
        coeffs[new_idx] = oldPhase;
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
    // would deal with duplicates here
    
    int num_words = 0;
    for (int seen_words = 0; seen_words < MAX_PAULI_WORDS; seen_words += SCAN_BLOCK_DIM - 1)
    {
        createFlags(local_id, prefixSumInput, pauli_words, phases, max_weight, seen_words);
        __syncthreads();
        sharedMemExclusiveScan(local_id, prefixSumInput, prefixSumOutput, prefixSumScratch, SCAN_BLOCK_DIM);
        __syncthreads();
        int newWords = organizeIdxs(local_id, prefixSumInput, seen_words, prefixSumOutput);
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
    // for (int q = 0; q < num_qubits; q++)
    // {
    //     if (word[q] == X || word[q] == Y)
    //         return false;
    // }
    return true;
}

__device__ __inline__ double2
computeExpecation(Pauli *pauli_words, cuDoubleComplex *coeffs, int num_words, int num_qubits)
{
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int word_idx = 0; word_idx < num_words; ++word_idx)
    {
        // printf("coeff: (%f, %f)\n", coeffs[word_idx].x, coeffs[word_idx].y);
        if (countExpectation(&pauli_words[num_qubits * word_idx], num_qubits))
        {
            cuDoubleComplex coeff = coeffs[word_idx];
            sum = cuCadd(sum, coeff);
        }
    }
    return sum;
}

__device__ __inline__ int
calculate_global_idx(int prev_idx, int &current_section, int &section_end, int total_sections, int *prev_words)
{
    // 2. Prefix sums over counts to locate which old section this falls in
    int index = prev_idx + 1;
    while (index >= section_end)
    {
        current_section++;
        if (current_section == total_sections)
        {
            return -1;
        }
        int g_start = current_section * MAX_PAULI_WORDS;
        index = g_start + (index - section_end);
        section_end = g_start + prev_words[current_section];
    }
    // 4. Actual index inside the original element array
    return index;
}

__device__ __inline__ void
write_global(int block_id, int local_id, int num_words, int num_qubits, 
             Pauli *pauli_words, cuDoubleComplex *coeffs)
{
    int global_start = block_id * MAX_PAULI_WORDS;
    int pglobal_start = global_start * num_qubits;
    for (int i = local_id; i < num_words; i += THREADS_PER_BLOCK)
    {
        int p_i = i * num_qubits;
        for (int q = 0; q < num_qubits; q++) {
            cuConstants.pauli_words[pglobal_start + p_i + q] = pauli_words[p_i + q];
        }
        cuConstants.coeffs[global_start + i] = coeffs[i];
    }
}
// Kernel implementation
__global__ void pauli_propagation_kernel(int max_weight)
{
    __shared__ Pauli pauli_words[SHARED_BYTES_PER_BLOCK];
    __shared__ cuDoubleComplex coeffs[MAX_PAULI_WORDS];
    __shared__ uint16_t prefixSumInput[SCAN_BLOCK_DIM];
    __shared__ uint16_t prefixSumOutput[SCAN_BLOCK_DIM];
    __shared__ uint16_t prefixSumScratch[SCAN_BLOCK_DIM * 2];

    const auto block_id = blockIdx.x;
    const auto local_id = threadIdx.x;
    const auto global_id = block_id * blockDim.x + local_id;

    const auto num_qubits = cuConstants.num_qubits;
    int num_words = MAX_PAULI_WORDS / 2;
    // must have half the number of bytes allowed incase every gate duplicates
    const auto max_words = MAX_PAULI_WORDS / 2;

    // Each thread gets its own local copy of the Pauli word and coefficient
    int current_section = 0;
    int section_end = cuConstants.prev_words[0];
    int total_sections = *cuConstants.prev_blocks;
    int global_idx = global_id - 1;
    int word_idx = local_id;
    for (; word_idx < max_words; word_idx += 1)
    {
        global_idx = calculate_global_idx(global_idx, current_section, section_end,
                                            total_sections, cuConstants.prev_words);
        if (global_idx < 0)
            break;
        int g_pword_idx = global_idx * num_qubits;
        int pword_idx = word_idx * num_qubits;
        for (int i = 0; i < num_qubits; i++)
        {
            pauli_words[pword_idx + i] = cuConstants.pauli_words[g_pword_idx + i];
        }
        coeffs[word_idx] = cuConstants.coeffs[global_idx];
    }    
    for (; word_idx < MAX_PAULI_WORDS; word_idx += THREADS_PER_BLOCK)
    {
        int pword_idx = word_idx * num_qubits;
        for (int i = 0; i < num_qubits; i++)
        {
            pauli_words[pword_idx + i] = I;
        }
        coeffs[word_idx] = make_cuDoubleComplex(0.0, 0.0);
    }
    __syncthreads();
    for (int gate_idx = *cuConstants.gate_idx; gate_idx >= 0; --gate_idx)
    {
        const GateType gate_type = cuConstants.gate_types[gate_idx];
        int2 gate_qubits;
        gate_qubits.x = cuConstants.gate_qubits[2 * gate_idx];
        gate_qubits.y = cuConstants.gate_qubits[2 * gate_idx + 1];
        double angle = cuConstants.gate_angles[gate_idx];
        // if (gate_idx <= 76 && local_id == 0)
        // {
        //     printf("Have %d num words\n", num_words);
        // }
        for (int i = local_id; i < num_words; i += THREADS_PER_BLOCK)
        {
            int pi = i * num_qubits;
            int extra_i = i + max_words;
            int pextra_i = extra_i * num_qubits;
            apply_gate_device(num_qubits, gate_type, gate_qubits, angle,
                              &pauli_words[pi], coeffs[i],
                              &pauli_words[pextra_i], coeffs[extra_i]);
        }
        num_words = cleanup(local_id, num_qubits, max_weight, pauli_words, coeffs,
                            prefixSumInput, prefixSumOutput, prefixSumScratch);
        bool exit = gridDim.x == 1
                        ? num_words > max_words
                        : (gate_type == RX || gate_type == RY || gate_type == RZ);
        if (gate_idx != 0 && exit)
        {
            write_global(block_id, local_id, num_words, num_qubits, pauli_words, coeffs);
            if (local_id == 0)
            {
                cuConstants.prev_words[blockIdx.x] = num_words;
            }
            if (global_id == 0)
            {
                *cuConstants.prev_blocks = gridDim.x;
                *cuConstants.gate_idx = gate_idx - 1;
            }
            return;
        } 
        // if (local_id == 0 && !exit) {
        //     printf("Gate %d - Num Words: %d\n", gate_idx, num_words);
        // }           
    }

    // Write final result to global memory (no cout in kernel!)
    if (local_id == 0)
    {
        double2 result = computeExpecation(pauli_words, coeffs, num_words, num_qubits);
        cuConstants.result[2 * blockIdx.x] = result.x;
        cuConstants.result[2 * blockIdx.x + 1] = result.y;
    }

    // Signal completion
    if (global_id == 0)
    {
        *cuConstants.gate_idx = -1;
    }
}

/*
 * =================================================================
 * ======================   HOST CODE   ============================
 * =================================================================
 */

bool checkCudaError(const string &place)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "\033[31m" << "CUDA ERROR: " << cudaGetErrorString(err) << " during " << place << "\033[0m" << endl;
        return true;
    }
    return false;
}

PauliSimulatorGPU::PauliSimulatorGPU(int num_qubits,
                                     const map<PauliWord, Complex> &init,
                                     const vector<Gate> &circuit)
    : num_qubits(num_qubits),
      d_pauli_words(nullptr), d_coeffs(nullptr), d_prev_words(nullptr), d_prev_blocks(nullptr),
      d_gate_types(nullptr), d_gate_qubits(nullptr), d_gate_angles(nullptr), d_gate_idx(nullptr),
      d_result(nullptr)
{
    cout << "[GPU] Initializing simulator for " << num_qubits << " qubits" << endl;
    cout << "[GPU] Initial observable has " << init.size() << " Pauli words" << endl;

    if (num_qubits > MAX_QUBITS)
    {
        cerr << "TRYING TO SIMULATE " << num_qubits << " QUBITS BUT CAN ONLY SIMULATE " << MAX_QUBITS << " QUBITS\n";
        return;
    }
    cout << "[GPU] Allocating Pauli words..." << endl;
    if (!allocatePauliWords(init))
    {
        cout << "[GPU] Failed to allocate Pauli words!" << endl;
        return;
    };
    cout << "[GPU] Allocating gates..." << endl;
    allocateGates(circuit);

    cout << "[GPU] Allocating result buffer..." << endl;
    double result[2 * MAX_BLOCKS] = {0.0};
    cudaMalloc(&d_result, 2 * sizeof(double) * MAX_BLOCKS);
    cudaMemcpy(d_result, result, 2 * sizeof(double) * MAX_BLOCKS, cudaMemcpyHostToDevice);
    checkCudaError("result allocation");

    cout << "[GPU] Setting up global constants..." << endl;
    GlobalConstants params;
    params.num_qubits = num_qubits;
    params.num_words = num_words / num_blocks;
    params.num_gates = num_gates;
    params.pauli_words = d_pauli_words;
    params.coeffs = (cuDoubleComplex *)d_coeffs;
    params.prev_words = d_prev_words;
    params.prev_blocks = d_prev_blocks;
    params.gate_types = d_gate_types;
    params.gate_qubits = d_gate_qubits;
    params.gate_angles = d_gate_angles;
    params.gate_idx = d_gate_idx;
    params.result = d_result;
    cudaMemcpyToSymbol(cuConstants, &params, sizeof(GlobalConstants));
    checkCudaError("params allocation");

    cout << "[GPU] Initialization complete. Ready to run!" << endl;
}

PauliSimulatorGPU::~PauliSimulatorGPU()
{
    cleanup();
}

void PauliSimulatorGPU::cleanup()
{
    if (d_pauli_words)
    {
        cudaFree(d_pauli_words);
        d_pauli_words = nullptr;
    }
    if (d_coeffs)
    {
        cudaFree(d_coeffs);
        d_coeffs = nullptr;
    }
    if (d_prev_words)
    {
        cudaFree(d_prev_words);
        d_prev_words = nullptr;
    }
    if (d_prev_blocks)
    {
        cudaFree(d_prev_blocks);
        d_prev_blocks = nullptr;
    }
    if (d_gate_types)
    {
        cudaFree(d_gate_types);
        d_gate_types = nullptr;
    }
    if (d_gate_qubits)
    {
        cudaFree(d_gate_qubits);
        d_gate_qubits = nullptr;
    }
    if (d_gate_angles)
    {
        cudaFree(d_gate_angles);
        d_gate_angles = nullptr;
    }
    if (d_result)
    {
        cudaFree(d_result);
        d_result = nullptr;
    }
    num_words = 0;

    cudaDeviceSynchronize();
}

bool PauliSimulatorGPU::allocatePauliWords(const map<PauliWord, Complex> &obs)
{
    num_words = obs.size();
    if (num_words == 0)
    {
        d_pauli_words = nullptr;
        d_coeffs = nullptr;
        return false;
    }
    num_blocks = (num_words + (MAX_PAULI_WORDS / 2) - 1) / (MAX_PAULI_WORDS / 2);
    if (num_blocks > MAX_BLOCKS)
    {
        cerr
            << "Want to allocate "
            << num_words << " words over "
            << num_blocks
            << " blocks, but can only allocate "
            << MAX_BLOCKS << " blocks at "
            << MAX_PAULI_WORDS / 2 << " per block"
            << endl;
        d_pauli_words = nullptr;
        d_coeffs = nullptr;
        num_words = 0;
        return false;
    }
    // Allocate device memory for Pauli words (1 byte per qubit per word)
    size_t pauli_words_size = MAX_PAULI_WORDS * MAX_BLOCKS * num_qubits * sizeof(Pauli);
    cudaMalloc(&d_pauli_words, pauli_words_size);

    // Allocate device memory for coefficients (2 doubles per word: real and imag)
    size_t coeffs_size = MAX_PAULI_WORDS * MAX_BLOCKS * 2 * sizeof(double);
    cudaMalloc(&d_coeffs, coeffs_size);

    size_t prev_words_size = MAX_BLOCKS * sizeof(int);
    cudaMalloc(&d_prev_words, prev_words_size);
    cudaMalloc(&d_prev_blocks, sizeof(int));


    // Create host buffers for flattened data
    vector<Pauli> h_pauli_words(MAX_BLOCKS * MAX_PAULI_WORDS * num_qubits, I);
    vector<double> h_coeffs(MAX_BLOCKS * MAX_PAULI_WORDS * 2, 0.0);
    int h_prev_words[MAX_BLOCKS] = {0};

    // Flatten the observable map
    int word_idx = 0;
    for (const auto &[pw, coeff] : obs)
    {
        // Copy Pauli operators for this word
        for (int qubit = 0; qubit < num_qubits; ++qubit)
        {
            h_pauli_words[word_idx * num_qubits + qubit] = pw.ops[qubit];
        }

        // Copy coefficient (real and imaginary parts)
        h_coeffs[word_idx * 2] = coeff.real();     // real part
        h_coeffs[word_idx * 2 + 1] = coeff.imag(); // imaginary part

        word_idx++;
    }
    h_prev_words[0] = num_words;
    

    // Copy data to device
    cudaMemcpy(d_pauli_words, h_pauli_words.data(), pauli_words_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, h_coeffs.data(), coeffs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_words, h_prev_words, prev_words_size, cudaMemcpyHostToDevice);
    int x = 1;
    cudaMemcpy(d_prev_blocks, &x, sizeof(int), cudaMemcpyHostToDevice);

    // Check for errors
    if (checkCudaError("gate allocation"))
    {
        cleanup();
        return false;
    }
    return true;
}

void PauliSimulatorGPU::allocateGates(const vector<Gate> &circuit)
{
    // Allocate and copy gates to device
    if (!circuit.empty())
    {
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
        cudaMalloc(&d_gate_idx, sizeof(int));
        cudaMemcpy(d_gate_types, gateTypes, gate_types_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate_qubits, gateQubits, gate_qubits_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate_angles, gateAngles, gate_angles_size, cudaMemcpyHostToDevice);
        num_gates--;
        cudaMemcpy(d_gate_idx, &num_gates, sizeof(int), cudaMemcpyHostToDevice);
        delete[] gateTypes;
        delete[] gateQubits;
        delete[] gateAngles;

        // Check for any CUDA errors
        if (checkCudaError("gate allocation"))
        {
            cleanup();
        }
    }
}

Complex PauliSimulatorGPU::runPropagation(int max_weight)
{
    cout << "[GPU] Starting propagation with max_weight=" << max_weight << endl;
    Complex err_res = Complex(-1.0, -1.0);
    if (num_words == 0)
    {
        cout << "[GPU] No words to process!" << endl;
        return err_res;
    }

    cout << "[GPU] Starting with " << num_words << " Pauli words across " << num_blocks << " blocks" << endl;

    // Launch the kernel
    // Initially don't load
    int current_gate = num_gates;
    vector<int> words(MAX_BLOCKS);

    int iteration = 0;
    while (current_gate >= 0)
    {
        iteration++;
        cout << "[GPU] Iteration " << iteration << ": Processing gate " << (num_gates - current_gate)
             << "/" << num_gates << " with " << num_words << " words" << endl;

        cout << "[GPU] Launching kernel with " << num_blocks << " blocks, " << THREADS_PER_BLOCK << " threads/block" << endl;

        // Check GPU status before launch
        cudaError_t pre_launch_err = cudaGetLastError();
        if (pre_launch_err != cudaSuccess) {
            cout << "[GPU] GPU in error state before kernel launch: " << cudaGetErrorString(pre_launch_err) << endl;
        }

        pauli_propagation_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(max_weight);
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            cout << "[GPU] Kernel launch failed: " << cudaGetErrorString(launch_err) << endl;
            cleanup();
            return err_res;
        }
        cout << "[GPU] Kernel launched successfully, synchronizing..." << endl;

        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess)
        {
            cout << "[GPU] Kernel synchronization failed: " << cudaGetErrorString(sync_err) << endl;
            cleanup();
            return err_res;
        }
        cout << "[GPU] Kernel execution completed successfully" << endl;

        cout << "[GPU] Kernel completed, copying results..." << endl;
        cudaMemcpy(&current_gate, d_gate_idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(words.data(), d_prev_words, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        if (checkCudaError("Memcpy"))
        {
            cout << "[GPU] Memory copy failed!" << endl;
            cleanup();
            return err_res;
        }

        int old_num_words = num_words;
        num_words = 0;
        for (int val : words)
        {
            num_words += val;
        }
        cout << "[GPU] After processing: " << num_words << " words (was " << old_num_words << ")" << endl;

        num_blocks = (num_words + (MAX_PAULI_WORDS / 2) - 1) / (MAX_PAULI_WORDS / 2);
        if (num_blocks > MAX_BLOCKS)
        {
            cerr << "[GPU] Too many words: " << num_words << " requires " << num_blocks << " blocks (max " << MAX_BLOCKS << ")" << endl;
            cleanup();
            return err_res;
        }
    }

    cout << "[GPU] All gates processed, copying final result..." << endl;

    // Add timeout check before final memcpy
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        cout << "[GPU] Device synchronization failed before final copy: " << cudaGetErrorString(sync_err) << endl;
        return err_res;
    }
    cout << "[GPU] Device synchronized, proceeding with final copy..." << endl;

    double result[2 * MAX_BLOCKS] = {0.0}; // Initialize to zeros
    cout << "[GPU] Copying " << (2 * MAX_BLOCKS) << " doubles from device..." << endl;

    cudaError_t memcpy_err = cudaMemcpy(result, d_result, 2 * sizeof(double) * MAX_BLOCKS, cudaMemcpyDeviceToHost);
    if (memcpy_err != cudaSuccess)
    {
        cout << "[GPU] Final result memcpy failed: " << cudaGetErrorString(memcpy_err) << endl;
        return err_res;
    }
    cout << "[GPU] Final memcpy successful, processing results..." << endl;

    // Check if result buffer has any valid data
    bool has_data = false;
    for (int i = 0; i < 2 * MAX_BLOCKS; i++) {
        if (abs(result[i]) > 1e-10) {
            has_data = true;
            break;
        }
    }
    cout << "[GPU] Result buffer contains data: " << (has_data ? "YES" : "NO") << endl;

    for (int i = 1; i < MAX_BLOCKS; i++)
    {
        result[0] += result[2 * i];
        result[1] += result[2 * i + 1];
    }

    Complex final_result(result[0], result[1]);
    cout << "[GPU] Propagation complete! Result: " << final_result << endl;
    cout << "[GPU] Cleaning up..." << endl;
    cleanup(); // Explicit cleanup
    cout << "[GPU] Cleanup complete. Returning result." << endl;
    return final_result;
}