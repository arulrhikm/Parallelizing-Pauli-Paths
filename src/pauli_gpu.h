#ifndef __PAULI_GPU_H__
#define __PAULI_GPU_H__
#include <map>
#include <vector>
#include "pauli.h"

class PauliSimulatorGPU
{
public:
    PauliSimulatorGPU(int num_qubits, 
                      const std::map<PauliWord, Complex> &init,
                      const std::vector<Gate> &circuit);
    ~PauliSimulatorGPU();

    // main GPU-based propagation
    Complex runPropagation(int max_weight);

    // manual cleanup (optional since destructor already handles it)
    void cleanup();

private:
    // device buffers
    Pauli *d_pauli_words;
    double *d_coeffs;
    int *d_prev_words;
    int *d_prev_blocks;
    GateType *d_gate_types;
    int *d_gate_qubits;
    double *d_gate_angles;
    int *d_gate_idx;
    double *d_result;

    int num_qubits;
    int num_words;
    int num_gates;
    int num_blocks;

    // host flatten/rebuild utilities
    bool allocatePauliWords(const std::map<PauliWord, Complex> &obs);
    void allocateGates(const std::vector<Gate> &circuit);
};
#endif
