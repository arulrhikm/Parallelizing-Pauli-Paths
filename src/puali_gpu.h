// PauliSimulatorGPU.hpp
#pragma once
#include <map>
#include <vector>
#include "pauli.h"
#include <cuda_runtime.h>

using HostComplex = Complex;

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
    GateType *d_gates;

    int num_qubits;

    // host flatten/rebuild utilities
    void flattenMapToDevice(const std::map<PauliWord, Complex> &obs);
};
