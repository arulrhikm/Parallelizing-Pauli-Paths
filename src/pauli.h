#ifndef PAULI_H
#define PAULI_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <complex>

using Complex = std::complex<double>;

// Pauli matrices encoded as simple enum values for convenience.
enum Pauli
{
    I = 0,
    X,
    Y,
    Z
};


enum GateType
{
    HADAMARD,
    CNOT,
    RZ,
    RX,
    RY,
    S,
    T
};

// A Pauli word applies one Pauli operator per qubit plus an overall phase.
struct PauliWord
{
    std::vector<Pauli> ops;
    Complex phase;
    PauliWord(int n);
    bool operator<(const PauliWord &) const;
    bool operator==(const PauliWord &) const;
    int weight() const;
    std::string to_string() const;
};

struct Gate
{
    GateType type;
    std::vector<int> qubits;
    double angle;
    Gate(GateType t, std::vector<int> q, double a = 0.0);
};

PauliWord apply_gate_conjugation(const Gate &g, const PauliWord &pw);
std::map<PauliWord, Complex> truncate_pauli_words(const std::map<PauliWord, Complex> &in, int max_w);
Complex compute_expectation(const PauliWord &pw);
Complex pauli_propagation(const std::map<PauliWord, Complex> &init,
                          const std::vector<Gate> &circuit,
                          int max_weight);
void print_observable(const std::map<PauliWord, Complex> &obs,
                      const std::string &label);

#endif
