#include "pauli.h"
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;
using Complex = complex<double>;

//PolyWord Struct
PauliWord::PauliWord(int n) : ops(n, I), phase(1.0) {}

bool PauliWord::operator<(const PauliWord &other) const
{
    if (ops != other.ops)
        return ops < other.ops;
        // Break ties by phase (mostly to keep map keys deterministic).
    if (abs(phase.real() - other.phase.real()) < 1e-10)
        return phase.imag() < other.phase.imag();
    return phase.real() < other.phase.real();
}

bool PauliWord::operator==(const PauliWord &other) const
{
    return ops == other.ops;
}

    // Count how many components are non-identity.
int PauliWord::weight() const
{
    int w = 0;
    for (auto p : ops)
    {
        if (p != I)
            ++w;
    }
    return w;
}

// Return a simple string representation like "XYZI".
string PauliWord::to_string() const
{
    string s;
    for (auto p : ops)
    {
        s += (p == I ? 'I' : p == X ? 'X'
                         : p == Y   ? 'Y'
                                    : 'Z');
    }
    return s;
}

// Gate construct
Gate::Gate(GateType t, vector<int> q, double a)
    : type(t), qubits(std::move(q)), angle(a) {}

// Conjugation by H: H P H (self-adjoint).
PauliWord apply_hadamard_conjugation(const Gate &g, const PauliWord &pw)
{
    PauliWord out = pw;
    int t = g.qubits[0];
    Pauli p = pw.ops[t];

    // Basic H mapping rules.
    if (p == X)
    {
        out.ops[t] = Z;
    }
    else if (p == Y)
    {
        out.ops[t] = Y;
        out.phase *= -1.0; // Y picks up a sign
    }
    else if (p == Z)
    {
        out.ops[t] = X;
    }
    return out;
}

// Conjugation by S: S P S†.
PauliWord apply_s_gate_conjugation(const Gate &g, const PauliWord &pw)
{
    PauliWord out = pw;
    int t = g.qubits[0];
    Pauli p = pw.ops[t];

    if (p == X)
    {
        out.ops[t] = Y;
    }
    else if (p == Y)
    {
        out.ops[t] = X;
        out.phase *= -1.0;
    }
    return out;
}

// Conjugation by CNOT using explicit component rules.
PauliWord apply_cnot_conjugation(const Gate &g, const PauliWord &pw)
{
    PauliWord out = pw;
    int c = g.qubits[0];
    int t = g.qubits[1];

    Pauli pc = pw.ops[c];
    Pauli pt = pw.ops[t];

    // These cases are derived from standard Pauli conjugation under CNOT.
    // Kept explicit for clarity rather than using a compact formula.
    if (pc == I && pt == I)
    {
        out.ops[c] = I;
        out.ops[t] = I;
    }
    else if (pc == I && pt == X)
    {
        out.ops[c] = I;
        out.ops[t] = X;
    }
    else if (pc == I && pt == Y)
    {
        out.ops[c] = Z;
        out.ops[t] = Y;
    }
    else if (pc == I && pt == Z)
    {
        out.ops[c] = Z;
        out.ops[t] = Z;
    }
    else if (pc == X && pt == I)
    {
        out.ops[c] = X;
        out.ops[t] = X;
    }
    else if (pc == X && pt == X)
    {
        out.ops[c] = X;
        out.ops[t] = I;
    }
    else if (pc == X && pt == Y)
    {
        out.ops[c] = Y;
        out.ops[t] = Z;
    }
    else if (pc == X && pt == Z)
    {
        out.ops[c] = Y;
        out.ops[t] = Y;
        out.phase *= -1.0;
    }
    else if (pc == Y && pt == I)
    {
        out.ops[c] = Y;
        out.ops[t] = X;
    }
    else if (pc == Y && pt == X)
    {
        out.ops[c] = Y;
        out.ops[t] = I;
    }
    else if (pc == Y && pt == Y)
    {
        out.ops[c] = X;
        out.ops[t] = Z;
        out.phase *= -1.0;
    }
    else if (pc == Y && pt == Z)
    {
        out.ops[c] = X;
        out.ops[t] = Y;
    }
    else if (pc == Z && pt == I)
    {
        out.ops[c] = Z;
        out.ops[t] = I;
    }
    else if (pc == Z && pt == X)
    {
        out.ops[c] = Z;
        out.ops[t] = X;
    }
    else if (pc == Z && pt == Y)
    {
        out.ops[c] = I;
        out.ops[t] = Y;
        out.phase *= -1.0;
    }
    else if (pc == Z && pt == Z)
    {
        out.ops[c] = I;
        out.ops[t] = Z;
    }

    return out;
}

// Dispatch the appropriate conjugation rule.
PauliWord apply_gate_conjugation(const Gate &g, const PauliWord &pw)
{
    switch (g.type)
    {
    case CNOT:
        return apply_cnot_conjugation(g, pw);
    case HADAMARD:
        return apply_hadamard_conjugation(g, pw);
    case S:
        return apply_s_gate_conjugation(g, pw);
    default:
        // STILL WORKING ON OTHER GATES
        return pw;
    }
}

// Filter Pauli words by weight limit.
map<PauliWord, Complex>
truncate_pauli_words(const map<PauliWord, Complex> &in, int max_w)
{
    map<PauliWord, Complex> out;

    for (auto &[pw, c] : in)
    {
        if (pw.weight() <= max_w)
            out[pw] = c;
    }
    return out;
}

// Expectation value on |0...0⟩: only Z/I survive.
Complex compute_expectation(const PauliWord &pw)
{
    for (auto p : pw.ops)
    {
        if (p == X || p == Y)
            return 0.0;
    }
    return pw.phase;
}

// Main Heisenberg-picture propagation of a Pauli observable.
Complex pauli_propagation(const map<PauliWord, Complex> &init,
                          const vector<Gate> &circuit,
                          int max_weight)
{
    map<PauliWord, Complex> obs = init;

    // Work backwards through the circuit.
    for (int i = (int)circuit.size() - 1; i >= 0; --i)
    {
        const Gate &g = circuit[i];
        map<PauliWord, Complex> updated;

        // Transform each term under conjugation.
        for (auto &[pw, coeff] : obs)
        {
            PauliWord transformed = apply_gate_conjugation(g, pw);

            // Use a canonical key with phase stripped out.
            PauliWord key(transformed.ops.size());
            key.ops = transformed.ops;
            key.phase = 1.0;

            Complex combined = coeff * transformed.phase;
            updated[key] += combined;
        }

        // Drop tiny coefficients to keep the map clean.
        map<PauliWord, Complex> filtered;
        for (auto &[pw, c] : updated)
        {
            if (abs(c) > 1e-10)
                filtered[pw] = c;
        }

        // Apply weight truncation.
        obs = truncate_pauli_words(filtered, max_weight);
        cout << "GATE FINISHED\n";
        for (auto &[pw, c] : obs)
        {
            
            cout << pw.to_string() << c << "\n";
        }
    }

    // Final expectation.
    Complex exp_val = 0.0;
    for (auto &[pw, c] : obs)
    {
        exp_val += c * compute_expectation(pw);
    }

    return exp_val;
}

// Print observable terms for debugging.
void print_observable(const map<PauliWord, Complex> &obs,
                      const string &label)
{
    cout << label << ":\n";
    for (auto &[pw, c] : obs)
    {
        if (abs(c) > 1e-10)
            cout << "  " << pw.to_string() << ": " << c << "\n";
    }
    cout << "\n";
}
