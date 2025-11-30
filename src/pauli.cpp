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

// T gate conjugation
PauliWord apply_t_gate_conjugation(const Gate &g, const PauliWord &pw)
{
    PauliWord out = pw;
    int t = g.qubits[0];
    Pauli p = pw.ops[t];
    
    // Simplified T gate - not fully Clifford
    if (p == X)
    {
        out.ops[t] = Y;
        out.phase *= Complex(1.0/sqrt(2.0), 1.0/sqrt(2.0));
    }
    else if (p == Y)
    {
        out.ops[t] = X;
        out.phase *= Complex(1.0/sqrt(2.0), -1.0/sqrt(2.0));
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

// Handle rotation gates that produce multiple Pauli terms
map<PauliWord, Complex> apply_gate_conjugation_multi(const Gate &g, const PauliWord &pw)
{
    map<PauliWord, Complex> result;
    int t = g.qubits[0];
    Pauli p = pw.ops[t];
    double theta = g.angle;
    
    if (g.type == RZ)
    {
        // X -> cos*X + sin*Y,  Y -> -sin*X + cos*Y,  Z -> Z
        if (p == X)
        {
            PauliWord px = pw; px.ops[t] = X; px.phase = 1.0;
            PauliWord py = pw; py.ops[t] = Y; py.phase = 1.0;
            result[px] = pw.phase * cos(theta);
            result[py] = pw.phase * sin(theta);
        }
        else if (p == Y)
        {
            PauliWord px = pw; px.ops[t] = X; px.phase = 1.0;
            PauliWord py = pw; py.ops[t] = Y; py.phase = 1.0;
            result[px] = pw.phase * (-sin(theta));
            result[py] = pw.phase * cos(theta);
        }
        else
        {
            PauliWord unchanged = pw;
            unchanged.phase = 1.0;
            result[unchanged] = pw.phase;
        }
    }
    else if (g.type == RX)
    {
        // Y -> cos*Y + sin*Z,  Z -> -sin*Y + cos*Z,  X -> X
        if (p == Y)
        {
            PauliWord py = pw; py.ops[t] = Y; py.phase = 1.0;
            PauliWord pz = pw; pz.ops[t] = Z; pz.phase = 1.0;
            result[py] = pw.phase * cos(theta);
            result[pz] = pw.phase * sin(theta);
        }
        else if (p == Z)
        {
            PauliWord py = pw; py.ops[t] = Y; py.phase = 1.0;
            PauliWord pz = pw; pz.ops[t] = Z; pz.phase = 1.0;
            result[py] = pw.phase * (-sin(theta));
            result[pz] = pw.phase * cos(theta);
        }
        else
        {
            PauliWord unchanged = pw;
            unchanged.phase = 1.0;
            result[unchanged] = pw.phase;
        }
    }
    else if (g.type == RY)
    {
        // X -> cos*X - sin*Z,  Z -> sin*X + cos*Z,  Y -> Y
        if (p == X)
        {
            PauliWord px = pw; px.ops[t] = X; px.phase = 1.0;
            PauliWord pz = pw; pz.ops[t] = Z; pz.phase = 1.0;
            result[px] = pw.phase * cos(theta);
            result[pz] = pw.phase * (-sin(theta));
        }
        else if (p == Z)
        {
            PauliWord px = pw; px.ops[t] = X; px.phase = 1.0;
            PauliWord pz = pw; pz.ops[t] = Z; pz.phase = 1.0;
            result[px] = pw.phase * sin(theta);
            result[pz] = pw.phase * cos(theta);
        }
        else
        {
            PauliWord unchanged = pw;
            unchanged.phase = 1.0;
            result[unchanged] = pw.phase;
        }
    }
    else
    {
        // Clifford gates only give one output
        PauliWord transformed = apply_gate_conjugation(g, pw);
        PauliWord key = transformed;
        key.phase = 1.0;
        result[key] = transformed.phase;
    }
    
    return result;
}

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
    case T:
        return apply_t_gate_conjugation(g, pw);
    default:
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

// Propagate observable backward through circuit
Complex pauli_propagation(const map<PauliWord, Complex> &init,
                          const vector<Gate> &circuit,
                          int max_weight)
{
    map<PauliWord, Complex> obs = init;
    
    // Go backwards through gates
    for (int i = (int)circuit.size() - 1; i >= 0; --i)
    {
        const Gate &g = circuit[i];
        map<PauliWord, Complex> updated;

        for (auto &[pw, coeff] : obs)
        {
            map<PauliWord, Complex> transformed_terms = apply_gate_conjugation_multi(g, pw);
            
            for (auto &[transformed, trans_phase] : transformed_terms)
            {
                // Strip phase from key
                PauliWord key(transformed.ops.size());
                key.ops = transformed.ops;
                key.phase = 1.0;

                updated[key] += coeff * trans_phase;
            }
        }

        // Drop small terms
        map<PauliWord, Complex> filtered;
        for (auto &[pw, c] : updated)
        {
            if (abs(c) > 1e-10)
                filtered[pw] = c;
        }

        obs = truncate_pauli_words(filtered, max_weight);
    }

    // Compute expectation value
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
