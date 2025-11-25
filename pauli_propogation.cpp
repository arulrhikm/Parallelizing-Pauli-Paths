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

// Pauli matrices encoded as simple enum values for convenience.
enum Pauli { I = 0, X = 1, Y = 2, Z = 3 };

// A Pauli word applies one Pauli operator per qubit plus an overall phase.
struct PauliWord {
    vector<Pauli> ops;
    Complex phase;

    PauliWord(int n) : ops(n, I), phase(1.0) {}

    bool operator<(const PauliWord& other) const {
        if (ops != other.ops) return ops < other.ops;

        // Break ties by phase (mostly to keep map keys deterministic).
        if (abs(phase.real() - other.phase.real()) < 1e-10)
            return phase.imag() < other.phase.imag();
        return phase.real() < other.phase.real();
    }

    bool operator==(const PauliWord& other) const {
        return ops == other.ops;
    }

    // Count how many components are non-identity.
    int weight() const {
        int w = 0;
        for (auto p : ops) {
            if (p != I) ++w;
        }
        return w;
    }

    // Return a simple string representation like "XYZI".
    string to_string() const {
        string s;
        for (auto p : ops) {
            s += (p == I ? 'I' :
                  p == X ? 'X' :
                  p == Y ? 'Y' : 'Z');
        }
        return s;
    }
};

// Supported gate types (only some are implemented below).
enum GateType { HADAMARD, CNOT, RZ, RX, RY, S, T };

struct Gate {
    GateType type;
    vector<int> qubits;
    double angle;

    Gate(GateType t, vector<int> q, double a = 0.0)
        : type(t), qubits(std::move(q)), angle(a) {}
};

// Conjugation by H: H P H (self-adjoint).
PauliWord apply_hadamard_conjugation(const Gate& g, const PauliWord& pw) {
    PauliWord out = pw;
    int t = g.qubits[0];
    Pauli p = pw.ops[t];

    // Basic H mapping rules.
    if (p == X) {
        out.ops[t] = Z;
    } else if (p == Y) {
        out.ops[t] = Y;
        out.phase *= -1.0;   // Y picks up a sign
    } else if (p == Z) {
        out.ops[t] = X;
    }
    return out;
}

// Conjugation by S: S P S†.
PauliWord apply_s_gate_conjugation(const Gate& g, const PauliWord& pw) {
    PauliWord out = pw;
    int t = g.qubits[0];
    Pauli p = pw.ops[t];

    if (p == X) {
        out.ops[t] = Y;
    } else if (p == Y) {
        out.ops[t] = X;
        out.phase *= -1.0;
    }
    return out;
}

// Conjugation by CNOT using explicit component rules.
PauliWord apply_cnot_conjugation(const Gate& g, const PauliWord& pw) {
    PauliWord out = pw;
    int c = g.qubits[0];
    int t = g.qubits[1];

    Pauli pc = pw.ops[c];
    Pauli pt = pw.ops[t];

    // These cases are derived from standard Pauli conjugation under CNOT.
    // Kept explicit for clarity rather than using a compact formula.
    if (pc == I && pt == I) {
        out.ops[c] = I; out.ops[t] = I;
    }
    else if (pc == I && pt == X) {
        out.ops[c] = I; out.ops[t] = X;
    }
    else if (pc == I && pt == Y) {
        out.ops[c] = Z; out.ops[t] = Y;
    }
    else if (pc == I && pt == Z) {
        out.ops[c] = Z; out.ops[t] = Z;
    }
    else if (pc == X && pt == I) {
        out.ops[c] = X; out.ops[t] = X;
    }
    else if (pc == X && pt == X) {
        out.ops[c] = X; out.ops[t] = I;
    }
    else if (pc == X && pt == Y) {
        out.ops[c] = Y; out.ops[t] = Z;
    }
    else if (pc == X && pt == Z) {
        out.ops[c] = Y; out.ops[t] = Y;
        out.phase *= -1.0;
    }
    else if (pc == Y && pt == I) {
        out.ops[c] = Y; out.ops[t] = X;
    }
    else if (pc == Y && pt == X) {
        out.ops[c] = Y; out.ops[t] = I;
    }
    else if (pc == Y && pt == Y) {
        out.ops[c] = X; out.ops[t] = Z;
        out.phase *= -1.0;
    }
    else if (pc == Y && pt == Z) {
        out.ops[c] = X; out.ops[t] = Y;
    }
    else if (pc == Z && pt == I) {
        out.ops[c] = Z; out.ops[t] = I;
    }
    else if (pc == Z && pt == X) {
        out.ops[c] = Z; out.ops[t] = X;
    }
    else if (pc == Z && pt == Y) {
        out.ops[c] = I; out.ops[t] = Y;
        out.phase *= -1.0;
    }
    else if (pc == Z && pt == Z) {
        out.ops[c] = I; out.ops[t] = Z;
    }

    return out;
}

// Dispatch the appropriate conjugation rule.
PauliWord apply_gate_conjugation(const Gate& g, const PauliWord& pw) {
    switch (g.type) {
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
truncate_pauli_words(const map<PauliWord, Complex>& in, int max_w) {
    map<PauliWord, Complex> out;

    for (auto& [pw, c] : in) {
        if (pw.weight() <= max_w)
            out[pw] = c;
    }
    return out;
}

// Expectation value on |0...0⟩: only Z/I survive.
Complex compute_expectation(const PauliWord& pw) {
    for (auto p : pw.ops) {
        if (p == X || p == Y) return 0.0;
    }
    return pw.phase;
}

// Main Heisenberg-picture propagation of a Pauli observable.
Complex pauli_propagation(const map<PauliWord, Complex>& init,
                          const vector<Gate>& circuit,
                          int max_weight) {
    map<PauliWord, Complex> obs = init;

    // Work backwards through the circuit.
    for (int i = (int)circuit.size() - 1; i >= 0; --i) {
        const Gate& g = circuit[i];
        map<PauliWord, Complex> updated;

        // Transform each term under conjugation.
        for (auto& [pw, coeff] : obs) {
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
        for (auto& [pw, c] : updated) {
            if (abs(c) > 1e-10)
                filtered[pw] = c;
        }

        // Apply weight truncation.
        obs = truncate_pauli_words(filtered, max_weight);
    }

    // Final expectation.
    Complex exp_val = 0.0;
    for (auto& [pw, c] : obs) {
        exp_val += c * compute_expectation(pw);
    }

    return exp_val;
}

// Print observable terms for debugging.
void print_observable(const map<PauliWord, Complex>& obs,
                      const string& label) {
    cout << label << ":\n";
    for (auto& [pw, c] : obs) {
        if (abs(c) > 1e-10)
            cout << "  " << pw.to_string() << ": " << c << "\n";
    }
    cout << "\n";
}

void test_hadamard_on_z() {
    cout << "=== Test 1: Hadamard on Z ===\n";
    cout << "Circuit: H(0)\n";
    cout << "Expected: H Z H = X -> <0|X|0> = 0\n";

    PauliWord z(1);
    z.ops[0] = Z;

    map<PauliWord, Complex> obs = { {z, 1.0} };
    vector<Gate> circ = { Gate(HADAMARD, {0}) };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_hadamard_on_x() {
    cout << "=== Test 2: Hadamard on X ===\n";
    cout << "Circuit: H(0)\n";
    cout << "Expected: H X H = Z -> <0|Z|0> = 1\n";

    PauliWord x(1);
    x.ops[0] = X;

    map<PauliWord, Complex> obs = { {x, 1.0} };
    vector<Gate> circ = { Gate(HADAMARD, {0}) };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_bell_state() {
    cout << "=== Test 3: Bell state, ZZ ===\n";
    cout << "Circuit: H(0), CNOT(0,1)\n";
    cout << "Expected: <ZZ> = 1\n";

    PauliWord zz(2);
    zz.ops[0] = Z;
    zz.ops[1] = Z;

    map<PauliWord, Complex> obs = { {zz, 1.0} };
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(CNOT, {0, 1})
    };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_bell_state_xx() {
    cout << "=== Test 4: Bell state, XX ===\n";
    cout << "Circuit: H(0), CNOT(0,1)\n";
    cout << "Expected: <XX> = 1\n";

    PauliWord xx(2);
    xx.ops[0] = X;
    xx.ops[1] = X;

    map<PauliWord, Complex> obs = { {xx, 1.0} };
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(CNOT, {0, 1})
    };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_identity_preservation() {
    cout << "=== Test 5: Identity preservation ===\n";
    cout << "Circuit: H, CNOT, H\n";
    cout << "Expected: I -> I -> <I> = 1\n";

    PauliWord id(2);  // all I

    map<PauliWord, Complex> obs = { {id, 1.0} };
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(CNOT, {0, 1}),
        Gate(HADAMARD, {1})
    };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_xi_to_xx() {
    cout << "=== Test 6: CNOT: XI -> XX ===\n";
    cout << "Expected: XX has zero expectation on |00⟩\n";

    PauliWord xi(2);
    xi.ops[0] = X;

    map<PauliWord, Complex> obs = { {xi, 1.0} };
    vector<Gate> circ = { Gate(CNOT, {0,1}) };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_ix_to_ix() {
    cout << "=== Test 7: CNOT: IX -> IX ===\n";
    cout << "Expected: <IX> = 0\n";

    PauliWord ix(2);
    ix.ops[1] = X;

    map<PauliWord, Complex> obs = { {ix, 1.0} };
    vector<Gate> circ = { Gate(CNOT, {0,1}) };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_iz_to_zz() {
    cout << "=== Test 8: CNOT: IZ -> ZZ ===\n";
    cout << "Expected: <ZZ> = 1 on |00⟩\n";

    PauliWord iz(2);
    iz.ops[1] = Z;

    map<PauliWord, Complex> obs = { {iz, 1.0} };
    vector<Gate> circ = { Gate(CNOT, {0,1}) };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_s_twice() {
    cout << "=== Test 9: S twice ===\n";
    cout << "Expected: S S Z S^{+} S^{+} = Z -> <Z> = 1\n";

    PauliWord z(1);
    z.ops[0] = Z;

    map<PauliWord, Complex> obs = { {z, 1.0} };
    vector<Gate> circ = { Gate(S,{0}), Gate(S,{0}) };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_ghz_state() {
    cout << "=== Test 10: GHZ state, ZZI ===\n";
    cout << "Circuit: H(0), CNOT(0,1), CNOT(0,2)\n";
    cout << "Expected: <ZZI> = 1\n";

    PauliWord zzi(3);
    zzi.ops[0] = Z;
    zzi.ops[1] = Z;

    map<PauliWord, Complex> obs = { {zzi, 1.0} };
    vector<Gate> circ = {
        Gate(HADAMARD,{0}),
        Gate(CNOT,{0,1}),
        Gate(CNOT,{0,2})
    };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_s_gate() {
    cout << "=== Test 11: S on X ===\n";
    cout << "Expected: S X S^{+} = Y -> <Y> = 0\n";

    PauliWord x(1);
    x.ops[0] = X;

    map<PauliWord, Complex> obs = { {x, 1.0} };
    vector<Gate> circ = { Gate(S,{0}) };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_double_hadamard() {
    cout << "=== Test 12: double Hadamard ===\n";
    cout << "Expected: H H Z H H = Z -> <Z> = 1\n";

    PauliWord z(1);
    z.ops[0] = Z;

    map<PauliWord, Complex> obs = { {z, 1.0} };
    vector<Gate> circ = {
        Gate(HADAMARD,{0}),
        Gate(HADAMARD,{0})
    };

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

int main() {
    cout << fixed << setprecision(6);

    cout << "========================================\n";
    cout << "   PAULI PROPAGATION SIMULATOR TESTS\n";
    cout << "========================================\n\n";

    test_hadamard_on_z();
    test_hadamard_on_x();
    test_cnot_bell_state();
    test_cnot_bell_state_xx();
    test_identity_preservation();
    test_cnot_xi_to_xx();
    test_cnot_ix_to_ix();
    test_cnot_iz_to_zz();
    test_s_twice();
    test_ghz_state();
    test_s_gate();
    test_double_hadamard();

    cout << "========================================\n";
    cout << "   ALL TESTS COMPLETED\n";
    cout << "========================================\n";

    return 0;
}
