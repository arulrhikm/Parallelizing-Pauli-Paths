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

void test_hadamard_on_z()
{
    cout << "=== Test 1: Hadamard on Z ===\n";
    cout << "Circuit: H(0)\n";
    cout << "Expected: H Z H = X -> <0|X|0> = 0\n";

    PauliWord z(1);
    z.ops[0] = Z;

    map<PauliWord, Complex> obs = {{z, 1.0}};
    vector<Gate> circ = {Gate(HADAMARD, {0})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_hadamard_on_x()
{
    cout << "=== Test 2: Hadamard on X ===\n";
    cout << "Circuit: H(0)\n";
    cout << "Expected: H X H = Z -> <0|Z|0> = 1\n";

    PauliWord x(1);
    x.ops[0] = X;

    map<PauliWord, Complex> obs = {{x, 1.0}};
    vector<Gate> circ = {Gate(HADAMARD, {0})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_bell_state()
{
    cout << "=== Test 3: Bell state, ZZ ===\n";
    cout << "Circuit: H(0), CNOT(0,1)\n";
    cout << "Expected: <ZZ> = 1\n";

    PauliWord zz(2);
    zz.ops[0] = Z;
    zz.ops[1] = Z;

    map<PauliWord, Complex> obs = {{zz, 1.0}};
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(CNOT, {0, 1})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_bell_state_xx()
{
    cout << "=== Test 4: Bell state, XX ===\n";
    cout << "Circuit: H(0), CNOT(0,1)\n";
    cout << "Expected: <XX> = 1\n";

    PauliWord xx(2);
    xx.ops[0] = X;
    xx.ops[1] = X;

    map<PauliWord, Complex> obs = {{xx, 1.0}};
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(CNOT, {0, 1})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_identity_preservation()
{
    cout << "=== Test 5: Identity preservation ===\n";
    cout << "Circuit: H, CNOT, H\n";
    cout << "Expected: I -> I -> <I> = 1\n";

    PauliWord id(2); // all I

    map<PauliWord, Complex> obs = {{id, 1.0}};
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(CNOT, {0, 1}),
        Gate(HADAMARD, {1})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_xi_to_xx()
{
    cout << "=== Test 6: CNOT: XI -> XX ===\n";
    cout << "Expected: XX has zero expectation on |00⟩\n";

    PauliWord xi(2);
    xi.ops[0] = X;

    map<PauliWord, Complex> obs = {{xi, 1.0}};
    vector<Gate> circ = {Gate(CNOT, {0, 1})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_ix_to_ix()
{
    cout << "=== Test 7: CNOT: IX -> IX ===\n";
    cout << "Expected: <IX> = 0\n";

    PauliWord ix(2);
    ix.ops[1] = X;

    map<PauliWord, Complex> obs = {{ix, 1.0}};
    vector<Gate> circ = {Gate(CNOT, {0, 1})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_cnot_iz_to_zz()
{
    cout << "=== Test 8: CNOT: IZ -> ZZ ===\n";
    cout << "Expected: <ZZ> = 1 on |00⟩\n";

    PauliWord iz(2);
    iz.ops[1] = Z;

    map<PauliWord, Complex> obs = {{iz, 1.0}};
    vector<Gate> circ = {Gate(CNOT, {0, 1})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_s_twice()
{
    cout << "=== Test 9: S twice ===\n";
    cout << "Expected: S S Z S^{+} S^{+} = Z -> <Z> = 1\n";

    PauliWord z(1);
    z.ops[0] = Z;

    map<PauliWord, Complex> obs = {{z, 1.0}};
    vector<Gate> circ = {Gate(S, {0}), Gate(S, {0})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_ghz_state()
{
    cout << "=== Test 10: GHZ state, ZZI ===\n";
    cout << "Circuit: H(0), CNOT(0,1), CNOT(0,2)\n";
    cout << "Expected: <ZZI> = 1\n";

    PauliWord zzi(3);
    zzi.ops[0] = Z;
    zzi.ops[1] = Z;

    map<PauliWord, Complex> obs = {{zzi, 1.0}};
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(CNOT, {0, 1}),
        Gate(CNOT, {0, 2})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_s_gate()
{
    cout << "=== Test 11: S on X ===\n";
    cout << "Expected: S X S^{+} = Y -> <Y> = 0\n";

    PauliWord x(1);
    x.ops[0] = X;

    map<PauliWord, Complex> obs = {{x, 1.0}};
    vector<Gate> circ = {Gate(S, {0})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void test_double_hadamard()
{
    cout << "=== Test 12: double Hadamard ===\n";
    cout << "Expected: H H Z H H = Z -> <Z> = 1\n";

    PauliWord z(1);
    z.ops[0] = Z;

    map<PauliWord, Complex> obs = {{z, 1.0}};
    vector<Gate> circ = {
        Gate(HADAMARD, {0}),
        Gate(HADAMARD, {0})};

    Complex r = pauli_propagation(obs, circ, 10);
    cout << "Result: " << r << "\n";
    cout << "Status: " << (abs(r - 1.0) < 1e-10 ? "PASS" : "FAIL") << "\n\n";
}

void run_all_tests() {
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
}