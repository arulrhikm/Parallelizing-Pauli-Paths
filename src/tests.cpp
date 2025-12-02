#include "pauli.h"
#include "pauli_gpu.h"
#include "tests.h"
#include <vector>
#include <map>
#include <string>
#include <complex>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using Complex = complex<double>;


static vector<TestCase> create_test_cases()
{
    vector<TestCase> tests;

    // Test 1: Hadamard on Z
    {
        PauliWord z(1);
        z.ops[0] = Z;
        tests.push_back({"Hadamard on Z",
                         1,
                         {{z, 1.0}},
                         {Gate(HADAMARD, {0})},
                         0.0,
                         1e-10});
    }

    // Test 2: Hadamard on X
    {
        PauliWord x(1);
        x.ops[0] = X;
        tests.push_back({"Hadamard on X",
                         1,
                         {{x, 1.0}},
                         {Gate(HADAMARD, {0})},
                         1.0,
                         1e-10});
    }

    // Test 3: Bell state, ZZ
    {
        PauliWord zz(2);
        zz.ops[0] = Z;
        zz.ops[1] = Z;
        tests.push_back({"Bell state, ZZ",
                         2,
                         {{zz, 1.0}},
                         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1})},
                         1.0,
                         1e-10});
    }

    // Test 4: Bell state, XX
    {
        PauliWord xx(2);
        xx.ops[0] = X;
        xx.ops[1] = X;
        tests.push_back({"Bell state, XX",
                         2,
                         {{xx, 1.0}},
                         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1})},
                         1.0,
                         1e-10});
    }

    // Test 5: Identity preservation
    {
        PauliWord id(2);
        tests.push_back({"Identity preservation",
                         2,
                         {{id, 1.0}},
                         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}), Gate(HADAMARD, {1})},
                         1.0,
                         1e-10});
    }

    // Test 6: CNOT: XI -> XX
    {
        PauliWord xi(2);
        xi.ops[0] = X;
        tests.push_back({"CNOT: XI -> XX",
                         2,
                         {{xi, 1.0}},
                         {Gate(CNOT, {0, 1})},
                         0.0,
                         1e-10});
    }

    // Test 7: CNOT: IX -> IX
    {
        PauliWord ix(2);
        ix.ops[1] = X;
        tests.push_back({"CNOT: IX -> IX",
                         2,
                         {{ix, 1.0}},
                         {Gate(CNOT, {0, 1})},
                         0.0,
                         1e-10});
    }

    // Test 8: CNOT: IZ -> ZZ
    {
        PauliWord iz(2);
        iz.ops[1] = Z;
        tests.push_back({"CNOT: IZ -> ZZ",
                         2,
                         {{iz, 1.0}},
                         {Gate(CNOT, {0, 1})},
                         1.0,
                         1e-10});
    }

    // Test 9: S twice
    {
        PauliWord z(1);
        z.ops[0] = Z;
        tests.push_back({"S twice",
                         1,
                         {{z, 1.0}},
                         {Gate(S, {0}), Gate(S, {0})},
                         1.0,
                         1e-10});
    }

    // Test 10: GHZ state, ZZI
    {
        PauliWord zzi(3);
        zzi.ops[0] = Z;
        zzi.ops[1] = Z;
        tests.push_back({"GHZ state, ZZI",
                         3,
                         {{zzi, 1.0}},
                         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}), Gate(CNOT, {0, 2})},
                         1.0,
                         1e-10});
    }

    // Test 11: S on X
    {
        PauliWord x(1);
        x.ops[0] = X;
        tests.push_back({"S on X",
                         1,
                         {{x, 1.0}},
                         {Gate(S, {0})},
                         0.0,
                         1e-10});
    }

    // Test 12: double Hadamard
    {
        PauliWord z(1);
        z.ops[0] = Z;
        tests.push_back({"Double Hadamard",
                         1,
                         {{z, 1.0}},
                         {Gate(HADAMARD, {0}), Gate(HADAMARD, {0})},
                         1.0,
                         1e-10});
    }

    // Test 13: T gate
    {
        PauliWord x(1);
        x.ops[0] = X;
        tests.push_back({"T gate on X",
                         1,
                         {{x, 1.0}},
                         {Gate(T, {0})},
                         0.0,
                         1e-10});
    }

    // Test 14: RZ rotation
    {
        PauliWord x(1);
        x.ops[0] = X;
        tests.push_back({"RZ(π/6) on X",
                         1,
                         {{x, 1.0}},
                         {Gate(RZ, {0}, M_PI/6.0)},
                         0.0,
                         1e-10});
    }

    // Test 15: RX rotation
    {
        PauliWord z(1);
        z.ops[0] = Z;
        double angle = M_PI / 4.0;
        tests.push_back({"RX(π/4) on Z",
                         1,
                         {{z, 1.0}},
                         {Gate(RX, {0}, angle)},
                         cos(angle),
                         1e-9});
    }

    // Test 16: RY rotation
    {
        PauliWord x(1);
        x.ops[0] = X;
        double angle = M_PI / 3.0;
        tests.push_back({"RY(π/3) on X",
                         1,
                         {{x, 1.0}},
                         {Gate(RY, {0}, angle)},
                         -sin(angle),
                         1e-9});
    }

    // Test 17: 3-qubit with rotation
    {
        PauliWord xxx(3);
        xxx.ops[0] = X;
        xxx.ops[1] = X;
        xxx.ops[2] = X;
        tests.push_back({"3-qubit XXX with RZ",
                         3,
                         {{xxx, 1.0}},
                         {Gate(HADAMARD, {0}), 
                          Gate(CNOT, {0, 1}), 
                          Gate(CNOT, {1, 2}),
                          Gate(RZ, {0}, M_PI/8)},
                         cos(M_PI/8),
                         1e-8});
    }

    // Test 18: Bell state + rotation
    {
        PauliWord zz(2);
        zz.ops[0] = Z;
        zz.ops[1] = Z;
        tests.push_back({"Bell state with RX rotation",
                         2,
                         {{zz, 1.0}},
                         {Gate(HADAMARD, {0}), 
                          Gate(CNOT, {0, 1}),
                          Gate(RX, {0}, M_PI/4)},
                         cos(M_PI/4),
                         1e-8});
    }

    // Test 19: 4-qubit GHZ
    {
        PauliWord zzzz(4);
        zzzz.ops[0] = Z;
        zzzz.ops[1] = Z;
        zzzz.ops[2] = Z;
        zzzz.ops[3] = Z;
        tests.push_back({"4-qubit ZZZZ GHZ-like",
                         4,
                         {{zzzz, 1.0}},
                         {Gate(HADAMARD, {0}), 
                          Gate(CNOT, {0, 1}), 
                          Gate(CNOT, {1, 2}),
                          Gate(CNOT, {2, 3})},
                         1.0,
                         1e-10});
    }

    // Test 20: Multiple small rotations
    {
        PauliWord z(2);
        z.ops[0] = Z;
        tests.push_back({"Multiple small rotations",
                         2,
                         {{z, 1.0}},
                         {Gate(RZ, {0}, 0.1),
                          Gate(RX, {0}, 0.1),
                          Gate(RY, {0}, 0.1),
                          Gate(CNOT, {0, 1})},
                         0.990033,
                         1e-5});
    }

    // Test 21: Bigger circuit
    {
        PauliWord obs(5);
        obs.ops[0] = Z;
        obs.ops[1] = Z;
        obs.ops[2] = Z;
        tests.push_back({"5-qubit mixed circuit",
                         5,
                         {{obs, 1.0}},
                         {Gate(HADAMARD, {0}),
                          Gate(CNOT, {0, 1}),
                          Gate(HADAMARD, {2}),
                          Gate(CNOT, {2, 3}),
                          Gate(S, {1}),
                          Gate(CNOT, {1, 4})},
                         0.0,
                         1e-10});
    }

    // Test 22: Deep layered circuit
    {
        PauliWord x(3);
        x.ops[1] = X;
        vector<Gate> circuit;
        for (int i = 0; i < 10; i++)
        {
            circuit.push_back(Gate(RZ, {0}, 0.05));
            circuit.push_back(Gate(HADAMARD, {1}));
            circuit.push_back(Gate(CNOT, {0, 1}));
        }
        tests.push_back({"Deep circuit 10 layers",
                         3,
                         {{x, 1.0}},
                         circuit,
                         0.0,
                         1e-6});
    }

    return tests;
}

bool run_single_test(const TestCase &test, int i, bool use_gpu)
{
    cout << "=== " << i + 1 << ". " << test.name << " ===\n";

    Complex result;
    if (use_gpu)
    {
        #ifndef CPU_ONLY
        PauliSimulatorGPU simulator(test.num_qubits, test.initial_obs, test.circuit);
        result = simulator.runPropagation(10);
        #else
        cout << "GPU not available, using CPU instead\n";
        result = pauli_propagation(test.initial_obs, test.circuit, 10);
        #endif
    }
    else
    {
        result = pauli_propagation(test.initial_obs, test.circuit, 10);
    }

    bool passed = abs(result - test.expected_result) < test.tolerance;

    if (passed)
    {
        cout << "\033[92m" << "Status: PASS"  << "\033[0m" << "\n\n";
    } else {
        cout << "\033[31m" << "Status: FAIL" << "\033[0m" << "\n";
        cout << "Result: " << result << "\n";
        cout << "Expected: " << test.expected_result << "\n\n";
    }
    return passed;
}

bool run_single_test(int i, bool use_gpu)
{
    auto test_cases = create_test_cases();
    return run_single_test(test_cases[i], i, use_gpu);
}

void run_all_tests(bool use_gpu)
{
    auto test_cases = create_test_cases();
    int total_tests = test_cases.size();
    int passed_tests = 0;

    cout << "Running " << total_tests << " tests using "
         << (use_gpu ? "GPU" : "CPU") << " simulator\n";
    cout << "========================================\n\n";

    int i = 0;
    for (const auto &test : test_cases)
    {
        if (run_single_test(test, i, use_gpu))
        {
            passed_tests++;
        }
        i++;
    }

    cout << "========================================\n";
    cout << "OVERALL RESULTS:\n";
    cout << "Passed: " << passed_tests << "/" << total_tests << "\n";
    cout << "Failed: " << (total_tests - passed_tests) << "/" << total_tests << "\n";
    cout << "Success Rate: " << fixed << setprecision(1)
         << (100.0 * passed_tests / total_tests) << "%\n";
    cout << setprecision(6);
}

