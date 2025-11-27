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

    return tests;
}

bool run_single_test(const TestCase &test, bool use_gpu)
{
    cout << "=== " << test.name << " ===\n";

    Complex result;
    if (use_gpu)
    {
        PauliSimulatorGPU simulator(test.num_qubits, test.initial_obs, test.circuit);
        result = simulator.runPropagation(10);
    }
    else
    {
        result = pauli_propagation(test.initial_obs, test.circuit, 10);
    }

    bool passed = abs(result - test.expected_result) < test.tolerance;

    cout << "Result: " << result << "\n";
    cout << "Expected: " << test.expected_result << "\n";
    cout << "Status: " << (passed ? "PASS" : "FAIL") << "\n\n";

    return passed;
}

void run_all_tests(bool use_gpu)
{
    auto test_cases = create_test_cases();
    int total_tests = test_cases.size();
    int passed_tests = 0;

    cout << "Running " << total_tests << " tests using "
         << (use_gpu ? "GPU" : "CPU") << " simulator\n";
    cout << "========================================\n\n";

    for (const auto &test : test_cases)
    {
        if (run_single_test(test, use_gpu))
        {
            passed_tests++;
        }
    }

    cout << "========================================\n";
    cout << "OVERALL RESULTS:\n";
    cout << "Passed: " << passed_tests << "/" << total_tests << "\n";
    cout << "Failed: " << (total_tests - passed_tests) << "/" << total_tests << "\n";
    cout << "Success Rate: " << fixed << setprecision(1)
         << (100.0 * passed_tests / total_tests) << "%\n";
}

// Optional: Individual test functions for backward compatibility
static void test_hadamard_on_z()
{
    auto tests = create_test_cases();
    run_single_test(tests[0], true);
}

static void test_hadamard_on_x()
{
    auto tests = create_test_cases();
    run_single_test(tests[1], true);
}

// ... similar individual test functions for other tests
