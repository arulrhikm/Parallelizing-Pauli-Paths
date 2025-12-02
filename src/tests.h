#ifndef __TEST_H__
#define __TEST_H__

#include <vector>
#include <map>
#include <string>
#include <complex>

struct TestCase
{
    std::string name;
    int num_qubits;
    std::map<PauliWord, Complex> initial_obs;
    std::vector<Gate> circuit;
    std::complex<double> expected_result;
    double tolerance;
};

double run_single_test(const TestCase &test, int i, bool use_gpu);
double run_single_test(int i, bool use_gpu);
void run_all_tests(bool use_gpu = true);

#endif