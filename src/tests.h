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
    int repeat = 1; // number of times to repeat CPU propagation to increase runtime for stress tests
};

// num_omp_threads > 0  => use OpenMP baseline (requires OMP_ENABLED build flag)
// num_omp_threads == 0 => sequential CPU or GPU depending on use_gpu flag
double run_single_test(const TestCase &test, int i, bool use_gpu, int num_omp_threads = 0);
double run_single_test(int i, bool use_gpu, int num_omp_threads = 0);
void run_all_tests(bool use_gpu = true, int num_omp_threads = 0);

#endif