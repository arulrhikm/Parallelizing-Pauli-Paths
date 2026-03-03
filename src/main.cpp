#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <complex>
#include "pauli.h"
#include "pauli_gpu.h"
#include "tests.h"

void print_help(const char *program_name)
{
    std::cout << "Usage:\n";
    std::cout << "  Format 1: " << program_name << " <test_number|all> <cpu|omp|gpu> [-j <threads>]\n";
    std::cout << "  Format 2: " << program_name << " [-c] [-t test_number] [-j threads] [-?]\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " 23 gpu           # Run test 23 on GPU\n";
    std::cout << "  " << program_name << " all cpu          # Run all tests on CPU (sequential)\n";
    std::cout << "  " << program_name << " all omp -j 16    # Run all tests with 16 OpenMP threads\n";
    std::cout << "  " << program_name << " -t 5 -c          # Run test 5 on CPU (flag format)\n";
    std::cout << "  " << program_name << " -t 5 -j 8        # Run test 5 with 8 OMP threads\n\n";
    std::cout << "Options:\n";
    std::cout << "  -?       : print this help message\n";
    std::cout << "  -c       : use CPU sequential (default is GPU)\n";
    std::cout << "  -t i     : run single test i\n";
    std::cout << "  -j N     : use N OpenMP threads (implies OMP mode, overrides -c/-gpu)\n";
}

int main(int argc, char *argv[])
{
    bool runGPU        = true;
    int  test_number   = 0;
    bool single_test   = false;
    int  num_omp_threads = 0;   // 0 = sequential; >0 = OpenMP mode

    if (argc >= 3 && argv[1][0] != '-')
    {
        // Format 1: positional arguments  test_num  mode  [optional -j N]
        try
        {
            std::string first_arg = argv[1];
            if (first_arg == "all") {
                single_test = false;
            } else {
                test_number = std::stoi(first_arg);
                single_test = true;
            }

            std::string mode = argv[2];
            if (mode == "cpu" || mode == "CPU") {
                runGPU = false;
            } else if (mode == "omp" || mode == "OMP") {
                runGPU = false;
                num_omp_threads = 1;   // default; -j may override below
            } else if (mode == "gpu" || mode == "GPU") {
                runGPU = true;
            } else {
                std::cerr << "Error: Mode must be 'cpu', 'omp', or 'gpu', got '"
                          << mode << "'\n";
                return 1;
            }

            // Optional trailing -j N
            for (int i = 3; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "-j" && i + 1 < argc) {
                    num_omp_threads = std::stoi(argv[++i]);
                    runGPU = false;
                } else {
                    std::cerr << "Error: Unknown extra argument '" << arg << "'\n";
                    return 1;
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: Invalid arguments\n";
            std::cerr << "Usage: " << argv[0] << " <test_number|all> <cpu|omp|gpu> [-j N]\n";
            return 1;
        }
    }
    else
    {
        // Format 2: flag-based arguments
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "-c") {
                runGPU = false;
            } else if (arg == "-t" && i + 1 < argc) {
                try {
                    test_number = std::stoi(argv[++i]);
                    single_test = true;
                } catch (const std::exception &e) {
                    std::cerr << "Error: Invalid test number '" << argv[i] << "'\n";
                    return 1;
                }
            } else if (arg == "-j" && i + 1 < argc) {
                try {
                    num_omp_threads = std::stoi(argv[++i]);
                    runGPU = false;
                } catch (const std::exception &e) {
                    std::cerr << "Error: Invalid thread count '" << argv[i] << "'\n";
                    return 1;
                }
            } else if (arg == "-?") {
                print_help(argv[0]);
                return 0;
            } else {
                std::cerr << "Error: Unknown argument '" << arg << "'\n";
                print_help(argv[0]);
                return 1;
            }
        }
    }

#ifndef OMP_ENABLED
    if (num_omp_threads > 0) {
        std::cerr << "Warning: -j / omp mode requested but this binary was compiled "
                     "without OMP_ENABLED.\n"
                     "         Falling back to sequential CPU mode.\n";
        num_omp_threads = 0;
    }
#endif

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========================================\n";
    std::cout << "   PAULI PROPAGATION SIMULATOR TESTS\n";
    std::cout << "========================================\n\n";

    if (runGPU)
        std::cout << "Running on: GPU\n";
    else if (num_omp_threads > 0)
        std::cout << "Running on: CPU (OpenMP, " << num_omp_threads << " threads)\n";
    else
        std::cout << "Running on: CPU (sequential)\n";

    if (single_test)
        std::cout << "Running single test: " << test_number << "\n\n";
    else
        std::cout << "Running all tests\n\n";

    if (single_test)
        run_single_test(test_number, runGPU, num_omp_threads);
    else
        run_all_tests(runGPU, num_omp_threads);

    std::cout << "========================================\n";
    std::cout << "   TESTS COMPLETED\n";
    std::cout << "========================================\n";
    return 0;
}
