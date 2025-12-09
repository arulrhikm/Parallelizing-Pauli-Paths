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
    std::cout << "  Format 1: " << program_name << " <test_number|all> <cpu|gpu>\n";
    std::cout << "  Format 2: " << program_name << " [-c] [-t test_number] [-?]\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " 23 gpu     # Run test 23 on GPU\n";
    std::cout << "  " << program_name << " all cpu    # Run all tests on CPU\n";
    std::cout << "  " << program_name << " -t 5 -c    # Run test 5 on CPU (flag format)\n\n";
    std::cout << "Options:\n";
    std::cout << "  -? : prints this help message\n";
    std::cout << "  -c : use CPU (default is GPU)\n";
    std::cout << "  -t i : run single test i\n";
}

int main(int argc, char *argv[])
{
    // Default values
    bool runGPU = true;
    int test_number = 0;
    bool single_test = false;

    // Support both formats:
    // Format 1: ./program test_num mode (e.g., ./program 23 gpu)
    // Format 2: ./program -t test_num -c (e.g., ./program -t 23 -c)
    
    if (argc >= 3 && argv[1][0] != '-')
    {
        // Format 1: positional arguments (test_num mode)
        try
        {
            std::string first_arg = argv[1];
            
            if (first_arg == "all")
            {
                single_test = false;
            }
            else
            {
                test_number = std::stoi(first_arg);
                single_test = true;
            }
            
            std::string mode = argv[2];
            if (mode == "cpu" || mode == "CPU")
            {
                runGPU = false;
            }
            else if (mode == "gpu" || mode == "GPU")
            {
                runGPU = true;
            }
            else
            {
                std::cerr << "Error: Mode must be 'cpu' or 'gpu', got '" << mode << "'\n";
                return 1;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: Invalid arguments\n";
            std::cerr << "Usage: " << argv[0] << " <test_number|all> <cpu|gpu>\n";
            std::cerr << "Example: " << argv[0] << " 23 gpu\n";
            return 1;
        }
    }
    else
    {
        // Format 2: flag-based arguments
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "-c")
            {
                runGPU = false;
            }
            else if (arg == "-t" && i + 1 < argc)
            {
                try
                {
                    test_number = std::stoi(argv[++i]);
                    single_test = true;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: Invalid test number '" << argv[i] << "'\n";
                    return 1;
                }
            }
            else if (arg == "-?")
            {
                print_help(argv[0]);
                return 0;
            }
            else
            {
                std::cerr << "Error: Unknown argument '" << arg << "'\n";
                print_help(argv[0]);
                return 1;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========================================\n";
    std::cout << "   PAULI PROPAGATION SIMULATOR TESTS\n";
    std::cout << "========================================\n\n";
    std::cout << "Running on: " << (runGPU ? "GPU" : "CPU") << "\n";

    if (single_test)
    {
        std::cout << "Running single test: " << test_number << "\n\n";
    }
    else
    {
        std::cout << "Running all tests\n\n";
    }

    // Run the appropriate test(s)
    if (single_test)
    {
        run_single_test(test_number, runGPU);
    }
    else
    {
        // If you implement a function to run all tests:
        run_all_tests(runGPU);
    }

    std::cout << "========================================\n";
    std::cout << "   TESTS COMPLETED\n";
    std::cout << "========================================\n";
    return 0;
}