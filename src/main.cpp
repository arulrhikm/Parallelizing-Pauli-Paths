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
    std::cout << "Usage: " << program_name << " [-d device] [-?]\n";
    std::cout << "Description\n";
    std::cout << "?: prints this message\n";
    std::cout << "d: specific device either \"cpu\" or \"gpu\" (default \"gpu\")\n";
}

int main(int argc, char *argv[])
{
    // Default device is GPU
    bool runGPU = true;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-d" && i + 1 < argc)
        {
            std::string device = argv[++i];
            if (device == "cpu")
            {
                runGPU = false;
            }
            else if (device == "gpu")
            {
                runGPU = true;
            }
            else
            {
                std::cerr << "Error: Unknown device '" << device << "'. Use 'cpu' or 'gpu'.\n";
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

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========================================\n";
    std::cout << "   PAULI PROPAGATION SIMULATOR TESTS\n";
    std::cout << "========================================\n\n";
    std::cout << "Running on: " << (runGPU ? "GPU" : "CPU") << "\n\n";

    // You'll need to modify run_all_tests() to accept the runGPU parameter
    // For now, I'll assume you'll update your test functions to use this flag
    run_all_tests(runGPU);
    // run_single_test(19, runGPU);

    std::cout << "========================================\n";
    std::cout << "   ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
    return 0;
}
