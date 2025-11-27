#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <complex>
#include "pauli.h"
#include "pauli_gpu.h"
#include "tests.h"


int main()
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========================================\n";
    std::cout << "   PAULI PROPAGATION SIMULATOR TESTS\n";
    std::cout << "========================================\n\n";

    run_all_tests(false);

    std::cout << "========================================\n";
    std::cout << "   ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
    return 0;
}
