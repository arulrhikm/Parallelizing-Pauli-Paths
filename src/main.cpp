#include <iostream>
#include <iomanip>
#include "pauli.h"
#include "pauli_gpu.h"

void run_all_tests(); // from tests.cpp

int main()
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========================================\n";
    std::cout << "   PAULI PROPAGATION SIMULATOR TESTS\n";
    std::cout << "========================================\n\n";

    run_all_tests();

    std::cout << "========================================\n";
    std::cout << "   ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
    return 0;
}
