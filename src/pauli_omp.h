#ifndef PAULI_OMP_H
#define PAULI_OMP_H

#include "pauli.h"
#include <map>
#include <unordered_map>
#include <vector>
#include <complex>

// Hash functor for PauliWord, enabling O(1) lookup in unordered_map.
// Equality is already defined via PauliWord::operator== (compares ops only).
struct PauliWordHash {
    std::size_t operator()(const PauliWord &pw) const noexcept {
        // FNV-style mixing over each Pauli operator byte
        std::size_t seed = pw.ops.size();
        for (auto op : pw.ops) {
            seed ^= static_cast<std::size_t>(op) + 0x9e3779b9u
                    + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using PauliMapOMP = std::unordered_map<PauliWord, std::complex<double>, PauliWordHash>;

// OpenMP-parallelized Pauli propagation.
//
// Identical algorithm to pauli_propagation() in pauli.cpp, but with two
// key changes for performance:
//   1. The inner loop over Pauli words is parallelized with OpenMP.
//      Each thread accumulates results into a thread-private unordered_map;
//      the per-thread maps are merged serially after the parallel section.
//   2. The accumulator uses unordered_map (O(1) amortized) instead of
//      std::map (O(log n)) for a fairer CPU baseline.
//
// num_threads = 1 reproduces the sequential unordered_map baseline.
std::complex<double> pauli_propagation_omp(
    const std::map<PauliWord, std::complex<double>> &init,
    const std::vector<Gate> &circuit,
    int max_weight,
    int num_threads = 1);

#endif // PAULI_OMP_H
