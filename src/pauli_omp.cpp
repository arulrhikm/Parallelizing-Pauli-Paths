#include "pauli_omp.h"
#include "pauli.h"

#include <unordered_map>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Filter Pauli words whose weight exceeds max_w.
static PauliMapOMP truncate_omp(const PauliMapOMP &in, int max_w)
{
    PauliMapOMP out;
    out.reserve(in.size());
    for (auto &[pw, c] : in) {
        if (pw.weight() <= max_w)
            out[pw] = c;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

Complex pauli_propagation_omp(const std::map<PauliWord, Complex> &init,
                               const std::vector<Gate> &circuit,
                               int max_weight,
                               int num_threads)
{
    // Load initial observable into unordered_map for O(1) lookup
    PauliMapOMP obs(init.begin(), init.end());

    // Gate loop (backward, matching the sequential implementation)
    for (int gi = static_cast<int>(circuit.size()) - 1; gi >= 0; --gi) {
        const Gate &g = circuit[gi];

        // Snapshot into a flat vector so threads can index with integers
        std::vector<std::pair<PauliWord, Complex>> words(obs.begin(), obs.end());
        const int n = static_cast<int>(words.size());

        // Allocate one accumulator per potential thread up front.
        // Using the num_threads clause guarantees thread IDs are in [0, num_threads).
        std::vector<PauliMapOMP> locals(num_threads);

        // Each Pauli word is independent: apply the gate conjugation and
        // accumulate resulting terms into the thread-private map.
        #pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
        for (int j = 0; j < n; ++j) {
            const int tid             = omp_get_thread_num();
            const PauliWord &pw       = words[j].first;
            const Complex   &coef     = words[j].second;

            // apply_gate_conjugation_multi is pure (no shared state written)
            auto terms = apply_gate_conjugation_multi(g, pw);

            for (auto &[tr, tr_phase] : terms) {
                // Strip phase into the coefficient; key uses phase = 1
                PauliWord key(static_cast<int>(tr.ops.size()));
                key.ops   = tr.ops;
                key.phase = 1.0;
                locals[tid][key] += coef * tr_phase;
            }
        }

        // Serial merge of thread-private maps
        PauliMapOMP updated;
        updated.reserve(obs.size() * 2);
        for (auto &lm : locals) {
            for (auto &[pw, c] : lm) {
                updated[pw] += c;
            }
        }

        // Drop numerically negligible terms, then apply weight truncation
        PauliMapOMP filtered;
        filtered.reserve(updated.size());
        for (auto &[pw, c] : updated) {
            if (std::abs(c) > 1e-10)
                filtered[pw] = c;
        }

        obs = truncate_omp(filtered, max_weight);
    }

    // Expectation value on |0...0>
    Complex exp_val = 0.0;
    for (auto &[pw, c] : obs) {
        exp_val += c * compute_expectation(pw);
    }
    return exp_val;
}
