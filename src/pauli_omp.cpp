#include "pauli_omp.h"
#include "pauli.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <complex>
#include <cmath>
#include <omp.h>

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// pauli_propagation_omp – redesigned for real parallelism
// ---------------------------------------------------------------------------
//
// Key insight:
//   • Clifford gates (H, S, T, CNOT) are BIJECTIONS on Pauli strings.
//     Every input word maps to exactly one output word and no two inputs
//     produce the same output.  Therefore NO merge step is needed – the
//     per-word transformation is completely independent and we can apply
//     the gate in-place across all words with a plain parallel for loop.
//
//   • Rotation gates (RZ, RX, RY) can fan out one word into two, so
//     they need a collect→sort→merge pass.  But using a sorted flat vector
//     (instead of per-thread unordered_maps) avoids the heavy hash
//     allocator overhead that caused the old code to be slower at 16 threads
//     than at 1 thread on some tests.
//
// For Clifford-heavy circuits the entire gate loop becomes a sequence of
// pure-parallel in-place transforms with no synchronisation, giving near-
// linear speedup with thread count.

static bool is_single_output(const Gate &g)
{
    // RZ / RX / RY are the only gates that can fan one word into two terms.
    return g.type != RZ && g.type != RX && g.type != RY;
}

Complex pauli_propagation_omp(const std::map<PauliWord, Complex> &init,
                               const std::vector<Gate>            &circuit,
                               int                                 max_weight,
                               int                                 num_threads)
{
    // ---- load initial observable into a flat vector -----------------------
    // All phases are normalised into the coefficient so pw.phase == 1 always.
    std::vector<std::pair<PauliWord, Complex>> obs;
    obs.reserve(init.size());
    for (auto &[pw, c] : init) {
        PauliWord key = pw;
        Complex   coef = c * pw.phase;
        key.phase = 1.0;
        obs.emplace_back(key, coef);
    }

    // ---- gate loop (backward) --------------------------------------------
    int gi = (int)circuit.size() - 1;

    while (gi >= 0) {
        const Gate &g = circuit[gi];

        if (is_single_output(g)) {
            // ------------------------------------------------------------------
            // Clifford block: find the longest consecutive run of single-output
            // gates starting at gi going backward, then process the ENTIRE
            // block in ONE parallel region.
            //
            // This avoids one OpenMP barrier per gate (which would dominate
            // at high thread counts with short-running gates).
            // ------------------------------------------------------------------
            int block_hi = gi;           // highest gate index in block (latest)
            int block_lo = gi;           // lowest  gate index in block (earliest)
            while (block_lo - 1 >= 0 && is_single_output(circuit[block_lo - 1]))
                --block_lo;
            // block covers circuit[block_lo .. block_hi] (apply hi→lo = backward)

            const int n = (int)obs.size();

            // Each word is completely independent through the whole Clifford block.
            // ONE parallel region, ONE barrier, any number of gates.
            //
            // Weight truncation is enforced gate-by-gate inside the inner loop:
            // if a word's weight exceeds max_weight after any gate, it is killed
            // immediately (coef set to 0 and loop broken).  This exactly matches
            // the sequential implementation which truncates after every gate.
            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int j = 0; j < n; ++j) {
                PauliWord pw  = obs[j].first;
                Complex   coef = obs[j].second;
                // Apply gates in reverse order (backward through the circuit)
                for (int k = block_hi; k >= block_lo; --k) {
                    PauliWord out = apply_gate_conjugation(circuit[k], pw);
                    coef *= out.phase;   // absorb phase into coefficient
                    out.phase = 1.0;
                    pw = std::move(out);
                    // Truncate: kill word if it exceeds max_weight (matches
                    // the per-gate truncation in the sequential implementation).
                    if (pw.weight() > max_weight) {
                        coef = 0.0;
                        break;
                    }
                }
                obs[j] = {std::move(pw), coef};
            }
            // Remove dead words (weight exceeded max_weight or negligible coeff).
            // No merge needed – Clifford bijection keeps surviving keys distinct.
            obs.erase(
                std::remove_if(obs.begin(), obs.end(),
                    [](const std::pair<PauliWord, Complex> &p) {
                        return std::abs(p.second) <= 1e-10;
                    }),
                obs.end());

            gi = block_lo - 1;   // skip the entire block

        } else {
            // ------------------------------------------------------------------
            // Rotation gate:  each word may fan out to 2 terms.
            // Collect into per-thread vectors, sort, then merge-reduce.
            // ------------------------------------------------------------------
            const int n = (int)obs.size();

            // Per-thread flat output buffers – much cheaper than hash maps.
            std::vector<std::vector<std::pair<PauliWord, Complex>>>
                local(num_threads);
            for (auto &b : local)
                b.reserve((n * 2) / num_threads + 64);

            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int j = 0; j < n; ++j) {
                int tid = omp_get_thread_num();
                auto terms =
                    apply_gate_conjugation_multi(g, obs[j].first);
                for (auto &[tr, phase] : terms) {
                    PauliWord key = tr;
                    key.phase = 1.0;
                    local[tid].emplace_back(key, obs[j].second * phase);
                }
            }

            // Collect all thread-local buffers into one flat vector
            std::size_t total = 0;
            for (auto &b : local) total += b.size();
            std::vector<std::pair<PauliWord, Complex>> all;
            all.reserve(total);
            for (auto &b : local)
                all.insert(all.end(),
                            std::make_move_iterator(b.begin()),
                            std::make_move_iterator(b.end()));

            // Sort so equal keys are adjacent
            std::sort(all.begin(), all.end(),
                      [](const auto &a, const auto &b) {
                          return a.first < b.first;
                      });

            // Merge equal keys, filter near-zero terms, apply weight truncation
            obs.clear();
            for (std::size_t i = 0; i < all.size(); ) {
                Complex sum = 0;
                std::size_t k = i;
                while (k < all.size() && all[k].first == all[i].first)
                    sum += all[k++].second;
                if (std::abs(sum) > 1e-10 &&
                    all[i].first.weight() <= max_weight)
                    obs.emplace_back(all[i].first, sum);
                i = k;
            }
            --gi;  // advance past this single rotation gate
        }
    }

    // ---- expectation value on |0…0⟩ --------------------------------------
    Complex exp_val = 0.0;
    for (auto &[pw, c] : obs)
        exp_val += c * compute_expectation(pw);
    return exp_val;
}
