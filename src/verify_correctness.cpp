// =============================================================================
// verify_correctness.cpp
// =============================================================================
// Standalone numerical correctness verifier for pauli_propagation_omp (CPU)
// and PauliSimulatorGPU (GPU).
//
// For every test case it computes:
//   truth   = pauli_propagation()          (sequential std::map, CPU)
//   omp_1   = pauli_propagation_omp(…,  1) (1 thread,  unordered_map)
//   omp_4   = pauli_propagation_omp(…,  4) (4 threads)
//   omp_16  = pauli_propagation_omp(…, 16) (16 threads)
//   gpu     = PauliSimulatorGPU::runPropagation()   (only when !CPU_ONLY)
//
// Then checks |result - truth| < tolerance.
//
// ---- Build (from src/) ----
//
// OMP only (Windows/Linux, no CUDA needed):
//   g++ -std=c++17 -O2 -DCPU_ONLY -DOMP_ENABLED -fopenmp -I. \
//       pauli.cpp pauli_omp.cpp verify_correctness.cpp \
//       -o verify_correctness_omp.exe
//
// GPU + OMP (GHC Linux with nvcc):
//   nvcc -std=c++17 -O2 -DOMP_ENABLED -fopenmp -ccbin g++-11 -I. \
//       pauli.cpp pauli_omp.cpp pauli_gpu.cu verify_correctness.cpp \
//       -o verify_correctness_gpu.exe
//
// ---- Run ----
//   ./verify_correctness_omp.exe
//   ./verify_correctness_gpu.exe
// =============================================================================

#include "pauli.h"
#ifdef OMP_ENABLED
#include "pauli_omp.h"
#endif
#ifndef CPU_ONLY
#include "pauli_gpu.h"
#endif

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Complex = std::complex<double>;

// ---- ANSI colours (degrade gracefully on Windows cmd) ----
static const char *GREEN = "\033[92m";
static const char *RED   = "\033[91m";
static const char *RESET = "\033[0m";

// ---- Null stream to silence GPU's verbose cout ----
struct NullBuf : public std::streambuf {
    int overflow(int c) override { return c; }
};
struct NullStream : public std::ostream {
    NullBuf buf;
    NullStream() : std::ostream(&buf) {}
};
static NullStream dev_null;

// =============================================================================
// Test case definition
// =============================================================================
struct Case {
    std::string                  name;
    int                          nq;
    std::map<PauliWord, Complex> obs;
    std::vector<Gate>            circuit;
    double                       tol;        // tolerance for OMP
    double                       gpu_tol;    // tolerance for GPU
    int                          max_weight = 10; // truncation weight (default 10)
};

// =============================================================================
// Test suite
// =============================================================================
static std::vector<Case> build_cases()
{
    std::vector<Case> v;

    // ---- Single-qubit Clifford ----
    {   PauliWord z(1); z.ops[0]=Z;
        v.push_back({"H|Z⟩",  1, {{z,1.0}}, {Gate(HADAMARD,{0})}, 1e-12, 1e-10}); }
    {   PauliWord x(1); x.ops[0]=X;
        v.push_back({"H|X⟩",  1, {{x,1.0}}, {Gate(HADAMARD,{0})}, 1e-12, 1e-10}); }
    {   PauliWord z(1); z.ops[0]=Z;
        v.push_back({"HH|Z⟩", 1, {{z,1.0}},
                     {Gate(HADAMARD,{0}),Gate(HADAMARD,{0})}, 1e-12, 1e-10}); }
    {   PauliWord z(1); z.ops[0]=Z;
        v.push_back({"SS|Z⟩", 1, {{z,1.0}},
                     {Gate(S,{0}),Gate(S,{0})}, 1e-12, 1e-10}); }
    {   PauliWord x(1); x.ops[0]=X;
        v.push_back({"S|X⟩",  1, {{x,1.0}}, {Gate(S,{0})}, 1e-12, 1e-10}); }
    {   PauliWord x(1); x.ops[0]=X;
        v.push_back({"T|X⟩",  1, {{x,1.0}}, {Gate(T,{0})}, 1e-12, 1e-10}); }

    // ---- Single-qubit rotations ----
    {   PauliWord x(1); x.ops[0]=X;
        v.push_back({"RZ(π/6)|X⟩", 1, {{x,1.0}},
                     {Gate(RZ,{0},M_PI/6)}, 1e-12, 1e-8}); }
    {   PauliWord z(1); z.ops[0]=Z;
        v.push_back({"RX(π/4)|Z⟩", 1, {{z,1.0}},
                     {Gate(RX,{0},M_PI/4)}, 1e-12, 1e-8}); }
    {   PauliWord x(1); x.ops[0]=X;
        v.push_back({"RY(π/3)|X⟩", 1, {{x,1.0}},
                     {Gate(RY,{0},M_PI/3)}, 1e-12, 1e-8}); }
    {   PauliWord z(2); z.ops[0]=Z;
        v.push_back({"RZ+RX+RY+CNOT", 2, {{z,1.0}},
                     {Gate(RZ,{0},0.1),Gate(RX,{0},0.1),
                      Gate(RY,{0},0.1),Gate(CNOT,{0,1})}, 1e-12, 1e-8}); }

    // ---- Two-qubit entangling ----
    {   PauliWord zz(2); zz.ops[0]=Z; zz.ops[1]=Z;
        v.push_back({"Bell ZZ",  2, {{zz,1.0}},
                     {Gate(HADAMARD,{0}),Gate(CNOT,{0,1})}, 1e-12, 1e-10}); }
    {   PauliWord xx(2); xx.ops[0]=X; xx.ops[1]=X;
        v.push_back({"Bell XX",  2, {{xx,1.0}},
                     {Gate(HADAMARD,{0}),Gate(CNOT,{0,1})}, 1e-12, 1e-10}); }
    {   PauliWord iz(2); iz.ops[1]=Z;
        v.push_back({"CNOT IZ→ZZ", 2, {{iz,1.0}}, {Gate(CNOT,{0,1})}, 1e-12, 1e-10}); }
    {   PauliWord xi(2); xi.ops[0]=X;
        v.push_back({"CNOT XI→XX", 2, {{xi,1.0}}, {Gate(CNOT,{0,1})}, 1e-12, 1e-10}); }

    // ---- Three-qubit ----
    {   PauliWord zzi(3); zzi.ops[0]=Z; zzi.ops[1]=Z;
        v.push_back({"GHZ ZZI", 3, {{zzi,1.0}},
                     {Gate(HADAMARD,{0}),Gate(CNOT,{0,1}),Gate(CNOT,{0,2})},
                     1e-12, 1e-10}); }
    {   PauliWord xxx(3); for(int i=0;i<3;i++) xxx.ops[i]=X;
        v.push_back({"XXX+RZ", 3, {{xxx,1.0}},
                     {Gate(HADAMARD,{0}),Gate(CNOT,{0,1}),
                      Gate(CNOT,{1,2}),Gate(RZ,{0},M_PI/8)}, 1e-10, 1e-8}); }

    // ---- Four-qubit ----
    {   PauliWord zzzz(4); for(int i=0;i<4;i++) zzzz.ops[i]=Z;
        v.push_back({"4q GHZ ZZZZ", 4, {{zzzz,1.0}},
                     {Gate(HADAMARD,{0}),Gate(CNOT,{0,1}),
                      Gate(CNOT,{1,2}),Gate(CNOT,{2,3})}, 1e-12, 1e-10}); }

    // ---- 5-qubit mixed ----
    {   PauliWord obs(5); obs.ops[0]=Z; obs.ops[1]=Z; obs.ops[2]=Z;
        v.push_back({"5q ZZZ mixed", 5, {{obs,1.0}},
                     {Gate(HADAMARD,{0}),Gate(CNOT,{0,1}),Gate(HADAMARD,{2}),
                      Gate(CNOT,{2,3}),Gate(S,{1}),Gate(CNOT,{1,4})},
                     1e-12, 1e-10}); }

    // ---- 7-qubit: deep Clifford circuit (300 words, 50 layers) ----
    {   int nq = 7;
        std::map<PauliWord,Complex> obs;
        std::mt19937_64 rng(999);
        std::uniform_int_distribution<int> d(0,3);
        for (int w=0; w<300; ++w) {
            PauliWord pw(nq);
            for (int q=0;q<nq;++q) {
                int op=d(rng);
                if(op==1) pw.ops[q]=X;
                else if(op==2) pw.ops[q]=Y;
                else if(op==3) pw.ops[q]=Z;
            }
            obs[pw]+=Complex(1.0,0);
        }
        std::vector<Gate> circ;
        for (int l=0;l<50;++l) {
            for (int q=0;q<nq;++q) circ.push_back(Gate(HADAMARD,{q}));
            for (int q=0;q+1<nq;++q) circ.push_back(Gate(CNOT,{q,q+1}));
        }
        v.push_back({"7q 300w 50L Clifford", nq, obs, circ, 1e-8, 1e-6}); }

    // ---- 7-qubit: rotation circuit with weight-4 truncation ----
    // max_weight=4 caps the live word count to ≤ C(7,4)+…+C(7,0) ≈ 99,
    // keeping intermediate GPU buffer usage well within MAX_BLOCKS=400 limit.
    {   int nq = 7;
        std::map<PauliWord,Complex> obs;
        std::mt19937_64 rng(1234);
        std::uniform_int_distribution<int> d(0,3);
        for (int w=0; w<100; ++w) {
            PauliWord pw(nq);
            for (int q=0;q<nq;++q) {
                int op=d(rng);
                if(op==1) pw.ops[q]=X;
                else if(op==2) pw.ops[q]=Y;
                else if(op==3) pw.ops[q]=Z;
            }
            obs[pw]+=Complex(1.0,0);
        }
        std::vector<Gate> circ;
        for (int l=0;l<20;++l) {
            for (int q=0;q<nq;++q) circ.push_back(Gate(RZ,{q}, 0.05*(l+1)));
            for (int q=0;q+1<nq;++q) circ.push_back(Gate(CNOT,{q,q+1}));
        }
        v.push_back({"7q 100w 20L RZ+CNOT (w≤4)", nq, obs, circ, 1e-8, 1e-6, 4}); }

    // =======================================================================
    // GPU-RIGOROUS TEST BLOCK
    // Each case is designed to probe a distinct aspect of the GPU kernel.
    // All use max_weight values that keep the live word count ≪ 102,400
    // (GPU capacity = MAX_PAULI_WORDS/2 × MAX_BLOCKS = 256 × 400).
    // Clifford-only circuits never expand word counts, so they can use
    // large initial observables comparable to the actual benchmark suite.
    // =======================================================================

    // ---- GPU-1: large-word Clifford (benchmark scale) ----
    // 5 000 random 7-qubit Pauli words, 100 H+CNOT layers, no truncation.
    // Clifford gates permute Pauli words (no splitting), so word count stays
    // ≤ 4^7 = 16 384.  Tests the GPU at the same scale as STRESS 30 (8K words).
    {   int nq = 7;
        std::map<PauliWord,Complex> obs;
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int> d(0,3);
        std::uniform_real_distribution<double> rd(-1.0, 1.0);
        for (int w=0; w<5000; ++w) {
            PauliWord pw(nq);
            for (int q=0;q<nq;++q) {
                int op=d(rng);
                if(op==1) pw.ops[q]=X;
                else if(op==2) pw.ops[q]=Y;
                else if(op==3) pw.ops[q]=Z;
            }
            obs[pw] += Complex(rd(rng), 0.0);
        }
        std::vector<Gate> circ;
        for (int l=0;l<100;++l) {
            for (int q=0;q<nq;++q) circ.push_back(Gate(HADAMARD,{q}));
            for (int q=0;q+1<nq;++q) circ.push_back(Gate(CNOT,{q,q+1}));
        }
        v.push_back({"GPU-1: 7q 5Kw 100L Clifford", nq, obs, circ, 1e-6, 1e-4}); }

    // ---- GPU-2: all rotation gate types (RZ + RX + RY + CNOT) w≤4 ----
    // Tests all non-Clifford gate paths in the GPU kernel together.
    {   int nq = 7;
        std::map<PauliWord,Complex> obs;
        std::mt19937_64 rng(2024);
        std::uniform_int_distribution<int> d(0,3);
        for (int w=0; w<80; ++w) {
            PauliWord pw(nq);
            for (int q=0;q<nq;++q) {
                int op=d(rng);
                if(op==1) pw.ops[q]=X;
                else if(op==2) pw.ops[q]=Y;
                else if(op==3) pw.ops[q]=Z;
            }
            obs[pw] += Complex(1.0, 0.0);
        }
        std::vector<Gate> circ;
        for (int l=0;l<15;++l) {
            for (int q=0;q<nq;++q) circ.push_back(Gate(RZ,{q}, 0.1*(l+1)));
            for (int q=0;q<nq;++q) circ.push_back(Gate(RX,{q}, 0.07*(l+1)));
            for (int q=0;q<nq;++q) circ.push_back(Gate(RY,{q}, 0.13*(l+1)));
            for (int q=0;q+1<nq;++q) circ.push_back(Gate(CNOT,{q,q+1}));
        }
        v.push_back({"GPU-2: 7q all-rotation 15L (w≤4)", nq, obs, circ, 1e-6, 1e-4, 4}); }

    // ---- GPU-3: complex-coefficient observable ----
    // Observable has both real and imaginary coefficients.
    // Tests that the GPU coefficient arithmetic is exact on complex inputs.
    {   int nq = 6;
        std::map<PauliWord,Complex> obs;
        std::mt19937_64 rng(314);
        std::uniform_int_distribution<int> d(0,3);
        std::uniform_real_distribution<double> rc(-1.0, 1.0);
        for (int w=0; w<200; ++w) {
            PauliWord pw(nq);
            for (int q=0;q<nq;++q) {
                int op=d(rng);
                if(op==1) pw.ops[q]=X;
                else if(op==2) pw.ops[q]=Y;
                else if(op==3) pw.ops[q]=Z;
            }
            obs[pw] += Complex(rc(rng), rc(rng));
        }
        std::vector<Gate> circ;
        for (int l=0;l<40;++l) {
            for (int q=0;q<nq;++q) circ.push_back(Gate(HADAMARD,{q}));
            for (int q=0;q<nq;++q) circ.push_back(Gate(S,{q}));
            for (int q=0;q+1<nq;++q) circ.push_back(Gate(CNOT,{q,q+1}));
        }
        v.push_back({"GPU-3: 6q complex-coeff 40L Clifford", nq, obs, circ, 1e-6, 1e-4}); }

    // ---- GPU-4: T-gate (non-Clifford, non-rotation) ----
    // T gate is implemented differently from RZ/RX/RY in the GPU kernel.
    // Verify it matches the CPU reference.
    {   int nq = 7;
        std::map<PauliWord,Complex> obs;
        std::mt19937_64 rng(777);
        std::uniform_int_distribution<int> d(0,3);
        for (int w=0; w<150; ++w) {
            PauliWord pw(nq);
            for (int q=0;q<nq;++q) {
                int op=d(rng);
                if(op==1) pw.ops[q]=X;
                else if(op==2) pw.ops[q]=Y;
                else if(op==3) pw.ops[q]=Z;
            }
            obs[pw] += Complex(1.0, 0.0);
        }
        std::vector<Gate> circ;
        for (int l=0;l<30;++l) {
            for (int q=0;q<nq;++q) circ.push_back(Gate(T,{q}));
            for (int q=0;q+1<nq;++q) circ.push_back(Gate(CNOT,{q,q+1}));
            for (int q=0;q<nq;++q) circ.push_back(Gate(HADAMARD,{q}));
        }
        v.push_back({"GPU-4: 7q T+CNOT+H 30L (w≤5)", nq, obs, circ, 1e-6, 1e-4, 5}); }

    // ---- GPU-5: 9-qubit circuit (near MAX_QUBITS=10) ----
    // Tests that the qubit-dimension code path is correct for larger systems.
    // Clifford-only so no word-count explosion.
    {   int nq = 9;
        std::map<PauliWord,Complex> obs;
        std::mt19937_64 rng(9999);
        std::uniform_int_distribution<int> d(0,3);
        for (int w=0; w<500; ++w) {
            PauliWord pw(nq);
            for (int q=0;q<nq;++q) {
                int op=d(rng);
                if(op==1) pw.ops[q]=X;
                else if(op==2) pw.ops[q]=Y;
                else if(op==3) pw.ops[q]=Z;
            }
            obs[pw] += Complex(1.0, 0.0);
        }
        std::vector<Gate> circ;
        for (int l=0;l<30;++l) {
            for (int q=0;q<nq;++q)   circ.push_back(Gate(HADAMARD,{q}));
            for (int q=0;q+1<nq;++q) circ.push_back(Gate(CNOT,{q,q+1}));
            circ.push_back(Gate(CNOT,{nq-1,0}));  // wrap-around CNOT
        }
        v.push_back({"GPU-5: 9q 500w 30L Clifford", nq, obs, circ, 1e-6, 1e-4}); }

    return v;
}

// =============================================================================
// Helpers
// =============================================================================
static void print_row(const std::string &label, Complex truth, Complex got,
                       double tol, bool &all_ok)
{
    double err = std::abs(got - truth);
    bool ok = (err <= tol);
    all_ok &= ok;

    std::string status = ok
        ? std::string(GREEN) + "PASS" + RESET
        : std::string(RED)   + "FAIL" + RESET;

    std::cout << "    " << std::left << std::setw(14) << label
              << " truth=" << std::scientific << std::setprecision(5)
              << truth.real() << (truth.imag()>=0?"+":"") << truth.imag() << "i"
              << "  got=" << got.real()
              << (got.imag()>=0?"+":"") << got.imag() << "i"
              << "  err=" << std::setprecision(2) << err
              << "  " << status << "\n";
}

// =============================================================================
// Main
// =============================================================================
int main()
{
    auto cases = build_cases();
    int omp_passed=0, omp_total=0;
    int gpu_passed=0, gpu_total=0, gpu_errors=0;

    std::cout << std::fixed;
    std::cout
        << "============================================================\n"
        << "  PAULI PROPAGATION CORRECTNESS VERIFIER\n"
        << "  Reference: pauli_propagation() – sequential std::map\n"
#ifdef OMP_ENABLED
        << "  Testing:   pauli_propagation_omp() at 1, 4, 16 threads\n"
#endif
#ifndef CPU_ONLY
        << "  Testing:   PauliSimulatorGPU::runPropagation()\n"
#endif
        << "============================================================\n\n";

    for (auto &tc : cases) {
        // ---- Reference ----
        Complex truth = pauli_propagation(tc.obs, tc.circuit, tc.max_weight);

        bool case_ok = true;

        std::cout << "--- " << tc.name << " ---\n";
        std::cout << "    truth = " << std::scientific << std::setprecision(6)
                  << truth.real() << (truth.imag()>=0?"+":"") << truth.imag() << "i\n";

        // ---- OMP ----
#ifdef OMP_ENABLED
        for (int j : {1, 4, 16}) {
            Complex got = pauli_propagation_omp(tc.obs, tc.circuit, tc.max_weight, j);
            std::string lbl = "OMP-" + std::to_string(j) + "t";
            print_row(lbl, truth, got, tc.tol, case_ok);
            ++omp_total;
            if (std::abs(got-truth) <= tc.tol) ++omp_passed;
        }
#endif

        // ---- GPU ----
#ifndef CPU_ONLY
        {
            // Silence GPU's verbose cout
            std::streambuf *saved = std::cout.rdbuf(dev_null.rdbuf());
            PauliSimulatorGPU sim(tc.nq, tc.obs, tc.circuit);
            Complex got = sim.runPropagation(tc.max_weight);
            std::cout.rdbuf(saved);

            bool err_result = (got.real() == -1.0 && got.imag() == -1.0);
            if (err_result) {
                std::cout << "    GPU            ERROR (simulator returned (-1,-1))\n";
                case_ok = false;
                ++gpu_errors;   // count errors so summary is not vacuously true
            } else {
                print_row("GPU", truth, got, tc.gpu_tol, case_ok);
                ++gpu_total;
                if (std::abs(got-truth) <= tc.gpu_tol) ++gpu_passed;
            }
        }
#endif

        std::cout << "  => Case " << (case_ok ? std::string(GREEN)+"PASS"+RESET
                                              : std::string(RED)  +"FAIL"+RESET)
                  << "\n\n";
    }

    int total_cases = (int)cases.size();

    std::cout << "============================================================\n"
              << "  SUMMARY (" << total_cases << " test cases)\n";

#ifdef OMP_ENABLED
    std::cout << "  OMP checks : " << omp_passed << "/" << omp_total
              << "  (" << std::fixed << std::setprecision(1)
              << (100.0*omp_passed/omp_total) << "%)\n";
#endif
#ifndef CPU_ONLY
    if (gpu_total > 0) {
        std::cout << "  GPU checks : " << gpu_passed << "/" << gpu_total
                  << "  (" << std::fixed << std::setprecision(1)
                  << (100.0*gpu_passed/gpu_total) << "%)\n";
    } else if (gpu_errors > 0) {
        std::cout << "  GPU checks : 0/" << gpu_errors
                  << " ERRORS  (no CUDA device, or GPU returned (-1,-1) for all cases)\n";
    } else {
        std::cout << "  GPU checks : skipped\n";
    }
#endif

    std::cout << "============================================================\n";

    bool all_pass = (omp_passed == omp_total);
#ifndef CPU_ONLY
    // Fail if there were any GPU errors OR any GPU results were wrong
    all_pass &= (gpu_errors == 0) && (gpu_passed == gpu_total);
#endif
    return all_pass ? 0 : 1;
}
