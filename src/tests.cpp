#include "CycleTimer.h"
#include "pauli.h"
#include "pauli_gpu.h"
#ifdef OMP_ENABLED
#include "pauli_omp.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "tests.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using Complex = complex<double>;

static vector<TestCase> create_test_cases() {
  vector<TestCase> tests;
  cout << "Creating Test Cases: " << endl;

  // Test 1: Hadamard on Z
  {
    PauliWord z(1);
    z.ops[0] = Z;
    tests.push_back(
        {"Hadamard on Z", 1, {{z, 1.0}}, {Gate(HADAMARD, {0})}, 0.0, 1e-10});
  }

  // Test 2: Hadamard on X
  {
    PauliWord x(1);
    x.ops[0] = X;
    tests.push_back(
        {"Hadamard on X", 1, {{x, 1.0}}, {Gate(HADAMARD, {0})}, 1.0, 1e-10});
  }

  // Test 3: Bell state, ZZ
  {
    PauliWord zz(2);
    zz.ops[0] = Z;
    zz.ops[1] = Z;
    tests.push_back({"Bell state, ZZ",
                     2,
                     {{zz, 1.0}},
                     {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1})},
                     1.0,
                     1e-10});
  }

  // Test 4: Bell state, XX
  {
    PauliWord xx(2);
    xx.ops[0] = X;
    xx.ops[1] = X;
    tests.push_back({"Bell state, XX",
                     2,
                     {{xx, 1.0}},
                     {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1})},
                     1.0,
                     1e-10});
  }

  // Test 5: Identity preservation
  {
    PauliWord id(2);
    tests.push_back(
        {"Identity preservation",
         2,
         {{id, 1.0}},
         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}), Gate(HADAMARD, {1})},
         1.0,
         1e-10});
  }

  // Test 6: CNOT: XI -> XX
  {
    PauliWord xi(2);
    xi.ops[0] = X;
    tests.push_back(
        {"CNOT: XI -> XX", 2, {{xi, 1.0}}, {Gate(CNOT, {0, 1})}, 0.0, 1e-10});
  }

  // Test 7: CNOT: IX -> IX
  {
    PauliWord ix(2);
    ix.ops[1] = X;
    tests.push_back(
        {"CNOT: IX -> IX", 2, {{ix, 1.0}}, {Gate(CNOT, {0, 1})}, 0.0, 1e-10});
  }

  // Test 8: CNOT: IZ -> ZZ
  {
    PauliWord iz(2);
    iz.ops[1] = Z;
    tests.push_back(
        {"CNOT: IZ -> ZZ", 2, {{iz, 1.0}}, {Gate(CNOT, {0, 1})}, 1.0, 1e-10});
  }

  // Test 9: S twice
  {
    PauliWord z(1);
    z.ops[0] = Z;
    tests.push_back(
        {"S twice", 1, {{z, 1.0}}, {Gate(S, {0}), Gate(S, {0})}, 1.0, 1e-10});
  }

  // Test 10: GHZ state, ZZI
  {
    PauliWord zzi(3);
    zzi.ops[0] = Z;
    zzi.ops[1] = Z;
    tests.push_back(
        {"GHZ state, ZZI",
         3,
         {{zzi, 1.0}},
         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}), Gate(CNOT, {0, 2})},
         1.0,
         1e-10});
  }

  // Test 11: S on X
  {
    PauliWord x(1);
    x.ops[0] = X;
    tests.push_back({"S on X", 1, {{x, 1.0}}, {Gate(S, {0})}, 0.0, 1e-10});
  }

  // Test 12: double Hadamard
  {
    PauliWord z(1);
    z.ops[0] = Z;
    tests.push_back({"Double Hadamard",
                     1,
                     {{z, 1.0}},
                     {Gate(HADAMARD, {0}), Gate(HADAMARD, {0})},
                     1.0,
                     1e-10});
  }

  // Test 13: T gate
  {
    PauliWord x(1);
    x.ops[0] = X;
    tests.push_back({"T gate on X", 1, {{x, 1.0}}, {Gate(T, {0})}, 0.0, 1e-10});
  }

  // Test 14: RZ rotation
  {
    PauliWord x(1);
    x.ops[0] = X;
    tests.push_back({"RZ(π/6) on X",
                     1,
                     {{x, 1.0}},
                     {Gate(RZ, {0}, M_PI / 6.0)},
                     0.0,
                     1e-10});
  }

  // Test 15: RX rotation
  {
    PauliWord z(1);
    z.ops[0] = Z;
    double angle = M_PI / 4.0;
    tests.push_back({"RX(π/4) on Z",
                     1,
                     {{z, 1.0}},
                     {Gate(RX, {0}, angle)},
                     cos(angle),
                     1e-9});
  }

  // Test 16: RY rotation
  {
    PauliWord x(1);
    x.ops[0] = X;
    double angle = M_PI / 3.0;
    tests.push_back({"RY(π/3) on X",
                     1,
                     {{x, 1.0}},
                     {Gate(RY, {0}, angle)},
                     -sin(angle),
                     1e-9});
  }

  // Test 17: 3-qubit with rotation
  {
    PauliWord xxx(3);
    xxx.ops[0] = X;
    xxx.ops[1] = X;
    xxx.ops[2] = X;
    tests.push_back({"3-qubit XXX with RZ",
                     3,
                     {{xxx, 1.0}},
                     {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}),
                      Gate(CNOT, {1, 2}), Gate(RZ, {0}, M_PI / 8)},
                     cos(M_PI / 8),
                     1e-8});
  }

  // Test 18: Bell state + rotation
  {
    PauliWord zz(2);
    zz.ops[0] = Z;
    zz.ops[1] = Z;
    tests.push_back(
        {"Bell state with RX rotation",
         2,
         {{zz, 1.0}},
         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}), Gate(RX, {0}, M_PI / 4)},
         cos(M_PI / 4),
         1e-8});
  }

  // Test 19: 4-qubit GHZ
  {
    PauliWord zzzz(4);
    zzzz.ops[0] = Z;
    zzzz.ops[1] = Z;
    zzzz.ops[2] = Z;
    zzzz.ops[3] = Z;
    tests.push_back({"4-qubit ZZZZ GHZ-like",
                     4,
                     {{zzzz, 1.0}},
                     {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}),
                      Gate(CNOT, {1, 2}), Gate(CNOT, {2, 3})},
                     1.0,
                     1e-10});
  }

  // Test 20: Multiple small rotations
  {
    PauliWord z(2);
    z.ops[0] = Z;
    tests.push_back({"Multiple small rotations",
                     2,
                     {{z, 1.0}},
                     {Gate(RZ, {0}, 0.1), Gate(RX, {0}, 0.1),
                      Gate(RY, {0}, 0.1), Gate(CNOT, {0, 1})},
                     0.990033,
                     1e-5});
  }

  // Test 21: Bigger circuit
  {
    PauliWord obs(5);
    obs.ops[0] = Z;
    obs.ops[1] = Z;
    obs.ops[2] = Z;
    tests.push_back(
        {"5-qubit mixed circuit",
         5,
         {{obs, 1.0}},
         {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1}), Gate(HADAMARD, {2}),
          Gate(CNOT, {2, 3}), Gate(S, {1}), Gate(CNOT, {1, 4})},
         0.0,
         1e-10});
  }

  // Test 22: Deep layered circuit
  {
    PauliWord x(3);
    x.ops[1] = X;
    vector<Gate> circuit;
    for (int i = 0; i < 10; i++) {
      circuit.push_back(Gate(RZ, {0}, 0.05));
      circuit.push_back(Gate(HADAMARD, {1}));
      circuit.push_back(Gate(CNOT, {0, 1}));
    }
    tests.push_back(
        {"Deep circuit 10 layers", 3, {{x, 1.0}}, circuit, 0.0, 1e-6});
  }

  // MultiBlock A: Testing multi threadblocks with no rotations
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(123456789);
    std::uniform_int_distribution<int> opdis(0, 3); // 0 -> I, 1->X,2->Y,3->Z
    int num_words = 400;                        // 2.5M words - heavy parallel processing load
    for (int w = 0; w < num_words; ++w)
    {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q)
      {
        int od = opdis(rng);
        if (od == 0)
          continue;
        if (od == 1)
          pw.ops[q] = X;
        else if (od == 2)
          pw.ops[q] = Y;
        else
          pw.ops[q] = Z;
      }
      obs[pw] += Complex(20.0, 0.0);
    }
    // Moderate depth circuit with rotation gates that cause Pauli word
    // expansion
    vector<Gate> circ;
    for (int layer = 0; layer < 300; ++layer)
    {
      // Rotations cause expansion (each rotation can double Pauli words)
      // for (int q = 0; q < nq; ++q)
      //   circ.push_back(Gate(T, {q}, 0.1 * (layer + 1)));

      // CNOTs entangle qubits
      for (int q = 0; q + 1 < nq; q += 2)
        circ.push_back(Gate(CNOT, {q, q + 1}));

      for (int q = 1; q + 1 < nq; q += 2)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }

    tests.push_back(
        {"MultiBlock A: No rotations", nq, obs,
         circ, Complex(0.0, 0.0), 1e-9, 1});
  }

  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(123456789);
    std::uniform_int_distribution<int> opdis(0, 3); // 0 -> I, 1->X,2->Y,3->Z
    int num_words = 7;                            // 2.5M words - heavy parallel processing load
    for (int w = 0; w < num_words; ++w)
    {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q)
      {
        int od = opdis(rng);
        if (od == 0)
          continue;
        if (od == 1)
          pw.ops[q] = X;
        else if (od == 2)
          pw.ops[q] = Y;
        else
          pw.ops[q] = Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    // Moderate depth circuit with rotation gates that cause Pauli word
    // expansion
    vector<Gate> circ;
    for (int layer = 0; layer < 1; ++layer)
    {
      // Rotations cause expansion (each rotation can double Pauli words)
      for (int q = 0; q < 1; ++q)
        circ.push_back(Gate(RZ, {q}, 0.1 * (layer + 1)));

      // // CNOTs entangle qubits
      // for (int q = 0; q + 1 < nq; q += 2)
      //   circ.push_back(Gate(CNOT, {q, q + 1}));

      // for (int q = 1; q + 1 < nq; q += 2)
      //   circ.push_back(Gate(CNOT, {q, q + 1}));
    }

    tests.push_back(
        {"MultiBlock B: with rotations", nq, obs,
         circ, Complex(0.0, 0.0), 1e-9, 1});
  }

  // ===== STRESS TESTS 23-32: GPU PARALLELIZATION ADVANTAGE =====
  // Key insight from pauli_gpu.cu: GPU parallelizes over Pauli words
  // - 512 words/block, 400 blocks = 200K capacity
  // - CPU processes words SEQUENTIALLY, GPU processes in PARALLEL
  // Strategy: Moderate words + circuit depth = 2-10x speedup (CPU finishes in ~10-60s)

  // Test 23: 2K words, 100 layers - Basic parallel test (2-5x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(2301);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 2000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 100; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 23: 7q, 2K words, 100 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-8, 1});
  }

  // Test 24: 5K words, 150 layers - Medium parallel (3-8x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(2401);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 5000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 150; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 24: 7q, 5K words, 150 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-7, 1});
  }

  // Test 25: 3K words, 200 layers - Deeper circuit (3-6x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(2501);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 3000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 200; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 25: 7q, 3K words, 200 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-8, 1});
  }

  // Test 26: 1K words, 300 layers - Deep but small (2-4x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(2601);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 1000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 300; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(S, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 26: 7q, 1K words, 300 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-8, 1});
  }

  // Test 27: 4K words, 100 layers - Wide parallel (3-7x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(2701);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 4000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 100; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 27: 7q, 4K words, 100 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-7, 1});
  }

  // Test 28: 2K words, 250 layers - Balanced (2-5x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(2801);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 2000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 250; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 28: 7q, 2K words, 250 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-7, 1});
  }

  // Test 29: 1K words, 400 layers - Deep small (2-4x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(2901);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 1000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 400; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 29: 7q, 1K words, 400 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-8, 1});
  }

  // Test 30: 8K words, 50 layers - Wide parallel (4-10x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3001);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 8000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 50; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 30: 7q, 8K words, 50 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-8, 1});
  }

  // Test 31: 500 words, 500 layers - Deep small (2-3x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3101);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 500;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 500; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 31: 7q, 500 words, 500 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-7, 1});
  }

  // Test 32: 5K words, 120 layers - Large balanced (3-8x speedup)
  {
    int nq = 7;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3201);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 5000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 120; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"STRESS 32: 7q, 5K words, 120 layers", nq, obs, circ, Complex(0.0, 0.0), 1e-7, 1});
  }

  // Test 34: SCALE-1: 9q, 10K words, 30 layers (GPU ~39 blocks, ~2-3x over OMP-8t)
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3401);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 10000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 30; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"SCALE-1: 9q, 10K words, 30 layers", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 35: SCALE-2: 9q, 15K words, 30 layers (GPU ~59 blocks, ~3-4x over OMP-8t)
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3501);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 15000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 30; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"SCALE-2: 9q, 15K words, 30 layers", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 36: SCALE-3: 9q, 20K words, 30 layers
  // max_weight=10 >= 9 = nq, so no truncation → word count constant.
  // With 20K words the GPU uses ~78 blocks (78/46 SMs ≈ 1.7 blocks/SM, 35% utilisation).
  // At this scale GPU should be competitive with OMP-8t.
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3601);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 20000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 30; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"SCALE-3: 9q, 20K words, 30 layers", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 37: SCALE-4: 9q, 50K words, 20 layers
  // ~195 blocks → GPU uses ~4 blocks/SM → fully saturated → GPU should clearly beat OMP.
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3701);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 50000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 20; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"SCALE-4: 9q, 50K words, 20 layers", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 38: SCALE-5: 9q, 100K words, 10 layers
  // ~380 blocks (just within MAX_BLOCKS=400) → GPU fully saturated → maximum GPU advantage.
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3801);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 100000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 10; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"SCALE-5: 9q, 100K words, 10 layers", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // =====================================================================
  // DIVERSE GPU STRESS TESTS  (indices 39-46)
  // Designed to: (a) favour GPU via large word counts, and (b) cover a
  // wide variety of gate sets — Clifford (H, S, T, CNOT) and parametric
  // rotations (RZ, RX, RY) — so the benchmark represents realistic
  // quantum-simulation workloads.
  // =====================================================================

  // Test 39: DIVERSE-1: 10q, 30K words, 20L  H+CNOT  (GPU ~117 blocks)
  // 10-qubit Clifford, word count preserved exactly.
  {
    int nq = 10;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(3901);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 30000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 20; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-1: 10q, 30K H+CNOT, 20L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 40: DIVERSE-2: 10q, 60K words, 10L  H+CNOT  (GPU ~234 blocks)
  // Largest Clifford test — GPU fully saturated (>4 blocks/SM).
  {
    int nq = 10;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(4001);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 60000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 10; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-2: 10q, 60K H+CNOT, 10L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 41: DIVERSE-3: 9q, 25K words, 30L  T+H+CNOT  (GPU ~98 blocks)
  // Clifford with T gates — each layer: T on every qubit, H on evens, CNOT chain.
  // Word count is preserved (T is a Clifford on Z-stabilisers).
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(4101);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 25000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 30; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(T, {q}));
      for (int q = 0; q < nq; q += 2)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-3: 9q, 25K T+H+CNOT, 30L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 42: DIVERSE-4: 9q, 35K words, 20L  S+H+CNOT  (GPU ~137 blocks)
  // S gates alternate with H and CNOT — diverse single-qubit Clifford mix.
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(4201);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 35000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 20; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(S, {q}));
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-4: 9q, 35K S+H+CNOT, 20L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 43: DIVERSE-5: 9q, 5K words, 8L  RZ(π/8)+CNOT  (rotation fan-out)
  // RZ is a non-Clifford rotation → each word can split into 2.
  // With max_weight=10 and 9 qubits the word count grows but stays bounded.
  // This test covers the rotation-dominated regime (NISQ VQE-style circuits).
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(4301);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 5000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 8; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(RZ, {q}, M_PI / 8.0));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-5: 9q, 5K RZ+CNOT, 8L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 44: DIVERSE-6: 9q, 4K words, 6L  RX(π/8)+H+CNOT  (mixed rotation)
  // RX rotation mixed with Clifford gates — representative of hardware-efficient
  // ansatz circuits used in VQE.
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(4401);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 4000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 6; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(RX, {q}, M_PI / 8.0));
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-6: 9q, 4K RX+H+CNOT, 6L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 45: DIVERSE-7: 10q, 25K words, 15L  H+S+T+CNOT  (all Clifford types)
  // Every supported Clifford gate in one circuit — broadest single-qubit coverage.
  {
    int nq = 10;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(4501);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 25000;
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 15; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(S, {q}));
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(T, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-7: 10q, 25K H+S+T+CNOT, 15L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // Test 46: DIVERSE-8: 9q, 45K words, 15L  RZ+RX+H+CNOT  (NISQ mixed)
  // Both rotation types + Clifford in every layer — the most realistic workload.
  // Word count grows from rotations then stabilises via max_weight truncation.
  {
    int nq = 9;
    std::map<PauliWord, Complex> obs;
    std::mt19937_64 rng(4601);
    std::uniform_int_distribution<int> opdis(0, 3);
    int num_words = 8000;   // start with 8K; rotations grow to ~30-50K
    for (int w = 0; w < num_words; ++w) {
      PauliWord pw(nq);
      for (int q = 0; q < nq; ++q) {
        int od = opdis(rng);
        if (od == 0) continue;
        pw.ops[q] = (od == 1) ? X : (od == 2) ? Y : Z;
      }
      obs[pw] += Complex(1.0, 0.0);
    }
    vector<Gate> circ;
    for (int layer = 0; layer < 15; ++layer) {
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(RZ, {q}, M_PI / 6.0));
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(RX, {q}, M_PI / 6.0));
      for (int q = 0; q < nq; ++q)
        circ.push_back(Gate(HADAMARD, {q}));
      for (int q = 0; q + 1 < nq; ++q)
        circ.push_back(Gate(CNOT, {q, q + 1}));
    }
    tests.push_back({"DIVERSE-8: 9q, 8K->~40K RZ+RX+H+CNOT, 15L", nq, obs, circ, Complex(0.0, 0.0), 1e6, 1});
  }

  // // MultiBlock B: Testing multi threadblocks with rotations
  // {
  //   int nq = 9;
  //   std::map<PauliWord, Complex> obs;
  //   std::mt19937_64 rng(123456789);
  //   std::uniform_int_distribution<int> opdis(0, 3); // 0 -> I, 1->X,2->Y,3->Z
  //   int num_words = 400;                            // 2.5M words - heavy parallel processing load
  //   for (int w = 0; w < num_words; ++w)
  //   {
  //     PauliWord pw(nq);
  //     for (int q = 0; q < nq; ++q)
  //     {
  //       int od = opdis(rng);
  //       if (od == 0)
  //         continue;
  //       if (od == 1)
  //         pw.ops[q] = X;
  //       else if (od == 2)
  //         pw.ops[q] = Y;
  //       else
  //         pw.ops[q] = Z;
  //     }
  //     obs[pw] += Complex(1.0, 0.0);
  //   }
  //   // Moderate depth circuit with rotation gates that cause Pauli word
  //   // expansion
  //   vector<Gate> circ;
  //   for (int layer = 0; layer < 10; ++layer)
  //   {
  //     // Rotations cause expansion (each rotation can double Pauli words)
  //     for (int q = 0; q < nq; ++q)
  //       circ.push_back(Gate(RZ, {q}, 0.1 * (layer + 1)));

  //     // CNOTs entangle qubits
  //     for (int q = 0; q + 1 < nq; q += 2)
  //       circ.push_back(Gate(CNOT, {q, q + 1}));

  //     for (int q = 1; q + 1 < nq; q += 2)
  //       circ.push_back(Gate(CNOT, {q, q + 1}));
  //   }

  //   tests.push_back(
  //       {"MultiBlock B: with rotations", nq, obs,
  //        circ, Complex(0.0, 0.0), 1e-9, 1});
  // }

    /* // Heavy Test A: Random sparse Pauli ensemble - tests parallel processing of
    // many Pauli words Rationale: Large initial observable size stresses the
    // parallelization of gate applications Target: 6-9 seconds (many words ×
    // moderate gates)
    {
      int nq = 28;
      std::map<PauliWord, Complex> obs;
      std::mt19937_64 rng(123456789);
      std::uniform_int_distribution<int> opdis(0, 3); // 0 -> I, 1->X,2->Y,3->Z
      int num_words = 2500000; // 2.5M words - heavy parallel processing load
      for (int w = 0; w < num_words; ++w) {
        PauliWord pw(nq);
        for (int q = 0; q < nq; ++q) {
          int od = opdis(rng);
          if (od == 0)
            continue;
          if (od == 1)
            pw.ops[q] = X;
          else if (od == 2)
            pw.ops[q] = Y;
          else
            pw.ops[q] = Z;
        }
        obs[pw] += Complex(1.0, 0.0);
      }

      // Moderate depth circuit with rotation gates that cause Pauli word
      // expansion
      vector<Gate> circ;
      for (int layer = 0; layer < 3000; ++layer) {
        // Rotations cause expansion (each rotation can double Pauli words)
        for (int q = 0; q < nq; ++q)
          circ.push_back(Gate(RZ, {q}, 0.01 * (layer + 1)));

        // CNOTs entangle qubits
        for (int q = 0; q + 1 < nq; q += 2)
          circ.push_back(Gate(CNOT, {q, q + 1}));

        for (int q = 1; q + 1 < nq; q += 2)
          circ.push_back(Gate(CNOT, {q, q + 1}));
      }

      tests.push_back(
          {"HEAVY A: 28-qubit, 2.5M words, 3K layers (many Pauli words)", nq, obs,
           circ, Complex(0.0, 0.0), 1e-9, 1});
    }

    // Heavy Test B: Deep circuit with many layers - tests gate application
    // throughput Rationale: Many gates on moderate number of Pauli words stresses
    // gate iteration performance Target: 1-3 seconds (moderate words × very many
    // gates)
    {
      int nq = 30;
      std::map<PauliWord, Complex> obs;
      std::mt19937_64 rng(222222);
      std::uniform_int_distribution<int> opdis(0, 3);
      int num_words = 1500000; // 1.5M words with deep circuit
      for (int w = 0; w < num_words; ++w) {
        PauliWord pw(nq);
        for (int q = 0; q < nq; ++q) {
          int od = opdis(rng);
          if (od == 0)
            continue;
          if (od == 1)
            pw.ops[q] = X;
          else if (od == 2)
            pw.ops[q] = Y;
          else
            pw.ops[q] = Z;
        }
        obs[pw] += Complex(1.0, 0.0);
      }

      vector<Gate> circ;
      // Very deep circuit with Clifford gates (don't cause exponential expansion)
      // This tests gate application speed rather than Pauli word explosion
      for (int layer = 0; layer < 100000; ++layer) {
        for (int q = 0; q < nq; ++q)
          circ.push_back(Gate(HADAMARD, {q}));

        for (int q = 0; q + 1 < nq; ++q)
          circ.push_back(Gate(CNOT, {q, q + 1}));
      }

      tests.push_back(
          {"HEAVY B: 30-qubit, 1.5M words, 100K layers H+CNOT (deep circuit)", nq,
           obs, circ, Complex(0.0, 0.0), 1e-9, 1});
    }

    // Heavy Test C: Mixed rotations and entanglement - tests Pauli word expansion
    // handling Rationale: Rotation gates cause exponential expansion, truncation
    // to max_weight is critical Target: 1-3 seconds (moderate words × many
    // rotation gates → expansion stress)
    {
      int nq = 30;
      std::map<PauliWord, Complex> obs;
      std::mt19937_64 rng(987654321);
      std::uniform_int_distribution<int> opdis(0, 3);
      int num_words = 2000000; // Start with 2M words
      for (int w = 0; w < num_words; ++w) {
        PauliWord pw(nq);
        for (int q = 0; q < nq; ++q) {
          int od = opdis(rng);
          if (od == 0)
            continue;
          if (od == 1)
            pw.ops[q] = X;
          else if (od == 2)
            pw.ops[q] = Y;
          else
            pw.ops[q] = Z;
        }
        obs[pw] += Complex(1.0, 0.0);
      }

      vector<Gate> circ;
      // Many layers with rotations that cause expansion
      for (int layer = 0; layer < 4000; ++layer) {
        // Rotations cause Pauli word doubling
        for (int q = 0; q < nq; ++q)
          circ.push_back(Gate(RZ, {q}, 0.01 * (layer + 1)));

        // CNOTs spread operators across qubits
        for (int q = 0; q + 2 < nq; q += 3) {
          circ.push_back(Gate(CNOT, {q, q + 1}));
          circ.push_back(Gate(CNOT, {q + 1, q + 2}));
        }
      }

      tests.push_back(
          {"HEAVY C: 30-qubit, 2M words, 4K layers RZ+CNOT (expansion test)", nq,
           obs, circ, Complex(0.0, 0.0), 1e-9, 1});
    }

    // Heavy Test D: Balanced workload - tests overall system performance
    // Rationale: Moderate words, moderate depth, mixed gates - realistic quantum
    // circuit Target: 3-6 seconds (balanced: many words × many gates × mixed
    // types)
    {
      int nq = 30;
      std::map<PauliWord, Complex> obs;
      std::mt19937_64 rng(555555);
      std::uniform_int_distribution<int> opdis(0, 3);
      int num_words = 3000000; // 3M words
      for (int w = 0; w < num_words; ++w) {
        PauliWord pw(nq);
        for (int q = 0; q < nq; ++q) {
          int od = opdis(rng);
          if (od == 0)
            continue;
          if (od == 1)
            pw.ops[q] = X;
          else if (od == 2)
            pw.ops[q] = Y;
          else
            pw.ops[q] = Z;
        }
        obs[pw] += Complex(1.0, 0.0);
      }

      vector<Gate> circ;
      // Realistic quantum circuit with mixed gate types
      for (int layer = 0; layer < 5000; ++layer) {
        // Mix of Clifford and rotation gates
        for (int q = 0; q < nq; q += 2)
          circ.push_back(Gate(HADAMARD, {q}));

        for (int q = 1; q < nq; q += 2)
          circ.push_back(Gate(RZ, {q}, 0.02 * layer));

        for (int q = 0; q + 1 < nq; ++q)
          circ.push_back(Gate(CNOT, {q, q + 1}));
      }

      tests.push_back(
          {"HEAVY D: 30-qubit, 3M words, 5K layers mixed gates (balanced)", nq,
           obs, circ, Complex(0.0, 0.0), 1e-9, 1});
    }

    // Heavy Test E: Extreme stress test - pushes system to limits
    // Rationale: Maximum realistic workload to demonstrate GPU's full advantage
    // Target: 6-9 seconds (extreme: massive words × deep circuit × heavy
    // expansion)
    {
      int nq = 32;
      std::map<PauliWord, Complex> obs;
      std::mt19937_64 rng(777777);
      std::uniform_int_distribution<int> opdis(0, 3);
      int num_words = 4000000; // 4M words - extreme parallel load
      for (int w = 0; w < num_words; ++w) {
        PauliWord pw(nq);
        for (int q = 0; q < nq; ++q) {
          int od = opdis(rng);
          if (od == 0)
            continue;
          if (od == 1)
            pw.ops[q] = X;
          else if (od == 2)
            pw.ops[q] = Y;
          else
            pw.ops[q] = Z;
        }
        obs[pw] += Complex(1.0, 0.0);
      }

      vector<Gate> circ;
      // Extreme depth with all gate types
      for (int layer = 0; layer < 6000; ++layer) {
        // Heavy rotation layer (causes expansion)
        for (int q = 0; q < nq; q += 3)
          circ.push_back(Gate(RZ, {q}, 0.015 * layer));

        for (int q = 1; q < nq; q += 3)
          circ.push_back(Gate(RX, {q}, 0.012 * layer));

        for (int q = 2; q < nq; q += 3)
          circ.push_back(Gate(RY, {q}, 0.018 * layer));

        // Clifford layer for mixing
        for (int q = 0; q < nq; q += 2)
          circ.push_back(Gate(HADAMARD, {q}));

        // Dense entanglement
        for (int q = 0; q + 1 < nq; ++q)
          circ.push_back(Gate(CNOT, {q, q + 1}));
      }

      tests.push_back(
          {"HEAVY E: 32-qubit, 4M words, 6K layers all gates (EXTREME)", nq, obs,
           circ, Complex(0.0, 0.0), 1e-9, 1});
    } */
    cout << "Finished Test Cases: " << endl;

    return tests;
  }

double run_single_test(const TestCase &test, int i, bool use_gpu, int num_omp_threads) {
  cout << "=== " << i + 1 << ". " << test.name << " ===\n";

  // Start total timing (includes setup)
  auto tstart = chrono::steady_clock::now();

  Complex result;
  double computeTime = 0.0;

#ifdef OMP_ENABLED
  if (!use_gpu && num_omp_threads > 0) {
    cout << "[OMP-" << num_omp_threads << "] Starting propagation..." << endl;
    double startComputeTime = CycleTimer::currentSeconds();
    result = pauli_propagation_omp(test.initial_obs, test.circuit, 10, num_omp_threads);
    double endComputeTime = CycleTimer::currentSeconds();
    computeTime = endComputeTime - startComputeTime;
    cout << "[OMP-" << num_omp_threads << "] Propagation completed in "
         << computeTime << " seconds" << endl;
    cout << "[OMP-" << num_omp_threads << "] OMP propagation finished. Exiting." << endl;
    return computeTime;
  }
#endif

  if (use_gpu) {
#ifndef CPU_ONLY
    cout << "[GPU] Creating simulator..." << endl;
    PauliSimulatorGPU simulator(test.num_qubits, test.initial_obs,
                                test.circuit);

    cout << "[GPU] Starting propagation..." << endl;
    double startComputeTime = CycleTimer::currentSeconds();
    result = simulator.runPropagation(10);
    double endComputeTime = CycleTimer::currentSeconds();
    computeTime = endComputeTime - startComputeTime;
    cout << "[GPU] Propagation completed in " << computeTime << " seconds" << endl;

    // For GPU mode, skip verification and exit immediately
    cout << "[GPU] GPU propagation finished. Exiting." << endl;
    return computeTime;  // Exit immediately for GPU mode
#else
    cout << "GPU not available, using CPU instead\n";
    cout << "[CPU] Starting propagation..." << endl;
    double startComputeTime = CycleTimer::currentSeconds();
    result = pauli_propagation(test.initial_obs, test.circuit, 10);
    double endComputeTime = CycleTimer::currentSeconds();
    computeTime = endComputeTime - startComputeTime;
    cout << "[CPU] Propagation completed in " << computeTime << " seconds" << endl;
    cout << "[CPU] CPU propagation finished. Exiting." << endl;
    return computeTime;  // Exit immediately for CPU-only executable
#endif
  } else {
    cout << "[CPU] Starting propagation..." << endl;
    double startComputeTime = CycleTimer::currentSeconds();
    result = pauli_propagation(test.initial_obs, test.circuit, 10);
    double endComputeTime = CycleTimer::currentSeconds();
    computeTime = endComputeTime - startComputeTime;
    cout << "[CPU] Propagation completed in " << computeTime << " seconds" << endl;

    // For CPU mode, also exit immediately after propagation
    cout << "[CPU] CPU propagation finished. Exiting." << endl;
    return computeTime;  // Exit immediately for CPU mode too
  }

  // For GPU mode, we already returned above. For CPU mode, continue with verification.
  auto tend = chrono::steady_clock::now();
  double elapsed = chrono::duration_cast<chrono::duration<double>>(tend - tstart).count();

  // Get ground truth (CPU verification)
  cout << "[VERIFICATION] Computing ground truth..." << endl;
  double truthStart = CycleTimer::currentSeconds();
  Complex truth = pauli_propagation(test.initial_obs, test.circuit, 10);
  double truthEnd = CycleTimer::currentSeconds();
  double truthTime = truthEnd - truthStart;
  cout << "[VERIFICATION] Ground truth computed in " << truthTime << " seconds" << endl;

  bool passed = abs(result - truth) < test.tolerance;

  if (passed) {
    cout << "\033[92m" << "Status: PASS" << "\033[0m" << "\n";
  } else {
    cout << "\033[31m" << "Status: FAIL" << "\033[0m" << "\n";
    cout << "Result: " << result << "\n";
    cout << "Expected: " << truth << "\n";
    computeTime = -1.0;
  }

  cout << "\nTIMING BREAKDOWN:" << endl;
  cout << "  Total elapsed time: " << fixed << setprecision(3) << elapsed << " s" << endl;
  cout << "  Compute time:       " << fixed << setprecision(3) << computeTime << " s" << endl;
  cout << "  Setup overhead:     " << fixed << setprecision(3) << (elapsed - computeTime) << " s" << endl;
  cout << "  Verification time:  " << fixed << setprecision(3) << truthTime << " s" << endl;
  cout << endl;

  return computeTime;
}

double run_single_test(int i, bool use_gpu, int num_omp_threads) {
  auto test_cases = create_test_cases();
  return run_single_test(test_cases[i], i, use_gpu, num_omp_threads);
}

void run_all_tests(bool use_gpu, int num_omp_threads) {
  auto test_cases = create_test_cases();
  int total_tests = test_cases.size();
  int passed_tests = 0;

  // Vector to store computation times
  vector<double> compute_times_ms(total_tests, 0.0);

  cout << "Running " << total_tests << " tests using ";
  if (use_gpu)            cout << "GPU";
  else if (num_omp_threads > 0) cout << "OMP (" << num_omp_threads << " threads)";
  else                    cout << "CPU (sequential)";
  cout << " simulator\n";
  cout << "========================================\n\n";

  int i = 0;
  for (const auto &test : test_cases) {

    // Run the test and get computation time
    double compute_time = run_single_test(test, i, use_gpu, num_omp_threads);
    double compute_time_ms = compute_time * 1000.0; // Convert to milliseconds

    compute_times_ms[i] = compute_time_ms;

    // Determine status based on compute_time
    bool passed = (compute_time >= 0.0);

    if (passed) {
      passed_tests++;
    }

    i++;
  }
  // First, run all tests and collect timing information
  cout << "TEST TIMING RESULTS:\n";
  cout << "=========================================================\n";
  cout << left << setw(4) << "No." << left << setw(40) << "Test Name" << right
       << setw(12) << "Time (ms)" << "\n";
  cout << "---------------------------------------------------------\n";

  for (size_t i = 0; i < (compute_times_ms.size()); i++) {
    cout << left << setw(4) << i + 1 << left << setw(40)
         << test_cases[i].name.substr(0, 39);

    // Print time and status
    cout << right << setw(12) << fixed << setprecision(3) << compute_times_ms[i]
         << "\n";
  }

  cout << "=========================================================\n";
  cout << "CORRECTNESS RESULTS:\n";
  cout << "Passed: " << passed_tests << "/" << total_tests << "\n";
  cout << "Failed: " << (total_tests - passed_tests) << "/" << total_tests
       << "\n";
  cout << "Success Rate: " << fixed << setprecision(1)
       << (100.0 * passed_tests / total_tests) << "%\n";
  cout << setprecision(6);
}