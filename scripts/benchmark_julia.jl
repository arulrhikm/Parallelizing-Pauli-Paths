#!/usr/bin/env julia
# =============================================================================
# benchmark_julia.jl  –  Task 1.2 external-tool comparison
# =============================================================================
# Benchmarks the full 23-test suite (STRESS 23-32, SCALE 1-5, DIVERSE 1-8)
# using PauliPropagation.jl.
#
# Gate coverage:
#   Clifford  H, S, CNOT    → CliffordGate(:H/:S/:CNOT, ...)   [timed]
#   Non-Clifford T, RZ, RX  → PauliPropagation.jl does not expose these
#                             as simple Clifford gates; tests using them
#                             are marked SKIP below.
#
# Install PauliPropagation.jl first:
#   julia -e 'using Pkg; Pkg.add("PauliPropagation")'
#
# Run:
#   julia scripts/benchmark_julia.jl
# =============================================================================

using Pkg
try
    using PauliPropagation
catch
    @warn "PauliPropagation.jl not found – installing..."
    Pkg.add("PauliPropagation")
    using PauliPropagation
end

using Random, Statistics, Printf

# ---------------------------------------------------------------------------
# Helper: build a random Pauli observable matching tests.cpp seeds + RNG
# ---------------------------------------------------------------------------
const CHAR_TO_SYM = Dict('X' => :X, 'Y' => :Y, 'Z' => :Z)

function make_random_obs(nq::Int, nwords::Int, seed::Int)
    rng = MersenneTwister(seed)
    pauli_chars = ['I', 'X', 'Y', 'Z']

    terms = Dict{String, Float64}()
    for _ in 1:nwords
        ops = join([pauli_chars[rand(rng, 1:4)] for _ in 1:nq])
        terms[ops] = get(terms, ops, 0.0) + 1.0
    end

    obs = PauliSum(nq)
    for (s, c) in terms
        syms  = Symbol[]
        qinds = Int[]
        for (q, ch) in enumerate(s)
            if ch != 'I'
                push!(syms,  CHAR_TO_SYM[ch])
                push!(qinds, q)
            end
        end
        isempty(syms) && continue
        if length(syms) == 1
            pstr = PauliString(nq, syms[1], qinds[1], c)
        else
            pstr = PauliString(nq, syms, qinds, c)
        end
        add!(obs, pstr.term, pstr.coeff)
    end
    return obs
end

# ---------------------------------------------------------------------------
# Helper: build a layered circuit
#
# Supported gate tags:
#   :h_cnot      – H on all qubits, CNOT chain
#   :s_cnot      – S on all qubits, CNOT chain
#   :s_h_cnot    – S then H on all qubits, CNOT chain
# ---------------------------------------------------------------------------
function build_circuit(nq::Int, nlayers::Int, gate_tag::Symbol)
    circuit = Gate[]
    for _ in 1:nlayers
        if gate_tag == :h_cnot
            for q in 1:nq; push!(circuit, CliffordGate(:H, q)); end
        elseif gate_tag == :s_cnot
            for q in 1:nq; push!(circuit, CliffordGate(:S, q)); end
        elseif gate_tag == :s_h_cnot
            for q in 1:nq; push!(circuit, CliffordGate(:S, q)); end
            for q in 1:nq; push!(circuit, CliffordGate(:H, q)); end
        else
            error("Unknown gate tag: $gate_tag")
        end
        for q in 1:(nq - 1)
            push!(circuit, CliffordGate(:CNOT, [q, q + 1]))
        end
    end
    return circuit
end

# ---------------------------------------------------------------------------
# Full 23-test suite (matching tests.cpp + benchmark_qiskit.py exactly)
# ---------------------------------------------------------------------------
struct BenchTest
    name     :: String
    nq       :: Int
    nwords   :: Int
    nlayers  :: Int
    seed     :: Int
    gate_tag :: Symbol     # :h_cnot | :s_cnot | :s_h_cnot | :skip
    idx      :: Int        # 0-based C++ test vector index
end

const SKIP = :skip

ALL_TESTS = [
    # ── STRESS (7-qubit Clifford) ────────────────────────────────────────
    BenchTest("STRESS 23: 7q, 2K words, 100 layers",  7,   2000, 100, 2301, :h_cnot,   24),
    BenchTest("STRESS 24: 7q, 5K words, 150 layers",  7,   5000, 150, 2401, :h_cnot,   25),
    BenchTest("STRESS 25: 7q, 3K words, 200 layers",  7,   3000, 200, 2501, :h_cnot,   26),
    BenchTest("STRESS 26: 7q, 1K words, 300 layers",  7,   1000, 300, 2601, :s_cnot,   27),
    BenchTest("STRESS 27: 7q, 4K words, 100 layers",  7,   4000, 100, 2701, :h_cnot,   28),
    BenchTest("STRESS 28: 7q, 2K words, 250 layers",  7,   2000, 250, 2801, :h_cnot,   29),
    BenchTest("STRESS 29: 7q, 1K words, 400 layers",  7,   1000, 400, 2901, :h_cnot,   30),
    BenchTest("STRESS 30: 7q, 8K words,  50 layers",  7,   8000,  50, 3001, :h_cnot,   31),
    BenchTest("STRESS 31: 7q, 500 words, 500 layers", 7,    500, 500, 3101, :h_cnot,   32),
    BenchTest("STRESS 32: 7q, 5K words, 120 layers",  7,   5000, 120, 3201, :h_cnot,   33),
    # ── SCALE (9-qubit Clifford, large observables) ──────────────────────
    BenchTest("SCALE-1: 9q, 10K words, 30 layers",    9,  10000,  30, 3401, :h_cnot,   34),
    BenchTest("SCALE-2: 9q, 15K words, 30 layers",    9,  15000,  30, 3501, :h_cnot,   35),
    BenchTest("SCALE-3: 9q, 20K words, 30 layers",    9,  20000,  30, 3601, :h_cnot,   36),
    BenchTest("SCALE-4: 9q, 50K words, 20 layers",    9,  50000,  20, 3701, :h_cnot,   37),
    BenchTest("SCALE-5: 9q, 100K words, 10 layers",   9, 100000,  10, 3801, :h_cnot,   38),
    # ── DIVERSE (9-10 qubit, varied gate sets) ───────────────────────────
    BenchTest("DIVERSE-1: 10q, 30K H+CNOT, 20L",     10,  30000,  20, 3901, :h_cnot,   39),
    BenchTest("DIVERSE-2: 10q, 60K H+CNOT, 10L",     10,  60000,  10, 4001, :h_cnot,   40),
    # T gate is NOT in CliffordGate API — treated as rotation by PauliPropagation.jl
    BenchTest("DIVERSE-3: 9q, 25K T+H+CNOT, 30L",    9,  25000,  30, 4101,  SKIP,      41),
    BenchTest("DIVERSE-4: 9q, 35K S+H+CNOT, 20L",    9,  35000,  20, 4201, :s_h_cnot,  42),
    # Rotation gates (RZ, RX) not supported as CliffordGate
    BenchTest("DIVERSE-5: 9q, 5K RZ+CNOT, 8L",       9,   5000,   8, 4301,  SKIP,      43),
    BenchTest("DIVERSE-6: 9q, 4K RX+H+CNOT, 6L",     9,   4000,   6, 4401,  SKIP,      44),
    BenchTest("DIVERSE-7: 10q, 25K H+S+T+CNOT, 15L", 10,  25000,  15, 4501,  SKIP,      45),
    BenchTest("DIVERSE-8: 9q, 8K RZ+RX+H+CNOT, 15L", 9,   8000,  15, 4601,  SKIP,      46),
]

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
MAX_WEIGHT = 10

println("=" ^ 70)
println("  PauliPropagation.jl BENCHMARK (Task 1.2)")
println("  $(Sys.cpu_info()[1].model)")
println("  Julia $(VERSION)")
println("=" ^ 70)
println()

# Warm-up run (trigger JIT compilation before timing)
let
    obs_w  = make_random_obs(3, 10, 42)
    circ_w = build_circuit(3, 2, :h_cnot)
    t0 = time()
    propagate!(circ_w, obs_w; max_weight = MAX_WEIGHT)
    println("  JIT warm-up: $(round(time() - t0, digits=3)) s")
end
println()

timings  = Float64[]
nterms_v = Int[]

println(@sprintf("%-50s  %10s  %8s", "Test", "Time (s)", "#Terms"))
println("-" ^ 72)

for bt in ALL_TESTS
    if bt.gate_tag == SKIP
        push!(timings,  -2.0)
        push!(nterms_v, -1)
        println(@sprintf("%-50s  %10s  %8s", bt.name[1:min(end,49)], "N/A", "SKIP"))
        continue
    end

    obs  = make_random_obs(bt.nq, bt.nwords, bt.seed)
    circ = build_circuit(bt.nq, bt.nlayers, bt.gate_tag)

    # Run twice; report the second (avoids per-run JIT overhead on first test)
    local elapsed = 0.0
    local nterms  = 0
    for _ in 1:2
        obs_copy = copy(obs)
        t0       = time()
        result   = propagate!(circ, obs_copy; max_weight = MAX_WEIGHT)
        elapsed  = time() - t0
        nterms   = length(result)
    end

    push!(timings,  elapsed)
    push!(nterms_v, nterms)
    println(@sprintf("%-50s  %10.4f  %8d", bt.name[1:min(end,49)], elapsed, nterms))
end

println()
println("=" ^ 70)
println("  SUMMARY")
valid = filter(t -> t > 0, timings)
if !isempty(valid)
    @printf "  Mean time (timed tests): %.4f s\n" mean(valid)
    @printf "  Max  time (timed tests): %.4f s\n" maximum(valid)
end
skipped = count(t -> t == -2.0, timings)
println("  Skipped (non-Clifford): $skipped tests")
println()
println("  Compare to scripts/benchmark_results.csv (cpu_seq column)")
println("  Test indices match C++ vector indices 24-46.")
println("=" ^ 70)

# ---------------------------------------------------------------------------
# Write results to scripts/benchmark_julia_results.csv
# ---------------------------------------------------------------------------
using DelimitedFiles: writedlm

results_file = joinpath(@__DIR__, "benchmark_julia_results.csv")
open(results_file, "w") do io
    println(io, "test,test_index,julia_time_s,nterms")
    for (bt, t, nt) in zip(ALL_TESTS, timings, nterms_v)
        println(io, "$(bt.name),$(bt.idx),$(t),$(nt)")
    end
end
println("\n  Results saved to: $results_file")
