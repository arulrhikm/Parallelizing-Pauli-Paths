#!/usr/bin/env julia
# =============================================================================
# benchmark_julia.jl  –  Task 1.2 external-tool comparison
# =============================================================================
# Benchmarks the same stress-test circuits (matching tests.cpp) using
# PauliPropagation.jl.
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
# Helper: build a random Pauli observable matching tests.cpp
#
# API (current PauliPropagation.jl):
#   PauliString(nqubits, paulis::Symbol,         qind::Int,          coeff=1.0)
#   PauliString(nqubits, paulis::Vector{Symbol}, qinds::Vector{Int}, coeff=1.0)
#   add!(psum, pstr.term, pstr.coeff)   # add into PauliSum via integer term
# ---------------------------------------------------------------------------
const CHAR_TO_SYM = Dict('X' => :X, 'Y' => :Y, 'Z' => :Z)

function make_random_obs(nq::Int, nwords::Int, seed::Int)
    rng = MersenneTwister(seed)
    pauli_chars = ['I', 'X', 'Y', 'Z']

    # Accumulate coefficients for duplicate strings (matches tests.cpp behavior)
    terms = Dict{String, Float64}()
    for _ in 1:nwords
        ops = join([pauli_chars[rand(rng, 1:4)] for _ in 1:nq])
        terms[ops] = get(terms, ops, 0.0) + 1.0
    end

    obs = PauliSum(nq)
    for (s, c) in terms
        syms   = Symbol[]
        qinds  = Int[]
        for (q, ch) in enumerate(s)
            if ch != 'I'
                push!(syms,  CHAR_TO_SYM[ch])
                push!(qinds, q)
            end
        end
        # Pure identity: skip (contributes a constant, irrelevant for propagation)
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
# Helper: build a layered circuit of H/S + CNOT gates
#
# API: circuit is a Vector{CliffordGate}
#   CliffordGate(:H,    qubit_index)
#   CliffordGate(:S,    qubit_index)
#   CliffordGate(:CNOT, [control, target])
# ---------------------------------------------------------------------------
function build_circuit(nq::Int, nlayers::Int, gate_type::Symbol)
    circuit = Gate[]
    for _ in 1:nlayers
        if gate_type == :h_cnot
            for q in 1:nq
                push!(circuit, CliffordGate(:H, q))
            end
        else   # :s_cnot
            for q in 1:nq
                push!(circuit, CliffordGate(:S, q))
            end
        end
        for q in 1:(nq - 1)
            push!(circuit, CliffordGate(:CNOT, [q, q + 1]))
        end
    end
    return circuit
end

# ---------------------------------------------------------------------------
# Stress test definitions (matching tests.cpp exactly)
# ---------------------------------------------------------------------------
struct StressTest
    name      :: String
    nq        :: Int
    nwords    :: Int
    nlayers   :: Int
    seed      :: Int
    gate_type :: Symbol   # :h_cnot or :s_cnot
end

STRESS_TESTS = [
    StressTest("STRESS 23: 7q, 2K words, 100 layers",  7, 2000, 100, 2301, :h_cnot),
    StressTest("STRESS 24: 7q, 5K words, 150 layers",  7, 5000, 150, 2401, :h_cnot),
    StressTest("STRESS 25: 7q, 3K words, 200 layers",  7, 3000, 200, 2501, :h_cnot),
    StressTest("STRESS 26: 7q, 1K words, 300 layers",  7, 1000, 300, 2601, :s_cnot),
    StressTest("STRESS 27: 7q, 4K words, 100 layers",  7, 4000, 100, 2701, :h_cnot),
    StressTest("STRESS 28: 7q, 2K words, 250 layers",  7, 2000, 250, 2801, :h_cnot),
    StressTest("STRESS 29: 7q, 1K words, 400 layers",  7, 1000, 400, 2901, :h_cnot),
    StressTest("STRESS 30: 7q, 8K words,  50 layers",  7, 8000,  50, 3001, :h_cnot),
    StressTest("STRESS 31: 7q, 500 words, 500 layers", 7,  500, 500, 3101, :h_cnot),
    StressTest("STRESS 32: 7q, 5K words, 120 layers",  7, 5000, 120, 3201, :h_cnot),
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
    # circuit comes FIRST, then the PauliSum
    propagate!(circ_w, obs_w; max_weight = MAX_WEIGHT)
    println("  JIT warm-up: $(round(time() - t0, digits=3)) s")
end
println()

timings = Float64[]

println(@sprintf("%-45s  %10s  %8s", "Test", "Time (s)", "#Terms"))
println("-" ^ 67)

for st in STRESS_TESTS
    obs  = make_random_obs(st.nq, st.nwords, st.seed)
    circ = build_circuit(st.nq, st.nlayers, st.gate_type)

    # Run twice; report the second (avoids per-run JIT overhead)
    local elapsed = 0.0
    local nterms  = 0
    for trial in 1:2
        obs_copy = copy(obs)
        t0       = time()
        # NOTE: propagate! takes (circuit, psum; kwargs...) — circuit first!
        result   = propagate!(circ, obs_copy; max_weight = MAX_WEIGHT)
        elapsed  = time() - t0
        nterms   = length(result)
    end

    push!(timings, elapsed)
    println(@sprintf("%-45s  %10.4f  %8d", st.name[1:min(end, 44)], elapsed, nterms))
end

println()
println("=" ^ 70)
println("  SUMMARY")
@printf "  Mean time: %.4f s\n" mean(timings)
@printf "  Max  time: %.4f s\n" maximum(timings)
println()
println("  Compare these times to scripts/benchmark_results.csv")
println("  (cpu_seq column) for test indices 24-33.")
println("=" ^ 70)
