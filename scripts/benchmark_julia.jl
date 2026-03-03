#!/usr/bin/env julia
# =============================================================================
# benchmark_julia.jl  –  Task 1.2 external-tool comparison
# =============================================================================
# Benchmarks the same 10 stress-test circuits (matching tests 23-32 in
# tests.cpp) using PauliPropagation.jl.
#
# Install PauliPropagation.jl first:
#   julia -e 'using Pkg; Pkg.add("PauliPropagation")'
#
# Run:
#   julia scripts/benchmark_julia.jl
#
# The script prints timing in the same format as run_benchmark.py so the
# results can be compared directly.
#
# PauliPropagation.jl reference:
#   https://github.com/MSRudolph/PauliPropagation.jl
#   Rudolph et al., "Classical simulations of noisy variational quantum
#   circuits via Pauli propagation" (2023)
# =============================================================================

using Pkg
# Auto-install if missing
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
# ---------------------------------------------------------------------------
function make_random_obs(nq::Int, nwords::Int, seed::Int)
    rng = MersenneTwister(seed)
    pauli_chars = ['I', 'X', 'Y', 'Z']

    # PauliPropagation.jl uses PauliSum of PauliString objects
    # PauliString(nq, paulis::String, coeff=1.0)
    terms = Dict{String, ComplexF64}()
    for _ in 1:nwords
        ops = join([pauli_chars[rand(rng, 1:4)] for _ in 1:nq])
        terms[ops] = get(terms, ops, 0.0) + 1.0
    end

    obs = PauliSum(nq)
    for (s, c) in terms
        add!(obs, c, PauliString(nq, s))
    end
    return obs
end

# ---------------------------------------------------------------------------
# Helper: build a layered circuit of H + CNOT (matches tests 23-32)
# ---------------------------------------------------------------------------
function make_h_cnot_circuit(nq::Int, nlayers::Int)
    circ = []
    for _ in 1:nlayers
        for q in 1:nq
            push!(circ, (gate=:H, qubit=q))
        end
        for q in 1:(nq-1)
            push!(circ, (gate=:CNOT, control=q, target=q+1))
        end
    end
    return circ
end

# Apply circuit using PauliPropagation.jl API
function apply_circuit!(obs, circ, nq)
    for g in circ
        if g.gate == :H
            hadamard!(obs, g.qubit)
        elseif g.gate == :CNOT
            cnot!(obs, g.control, g.target)
        elseif g.gate == :S
            sgate!(obs, g.qubit)
        end
    end
end

# ---------------------------------------------------------------------------
# Stress test definitions (matching tests.cpp exactly)
# ---------------------------------------------------------------------------
struct StressTest
    name::String
    nq::Int
    nwords::Int
    nlayers::Int
    seed::Int
    gate_type::Symbol   # :h_cnot or :s_cnot
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
# Build circuit matching tests.cpp gate patterns
# ---------------------------------------------------------------------------
function build_circuit(st::StressTest)
    nq, nl = st.nq, st.nlayers
    circ = []
    if st.gate_type == :h_cnot
        for _ in 1:nl
            for q in 1:nq
                push!(circ, (gate=:H, qubit=q, control=0, target=0))
            end
            for q in 1:(nq-1)
                push!(circ, (gate=:CNOT, qubit=0, control=q, target=q+1))
            end
        end
    else  # :s_cnot
        for _ in 1:nl
            for q in 1:nq
                push!(circ, (gate=:S, qubit=q, control=0, target=0))
            end
            for q in 1:(nq-1)
                push!(circ, (gate=:CNOT, qubit=0, control=q, target=q+1))
            end
        end
    end
    return circ
end

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

# Warm-up (JIT compile)
let
    obs = make_random_obs(3, 10, 42)
    circ = build_circuit(StressTest("warmup", 3, 10, 2, 42, :h_cnot))
    t0 = time()
    propagate!(obs, circ, max_weight=MAX_WEIGHT)
    println("  JIT warm-up: $(round(time()-t0, digits=3)) s")
end
println()

timings = Float64[]

println(@sprintf("%-45s  %10s", "Test", "Time (s)"))
println("-" ^ 58)

for st in STRESS_TESTS
    obs  = make_random_obs(st.nq, st.nwords, st.seed)
    circ = build_circuit(st)

    # Run twice; take second (avoids first-call overhead)
    for trial in 1:2
        obs_copy = deepcopy(obs)
        t0 = time()
        propagate!(obs_copy, circ, max_weight=MAX_WEIGHT)
        elapsed = time() - t0
        if trial == 2
            push!(timings, elapsed)
            println(@sprintf("%-45s  %10.4f", st.name[1:min(end,44)], elapsed))
        end
    end
end

println()
println("=" ^ 70)
println("  SUMMARY")
println("  Mean time: $(round(mean(timings), digits=4)) s")
println("  Max  time: $(round(maximum(timings), digits=4)) s")
println()
println("  Compare these times to scripts/benchmark_results.csv")
println("  (CPU-seq, OMP, GPU columns) for the same test indices 23-32.")
println("=" ^ 70)
