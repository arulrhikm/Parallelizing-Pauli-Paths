#!/usr/bin/env python3

# -----------------------------
# Parameter Definitions
# -----------------------------
THREADS_PER_BLOCK = 512
# would like to pump this up to like a thoushand
NUM_QUBITS = 20
# kinda low :(
PAULIWORDS_PER_THREAD = 1
# this is the funkiest parameter, on the one hand want lots of pauli words on the other hand a 
# little misleading as the pauli words will start low and hopeful not grow that quickly
# idk??
SIZE_PAULI = 1  # could be .25 fairly easily (just pack multiple enums into single byte)
SIZE_COMPLEX = 16  # could be 8 at the expense of accurancy
SIZE_PREFIX_IDX = 2  # could probably be 2 if I just changed the exclusiveScan file

# -----------------------------
# Helper: format bytes nicely
# -----------------------------
def fmt_bytes(n):
    if n < 1024:
        return f"{n} B"
    elif n < 1024**2:
        return f"{n} B ({n/1024:.2f} KB)"
    else:
        return f"{n} B ({n/1024**2:.2f} MB)"

# -----------------------------
# Calculations
# -----------------------------
num_pauli_words = THREADS_PER_BLOCK * PAULIWORDS_PER_THREAD

pauli_words = SIZE_PAULI * num_pauli_words * NUM_QUBITS
coeffs = SIZE_COMPLEX * num_pauli_words

prefixInput = SIZE_PREFIX_IDX * THREADS_PER_BLOCK
prefixOutput = SIZE_PREFIX_IDX * THREADS_PER_BLOCK
prefixScratch = 2 * SIZE_PREFIX_IDX * THREADS_PER_BLOCK
prefix = prefixInput + prefixOutput + prefixScratch

total = pauli_words + coeffs + prefix

# -----------------------------
# Output
# -----------------------------
print(f"Number of qubits: {NUM_QUBITS}")
print(f"Pauli words: {num_pauli_words // 2}")
print(f"Total bytes needed: {fmt_bytes(total)}")

print("\nBreakdown:")
print(f"  pauli_words: {fmt_bytes(pauli_words)}")
print(f"  coeffs:      {fmt_bytes(coeffs)}")
print(f"  prefix:      {fmt_bytes(prefix)}")

