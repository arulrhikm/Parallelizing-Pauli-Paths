#!/usr/bin/env python3
"""
Algorithmic Visualization - Pauli word dynamics, Clifford vs non-Clifford, memory usage
"""

import subprocess
import re
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

def create_clifford_analysis():
    """Create visualization of Clifford vs non-Clifford gate behavior"""
    if not HAS_MATPLOTLIB:
        return
    
    # Conceptual analysis: Clifford gates keep Pauli words constant/linear
    # Non-Clifford gates (rotations) cause exponential growth
    
    depths = np.linspace(1, 50, 50)
    
    # Clifford: linear growth (H, CNOT, S keep words manageable)
    clifford_growth = 1 + 0.05 * depths
    
    # Non-Clifford: exponential growth (rotations double words)
    nonclifford_growth = 1 + np.exp(0.1 * depths) - 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear scale
    ax1.plot(depths, clifford_growth, label='Clifford Gates (Linear)', linewidth=2, color='blue')
    ax1.plot(depths, nonclifford_growth, label='Non-Clifford Gates (Exponential)', linewidth=2, color='red')
    ax1.set_xlabel('Circuit Depth')
    ax1.set_ylabel('Pauli Word Count (normalized)')
    ax1.set_title('Clifford vs Non-Clifford Gate Behavior')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(depths, clifford_growth, label='Clifford Gates', linewidth=2, color='blue')
    ax2.semilogy(depths, nonclifford_growth, label='Non-Clifford Gates', linewidth=2, color='red')
    ax2.set_xlabel('Circuit Depth')
    ax2.set_ylabel('Pauli Word Count (log scale)')
    ax2.set_title('Clifford vs Non-Clifford (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clifford_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: clifford_analysis.png")

def create_memory_analysis():
    """Create memory usage pattern visualization"""
    if not HAS_MATPLOTLIB:
        return
    
    # Test configurations
    test_configs = {
        25: {'words': 30000, 'layers': 500},
        26: {'words': 5000, 'layers': 150},
        27: {'words': 3000, 'layers': 200},
        28: {'words': 1000, 'layers': 300},
        29: {'words': 4000, 'layers': 100},
        30: {'words': 2000, 'layers': 250},
        31: {'words': 1000, 'layers': 400},
        32: {'words': 8000, 'layers': 50},
        33: {'words': 500, 'layers': 500},
        34: {'words': 5000, 'layers': 120}
    }
    
    # Estimate memory usage
    # Each Pauli word: ~(qubits * 8 bytes) for storage + overhead
    qubits = 7  # Most tests use 7 qubits
    bytes_per_word = qubits * 8 + 16  # Rough estimate
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    tests = list(range(25, 35))
    words = [test_configs[t]['words'] for t in tests]
    memory_mb = [(w * bytes_per_word) / (1024 * 1024) for w in words]
    
    # Memory usage by test
    ax1.bar(tests, memory_mb, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Test Number')
    ax1.set_ylabel('Estimated Memory (MB)')
    ax1.set_title('Memory Usage by Test')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Memory vs word count
    ax2.scatter(words, memory_mb, s=100, alpha=0.6, color='green')
    ax2.set_xlabel('Pauli Word Count')
    ax2.set_ylabel('Estimated Memory (MB)')
    ax2.set_title('Memory Usage vs Word Count')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('memory_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: memory_analysis.png")

def create_pauli_evolution():
    """Create Pauli word evolution visualization"""
    if not HAS_MATPLOTLIB:
        return
    
    # Conceptual: Pauli words evolve during circuit execution
    # With Clifford gates: relatively stable
    # With rotations: exponential growth
    
    layers = np.linspace(0, 100, 101)
    
    # Scenario 1: Clifford-only circuit (stable)
    clifford_evolution = 1000 * np.ones_like(layers)  # Constant
    
    # Scenario 2: Mixed circuit (moderate growth)
    mixed_evolution = 1000 * (1 + 0.02 * layers)
    
    # Scenario 3: Rotation-heavy (exponential)
    rotation_evolution = 1000 * np.exp(0.05 * layers)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear scale
    ax1.plot(layers, clifford_evolution, label='Clifford-only', linewidth=2, color='blue')
    ax1.plot(layers, mixed_evolution, label='Mixed gates', linewidth=2, color='orange')
    ax1.plot(layers, rotation_evolution, label='Rotation-heavy', linewidth=2, color='red')
    ax1.set_xlabel('Circuit Layer')
    ax1.set_ylabel('Pauli Word Count')
    ax1.set_title('Pauli Word Evolution During Circuit Execution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(layers, clifford_evolution, label='Clifford-only', linewidth=2, color='blue')
    ax2.semilogy(layers, mixed_evolution, label='Mixed gates', linewidth=2, color='orange')
    ax2.semilogy(layers, rotation_evolution, label='Rotation-heavy', linewidth=2, color='red')
    ax2.set_xlabel('Circuit Layer')
    ax2.set_ylabel('Pauli Word Count (log scale)')
    ax2.set_title('Pauli Word Evolution (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pauli_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: pauli_evolution.png")

def main():
    print("="*60)
    print("ALGORITHMIC VISUALIZATION")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - install with: pip install matplotlib")
        return
    
    print("\nGenerating visualizations...")
    create_clifford_analysis()
    create_memory_analysis()
    create_pauli_evolution()
    
    print("\nGenerated files:")
    print("  - clifford_analysis.png")
    print("  - memory_analysis.png")
    print("  - pauli_evolution.png")

if __name__ == "__main__":
    main()

