#include "pauli.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

void diagnostic_propagation(const map<PauliWord, Complex> &init,
                           const vector<Gate> &circuit,
                           int max_weight)
{
    map<PauliWord, Complex> obs = init;
    
    cout << "Initial observable: " << obs.size() << " Pauli words\n";
    print_observable(obs, "Start");
    
    for (int i = (int)circuit.size() - 1; i >= 0; --i)
    {
        const Gate &g = circuit[i];
        map<PauliWord, Complex> updated;

        for (auto &[pw, coeff] : obs)
        {
            map<PauliWord, Complex> transformed_terms = apply_gate_conjugation_multi(g, pw);
            
            for (auto &[transformed, trans_phase] : transformed_terms)
            {
                PauliWord key(transformed.ops.size());
                key.ops = transformed.ops;
                key.phase = 1.0;
                updated[key] += coeff * trans_phase;
            }
        }

        map<PauliWord, Complex> filtered;
        for (auto &[pw, c] : updated)
        {
            if (abs(c) > 1e-10)
                filtered[pw] = c;
        }

        obs = truncate_pauli_words(filtered, max_weight);
        
        string gate_name;
        switch(g.type) {
            case HADAMARD: gate_name = "HADAMARD"; break;
            case CNOT: gate_name = "CNOT"; break;
            case S: gate_name = "S"; break;
            case T: gate_name = "T"; break;
            case RZ: gate_name = "RZ"; break;
            case RX: gate_name = "RX"; break;
            case RY: gate_name = "RY"; break;
            default: gate_name = "UNKNOWN"; break;
        }
        
        cout << "\nAfter gate " << (circuit.size() - i) << " (" << gate_name;
        if (g.type == RZ || g.type == RX || g.type == RY) {
            cout << ", angle=" << g.angle;
        }
        cout << "): " << obs.size() << " Pauli words\n";
        
        if (obs.size() <= 10) {
            print_observable(obs, "Current state");
        } else {
            cout << "(Too many to display)\n";
        }
    }
}

int main()
{
    cout << "========================================\n";
    cout << "   PAULI WORD EXPANSION DIAGNOSTIC\n";
    cout << "========================================\n\n";
    
    // Example 1: Clifford gates only
    cout << "Example 1: Pure Clifford Circuit (H + CNOT)\n";
    cout << "--------------------------------------------\n";
    {
        PauliWord x(2);
        x.ops[0] = X;
        map<PauliWord, Complex> init = {{x, 1.0}};
        vector<Gate> circuit = {Gate(HADAMARD, {0}), Gate(CNOT, {0, 1})};
        diagnostic_propagation(init, circuit, 10);
    }
    
    cout << "\n\n";
    
    // Example 2: Single rotation
    cout << "Example 2: Single RZ Rotation (causes expansion)\n";
    cout << "--------------------------------------------------\n";
    {
        PauliWord x(1);
        x.ops[0] = X;
        map<PauliWord, Complex> init = {{x, 1.0}};
        vector<Gate> circuit = {Gate(RZ, {0}, M_PI/6)};
        diagnostic_propagation(init, circuit, 10);
    }
    
    cout << "\n\n";
    
    // Example 3: Multiple rotations
    cout << "Example 3: Multiple Rotations (exponential expansion)\n";
    cout << "------------------------------------------------------\n";
    {
        PauliWord z(2);
        z.ops[0] = Z;
        map<PauliWord, Complex> init = {{z, 1.0}};
        vector<Gate> circuit = {
            Gate(RZ, {0}, 0.1),
            Gate(RX, {0}, 0.15),
            Gate(RY, {1}, 0.2),
            Gate(CNOT, {0, 1}),
            Gate(RZ, {0}, 0.1)
        };
        diagnostic_propagation(init, circuit, 10);
    }
    
    cout << "\n\n";
    
    // Example 4: Deeper circuit
    cout << "Example 4: Deep Circuit (many layers)\n";
    cout << "--------------------------------------\n";
    {
        PauliWord x(3);
        x.ops[1] = X;
        map<PauliWord, Complex> init = {{x, 1.0}};
        vector<Gate> circuit;
        for (int i = 0; i < 5; i++) {
            circuit.push_back(Gate(RZ, {0}, 0.1));
            circuit.push_back(Gate(RX, {1}, 0.1));
            circuit.push_back(Gate(CNOT, {0, 1}));
        }
        diagnostic_propagation(init, circuit, 10);
    }
    
    cout << "\n========================================\n";
    cout << "   DIAGNOSTIC COMPLETE\n";
    cout << "========================================\n";
    
    return 0;
}

