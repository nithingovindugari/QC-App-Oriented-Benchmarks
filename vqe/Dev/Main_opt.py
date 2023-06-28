"""
This script is an example illustrating the process of estimating the accuracy of the pUCCD algorithm on mock hydrogen chains. 
The script simulates hydrogen chains of different lengths (num_qubits), constructs the corresponding pUCCD circuits, and then computes their expectation values using a noiseless simulation.
"""
import sys
sys.path[1:1] = [ "Dev", "hydrogen-lattice/Dev" ]
sys.path[1:1] = [ "../../Dev", "../../Dev", "../../hydrogen-lattice/Dev/" ]

import numpy as np
import ansatz,simulator
from ansatz import PUCCD 
from simulator import Simulator 
from scipy.optimize import minimize
import matplotlib.pyplot as plt



# Create an instance of the Simulator class for noiseless simulations
ideal_backend = simulator.Simulator()

# Initialize an empty list to accumulate simulation data for hydrogen chains of different lengths
simulation_data = []

# Instantiate the pUCCD algorithm
puccd = PUCCD()

# Define the number of shots (number of repetitions of each quantum circuit)
# For the noiseless simulation, we use 10,000 shots.
# For the statevector simulator, this would be set to None.
shots = 10_000

# Initialize an empty list to store the lowest energy values
lowest_energy_values = []


def compute_energy(circuit, operator, shots, parameters):
    
    # Bind the parameters to the circuit
    bound_circuit = circuit.bind_parameters(parameters)
    
    # Compute the expectation value of the circuit with respect to the Hamiltonian for optimization
    energy = ideal_backend.compute_expectation(circuit, operator=operator, shots=shots)
    
    # Append the energy value to the list
    lowest_energy_values.append(energy)
    
    return energy


# Loop over hydrogen chains with different numbers of qubits (from 2 to 4 in this example)
for num_qubits in range(2, 5):
    # Construct the pUCCD circuit for the current mock hydrogen chain
    circuit = puccd.build_circuit(num_qubits)

    operator = puccd.generate_mock_hamiltonian(num_qubits)

    # Initialize the parameters with -1e-3 or 1e-3
    initial_parameters = [np.random.choice([-1e-3, 1e-3]) for _ in range(len(circuit.parameters))]
    circuit.assign_parameters(initial_parameters, inplace=True)

    # Initialize the COBYLA optimizer
    optimizer = 'COBYLA'

    # Set the maximum number of iterations, tolerance, and display options
    max_iterations = 15
    tolerance = 1e-3
    display = False

    # Optimize the circuit parameters using the optimizer
    optimized_parameters = minimize(
        lambda parameters: compute_energy(circuit, operator, shots=shots, parameters=parameters),
        x0=initial_parameters,
        method=optimizer,
        tol=tolerance
        options={'maxiter': max_iterations, 'disp': display}
    )

    # Extract the parameter values from the optimizer result
    optimized_values = optimized_parameters.x

    # Create a dictionary of {parameter: value} pairs
    parameter_values = {param: value for param, value in zip(circuit.parameters, optimized_values)}

    # Assign the optimized values to the circuit parameters
    circuit.assign_parameters(parameter_values, inplace=True)
    
    ideal_energy = ideal_backend.compute_expectation(circuit, operator=operator, shots=shots)
    
    print(ideal_energy)
    
    classical_energy = np.linalg.eigvalsh(real_hamiltonian.paired_matrix)[0]
    print(classical_energy, "classical_energy")
    # Plot the lowest energy values after each iteration
    # plt.figure()
    print(len(lowest_energy_values), "len(lowest_energy_values)")
    plt.plot(range(len(lowest_energy_values)), lowest_energy_values, label='Quantum Energy')
    plt.axhline(y=np.array([classical_energy]), color='r', linestyle='--', label='Classical Energy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Energy')
    plt.title('Energy Comparison: Quantum vs. Classical')
    plt.legend()
    plt.show()
    plt.pause(0.001)
