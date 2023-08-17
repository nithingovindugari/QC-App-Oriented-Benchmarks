from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

# Variational circuit ansatzes
def ansatz_block(num_qubits, reps):
    
    # reps is Number of times ry and cx gates are repeated
    qc = QuantumCircuit(num_qubits)
    parameter_vector = ParameterVector("t", length=num_qubits*reps*2)
    # print("parameter_vector",parameter_vector)
    counter = 0
    for rep in range(reps):
        for i in range(num_qubits):
            theta = parameter_vector[counter]
            qc.ry(theta, i)
            counter += 1
        
        for i in range(num_qubits):
            theta = parameter_vector[counter]
            qc.rx(theta, i)
            counter += 1
    
        for j in range(0, num_qubits - 1, 1):
            if rep<reps-1:
                qc.cx(j, j + 1)
    # print("counter",counter)
    return qc

# Define the convolutional circuits
def conv_ansatz(thetas):
    # Your implementation for conv_circ_1 function here
    # print(thetas)
    conv_circ = QuantumCircuit(2)
    conv_circ.rx(thetas[0], 0)
    conv_circ.rx(thetas[1], 1)
    conv_circ.rz(thetas[2], 0)
    conv_circ.rz(thetas[3], 1)


    conv_circ.crx(thetas[4], 0, 1)  
    conv_circ.crx(thetas[5], 1, 0)

    conv_circ.rx(thetas[6], 0)
    conv_circ.rx(thetas[7], 1)
    conv_circ.rz(thetas[8], 0)
    conv_circ.rz(thetas[9], 1)

    conv_circ.crz(thetas[10], 1, 0)  
    conv_circ.x(1)
    conv_circ.crx(thetas[11], 1, 0)
    conv_circ.x(1)

    #conv_circ = QuantumCircuit(2)
    #conv_circ.crx(thetas[0], 0, 1)

    # print(conv_circ)
    return conv_circ



def ansatz_qcnn(num_qubits):

    qc = QuantumCircuit(num_qubits)

    num_layers = int(np.ceil(np.log2(num_qubits)))
    num_parameters_per_layer = 12
    num_parameters=num_layers*num_parameters_per_layer
    parameter_vector = ParameterVector("t", length=num_parameters)    

    for i_layer in range(num_layers):
        for i_sub_layer in [0 , 2**i_layer]:            
            for i_q1 in range(i_sub_layer, num_qubits, 2**(i_layer+1)):
                i_q2=2**i_layer+i_q1
                if i_q2<num_qubits:
                    qc=qc.compose(conv_ansatz(parameter_vector[num_parameters_per_layer*i_layer:num_parameters_per_layer*(i_layer+1)]), qubits=(i_q1,i_q2))
                    #print("i_q1",i_q1,"i_q2",i_q2)
    
    #print(qc)
    return qc




