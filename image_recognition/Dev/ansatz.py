from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

# Variational circuit ansatzes
def parametrized_block(thetas, num_qubits, reps):
    
    # reps is Number of times ry and cx gates are repeated
    qc = QuantumCircuit(num_qubits)    
    # print("parameter_vector",parameter_vector)
    counter = 0
    for rep in range(reps):
        for i in range(num_qubits):
            theta = thetas[counter]
            qc.ry(theta, i)
            counter += 1
        
        for i in range(num_qubits):
            theta = thetas[counter]
            qc.rx(theta, i)
            counter += 1
    
        for j in range(0, num_qubits - 1, 1):
            if rep<reps-1:
                qc.cx(j, j + 1)
    # print("counter",counter)
    return qc

# Ansatz from paper https://arxiv.org/pdf/2108.00661.pdf
def parameterized_2q_gate_1(thetas, num_reps=1):
    # Your implementation for conv_circ_1 function here
    # print(thetas)
    conv_circ = QuantumCircuit(2)

    for i in range(num_reps):
        conv_circ.rx(thetas[12*i+0], 0)
        conv_circ.rx(thetas[12*i+1], 1)
        conv_circ.rz(thetas[12*i+2], 0)
        conv_circ.rz(thetas[12*i+3], 1)


        conv_circ.crx(thetas[12*i+4], 0, 1)  
        conv_circ.crx(thetas[12*i+5], 1, 0)

        conv_circ.rx(thetas[12*i+6], 0)
        conv_circ.rx(thetas[12*i+7], 1)
        conv_circ.rz(thetas[12*i+8], 0)
        conv_circ.rz(thetas[12*i+9], 1)

        conv_circ.crz(thetas[12*i+10], 1, 0)  
        conv_circ.x(1)
        conv_circ.crx(thetas[12*i+11], 1, 0)
        conv_circ.x(1)

    #conv_circ = QuantumCircuit(2)
    #conv_circ.crx(thetas[0], 0, 1)

    # print(conv_circ)
    return conv_circ

#Most general two qubit gate ansatz
def parameterized_2q_gate_2(thetas, num_reps=1):

    conv_circ = QuantumCircuit(2)

    for i in range(num_reps):
        conv_circ.rx(thetas[15*i+0], 0)
        conv_circ.rz(thetas[15*i+1], 0)
        conv_circ.rx(thetas[15*i+2], 0)
    
        conv_circ.rx(thetas[15*i+3], 1)
        conv_circ.rz(thetas[15*i+4], 1)
        conv_circ.rx(thetas[15*i+5], 1)
    
        conv_circ.cx(1, 0)
        conv_circ.rz(thetas[15*i+6], 0)
        conv_circ.ry(thetas[15*i+7], 1)
        conv_circ.cx(0, 1)
        conv_circ.ry(thetas[15*i+8], 1)
        conv_circ.cx(1, 0)
    
        conv_circ.rx(thetas[15*i+9], 0)
        conv_circ.rz(thetas[15*i+10], 0)
        conv_circ.rx(thetas[15*i+11], 0)
    
        conv_circ.rx(thetas[15*i+12], 1)
        conv_circ.rz(thetas[15*i+13], 1)
        conv_circ.rx(thetas[15*i+14], 1)

    return conv_circ

    
def ansatz(ansatz_type,num_qubits, num_reps=1):

    qc = QuantumCircuit(num_qubits) 

    if ansatz_type == "block":
        parameter_vector = ParameterVector("t", length=num_qubits*num_reps*2)
        qc=qc.compose(parametrized_block(parameter_vector,num_qubits, num_reps), qubits=range(num_qubits))
    elif ansatz_type == "qcnn uniform":
        num_layers = int(np.ceil(np.log2(num_qubits)))
        num_parameters_per_layer = 15 * num_reps
        num_parameters=num_layers*num_parameters_per_layer
        parameter_vector = ParameterVector("t", length=num_parameters)  
        qc = QuantumCircuit(num_qubits)  
        for i_layer in range(num_layers):
            for i_sub_layer in [0 , 2**i_layer]:            
                for i_q1 in range(i_sub_layer, num_qubits, 2**(i_layer+1)):
                    i_q2=2**i_layer+i_q1
                    if i_q2<num_qubits:
                        qc=qc.compose(parameterized_2q_gate_2(parameter_vector[num_parameters_per_layer*i_layer:num_parameters_per_layer*(i_layer+1)], num_reps=num_reps), qubits=(i_q1,i_q2))  
                        #print("i_q1",i_q1,"i_q2",i_q2)
    elif ansatz_type == "qcnn unique":
        num_layers = int(np.ceil(np.log2(num_qubits)))
        num_parameters_per_conv = 15 * num_reps
        parameter_vector = ParameterVector("t", length=0)  
        qc = QuantumCircuit(num_qubits)  
        i_conv=0
        for i_layer in range(num_layers):
            for i_sub_layer in [0 , 2**i_layer]:            
                for i_q1 in range(i_sub_layer, num_qubits, 2**(i_layer+1)):
                    i_q2=2**i_layer+i_q1
                    if i_q2<num_qubits:
                        parameter_vector.resize((i_conv+1)*num_parameters_per_conv)
                        qc=qc.compose(parameterized_2q_gate_2(parameter_vector[num_parameters_per_conv*i_conv:num_parameters_per_conv*(i_conv+1)], num_reps=num_reps), qubits=(i_q1,i_q2)) 
                        i_conv+=1
    else:
        print("ansatz_type not recognized")
    
    #print(qc)
    return qc




