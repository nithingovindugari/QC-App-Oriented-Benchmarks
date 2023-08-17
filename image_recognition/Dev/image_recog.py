
# Importing the required libraries for the project

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize
import matplotlib.pyplot as plt
#from qiskit.algorithms.optimizers import SPSA
from noisyopt import minimizeSPSA
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer, transpile,execute
from qiskit.circuit.library import RealAmplitudes
from sklearn.metrics import accuracy_score,log_loss, mean_squared_error
from qiskit.circuit import ParameterVector

from qiskit.algorithms.optimizers import SPSA

import expectation_calc
from ansatz import ansatz_block, ansatz_qcnn, conv_ansatz

global expectation_calc_method
batch_index = 0

# change the below variable to True if you want to use expectation_calc.py file for expectation value calculation
expectation_calc_method = True

# Dev Note :- Each image has Intensity from 0 to 255

''' All steps are explained below
    1. Fetch the MNIST dataset
    2. Access the data and target  
        i) filter the data with 7 and 9 ( for initial development we are doing binary classification )
    3. Split the data into train and test data ( 80% for training and 20% for testing )
    4. pca for dimensionality reduction (x_scaled_data is the scaled data)
    5. batching the data ( we are not using batching for now)
    6. custom varational circuit is defined for now instead of QCNN (var_circuit)
        i)   it will be updated once we have the QCNN
    7. Input data is encoded into quantum state and then passed through the custom variational circuit(qcnn_model)
   
   Pending :- loss function is not minimized proply to get convergance and possible reasons are 
        i) we are not encoding the data properly into quantum state 
        ii) expectation value calcuation needs to be improved
        ii) cobyla optimizer coulen't find the minimum value
        iv) we need to check if we need to use classical neural network to extract labels from the circuit output
        
    8. loss function is defined (loss) as our objective is to minimize the loss ( Currently changes are in progess)
        i)   Function to predict the label ( 7 or 9 i.e, 0,1) from circuit output is 
        ii)  loss function is pending and will be updated once the above function is done
        iii) need to check if to use classical neural network to extract labels from the circuit output
    9. Optimizer is defined (optimizer) 
    10. Testing the model (Pending :-  Improve the code to test the model on test data)
    '''

import expectation_calc

np.random.seed(72)

train_accuracy_history = []
test_accuracy_history = []
loss_history = []

# Number of qubits for the quantum circuit
num_qubits = 4

# Fetch the MNIST dataset from openml
mnist = fetch_openml('mnist_784', parser='auto', version=1, as_frame=False)

# Access the data and target
# x has all the pixel values of image and has Data shape of (70000, 784)  here 784 is 28*28
x = mnist.data
print("shape of x:",x.shape)

# y has all the labels of the images and has Target shape of (70000,)
y = mnist.target
print("shape of y:",y.shape)


# convert the given y data to integers so that we can filter the data
y = y.astype(int)

# Filtering only values with 7 or 9 as we are doing binary classification for now and we will extend it to multi-class classification
binary_filter = (y == 7) | (y == 9)

# Filtering the x and y data with the binary filter
x = x[binary_filter]
y = y[binary_filter]

# create a new y data with 0 and 1 values with 7 as 0 and 9 as 1 so that we can use it for binary classification
y = (y == 9).astype(int)


''' Here X_train has all the training pixel values of image (i.e, 80% ) and has Data shape of (56000, 784) here 784 is 28*28
    Here y_train has all the training labels of the images (i.e, 80% ) and has Target shape of (56000,)
    
    Here X_test has all the testing pixel values of images (i.e, 20% ) and has Data shape of (14000, 784) 
    Here y_test has all the testing labels of the images (i.e, 20% ) and has Target shape of (14000,)'''

# Here we have only two classes data out of above mentioned data    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, train_size=200, random_state=42)
# print(x.shape, y.shape, x_train.shape, y_train.shape, x_test.shape, y_test.sha?pe)

## Normalize images between 0 to 1
#x= normalize(x, norm='l2', axis=1)
#x_test =  normalize(x_test, norm='l2', axis=1)
#x_train = normalize(x_train, norm='l2', axis=1)

pca = PCA(n_components = num_qubits).fit(x)

# Apply PCA on the data to reduce the dimensions and it is in the form of 'numpy.ndarray'
x_pca= pca.transform(x)
x_train_pca = pca.transform(x_train)
x_test_pca =  pca.transform(x_test)

#Print y_test and x_test one by one
#print('PCA compressed image')
#for i in range(len(y_test)):
#    print("y:", y_test[i], "x:", x_test_pca[i])

# To visualize if prinicipal components are enough and to decide the number of principal components to keep 
pca_check = False 
if pca_check == True:
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()

#x_scaled_test = x_test_pca
#x_scaled_train = x_train_pca
# Create an instance of MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit(x_pca)


# Apply min-max scaling to the data to bring the values between 0 and 1
x_scaled_test =  scaler.transform(x_test_pca)
x_scaled_train = scaler.transform(x_train_pca)

#Print y_test and x_test one by one
#print('Scaled image')
#for i in range(len(y_test)):
#    print("y:", y_test[i], "x:", x_scaled_test[i])

# Data frame here to just to visualize the data and Not needed once model is defined
pca_vis = False
if pca_vis == True:
    x_vis = pd.DataFrame(x_scaled_train)
    stats = x_vis.describe()
    print(stats)







# model to be used for training which has input data encoded and variational circuit is appended to it
def model(theta, x, num_qubits, reps=3):
    qc = QuantumCircuit(num_qubits)
    
    # feature mapping 
    for j in range(num_qubits):
        qc.ry(x[j], j )

    # Append the variational circuit ( Ansatz ) to the quantum circuit
    if ansatz_type == 'block':
        var_circ = ansatz_block(num_qubits, reps)
    elif ansatz_type == 'qcnn':
        var_circ = ansatz_qcnn(num_qubits)
    else:
        print("Invalid var_circ_type")

    var_circ.assign_parameters(theta, inplace=True)
    qc.compose(var_circ, inplace=True)
    # qc.measure_all()  # Measure all qubits will be changed to measure only 7 qubits if needed
    # Measure only the first 7 qubits
    # Add a classical register with 7 bits to store the results of the measurements

    return qc


# Define the quantum instance to be used for training
backend = Aer.get_backend('qasm_simulator')

# threshold to be used for prediction of label from the circuit output
global threshold 

threshold = 0.5

# function to calculate the expectation value ( pending :- Analysis in progress to improve or replace below function )
def expectation_values(result):
    expectation = 0
    probabilities = {}
    for outcome, count in result.items():
        bitstring = outcome#[::-1]  # Reverse the bitstring
        decimal_value = int(bitstring, 2)  # Convert the bitstring to decimal
        probability = count / num_shots  # Compute the probability of each bitstring
        expectation += decimal_value * probability / 2 ** (num_qubits/2)  # Compute the expectation value
    if expectation > threshold:
        prediction_label = 1
    else:
        prediction_label = 0
    #print("expectation",expectation)
    #return prediction_label


def extract_label(result):
    bitstring = result['counts'].most_common(1)[0][0]
    prediction_label = int(bitstring[::-1], 2)
    if prediction_label == 0:
        return 0
    else:
        return 1
    
# function to calculate the loss function will be update after finding the suitable loss function
 # Tried applying log loss ( cross entropy) but it is not minimizing the loss
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

# function to calculate the loss function

def loss_function(theta, is_draw_circ=False, is_print=False):
    
    #print('called loss function')

    predictions=calculate_predictions(x_batch, theta, num_qubits, num_shots, reps=reps)
        
    # Cross entropy loss
    #loss = log_loss(y_train, prediction_label)
    loss = mean_squared_error(y_batch, predictions)
    #print("cross entropy loss:", loss)
    #print("theta:", theta)
    #loss = square_loss(y_train, prediction_label)
    #print("loss:", loss)

    if is_print:
        accuracy = calculate_accuracy(y_batch, predictions)
        print("Batch index:", batch_index  , "Train Accuracy:", accuracy, "loss:", loss)
        train_accuracy_history.append(accuracy)

        #test_accuracy=calculate_accuracy(x_scaled_test, y_test, theta, num_qubits, reps, num_shots)
        #print("Test Accuracy:", test_accuracy)

    return loss

def callback(theta):

    #Create a batch from random indices

    #print("theta:", theta)
    loss=loss_function(theta, is_draw_circ=False, is_print=True)
    loss_history.append(loss)

    #Calculate the accuracy on test data
    test_predictions=calculate_predictions(x_scaled_test, theta, num_qubits, num_shots, reps=reps)
    test_accuracy=calculate_accuracy(y_test, test_predictions)
    test_accuracy_history.append(test_accuracy)

    #print(theta)
    ##Calculate squared difference between test_predications and y_test and print it
    #squared_difference = (y_test - test_predictions) ** 2
    #print("squared_difference",squared_difference)
    #print("test_predictions",test_predictions)


    global x_batch, y_batch, batch_index    
    indices = np.random.choice(len(x_scaled_train), size=batch_size, replace=False)
    x_batch = x_scaled_train[indices]
    y_batch = y_train[indices]

    # Get modulus of batch_index with number of batches
    #global x_batch, y_batch, batch_index
    #i_batch = batch_index % num_batches
    #x_batch = x_scaled_train[i_batch * batch_size : (i_batch + 1) * batch_size]
    #y_batch = y_train[i_batch * batch_size : (i_batch + 1) * batch_size]

    batch_index=batch_index+1


# function to calculate the prediction of the model
def calculate_predictions(x, theta, num_qubits, num_shots, reps=3):
    predictions = []
    for data_point in x:
        qc = model(theta, data_point, num_qubits, reps)
        #print(qc)
        val = expectation_calc.calculate_expectation(qc,shots=num_shots,num_qubits=num_qubits)   
        val=(val+1)*0.5
        predictions.append(val)

    return predictions


def calculate_accuracy(y, predictions):

    prediction_labels = []
    for i in range(len(y)):
        if predictions[i] > 0.5:
            prediction_labels.append(1)
        else:
            prediction_labels.append(0)

    accuracy = accuracy_score(y, prediction_labels)
  

    return accuracy

# Initialize  epochs
num_epochs = 100

# Batch size for the optimizer
batch_size = 50
num_batches = int(np.ceil(len(x_scaled_train) / batch_size))

# Learning rate
init_step_size = 0.1

# Number of shots to run the program (experiment)
num_shots = 1000
reps = 3

# Choose the variational circuit
ansatz_type = 'qcnn' # 'block' or 'qcnn'

# Initialize the weights for the QNN model
if ansatz_type == 'block':
    num_parameters= num_qubits * reps * 2
elif ansatz_type == 'qcnn':
    num_parameters_per_layer=12
    num_layers = int(np.ceil(np.log2(num_qubits)))
    num_parameters = num_parameters_per_layer*num_layers
else:
    print("Invalid ansatz_type")

weights = np.random.rand(num_parameters)
#weights = np.zeros(num_parameters)
print("Number of parameters:", len(weights))

global x_batch, y_batch
indices = np.random.choice(len(x_scaled_train), size=batch_size, replace=False)
x_batch = x_scaled_train[indices]
y_batch = y_train[indices]

#global x_batch, y_batch
#i_batch = batch_index % num_batches
#x_batch = x_scaled_train[i_batch * batch_size : (i_batch + 1) * batch_size]
#y_batch = y_train[i_batch * batch_size : (i_batch + 1) * batch_size]

#for k in range(5):
#    indices = np.random.choice(len(x_scaled_train), size=batch_size, replace=False)
#    x_batch = x_scaled_train[indices]
#    y_batch = y_train[indices]
#    predictions=calculate_predictions(x_batch, weights, num_qubits, reps, num_shots)
#
#    #Form and print a 2 column vector from indices and predictions
#    indices = indices.reshape(-1, 1)
#    predictions = np.array(predictions).reshape(-1, 1)
#    data = np.hstack((indices, predictions))
#    print(data)


#res = minimize(loss_function, x0 = weights, method="COBYLA", tol=0.001, callback=callback, options={'disp': False, 'rhobeg': init_step_size} )
res= minimizeSPSA(loss_function, x0=weights, a=0.3, c=0.3, niter=1000, callback=callback, paired=False)

# Will increase the number of epochs once the code is fine tuned to get convergance 
#accuracy_values = []
#for epoch in range(num_epochs):

#    for i_batch in range(num_batches):
        # Minimize the loss function

        #res = minimize(loss_function, x0 = weights, method="COBYLA", tol=0.001, callback=None, options={'maxiter': 50, 'disp': False, 'rhobeg': init_step_size} )
        #theta=SPSA(maxiter=100).minimize(loss_function, x0=weights)

        #weights = res.x

        #print("Batch number:",i_batch, "loss",res.fun)

    #train_accuracy = calculate_accuracy(x_scaled_train, y_train, res.x, num_qubits, reps, num_shots)
    #print(f"Epoch {epoch+1}/{num_epochs}, train accuracy = {train_accuracy:.4f}")
    #accuracy_values.append(train_accuracy)
    
    #init_step_size = init_step_size / 1.4

#print("result", res)
    # print(type(theta.x))
# To find threshold
# print("max_values",max_values)
# max = max(max_values)
# average_value = sum(max_values) / len(max_values)
# print("average",average_value)
# last step is to test the model on test data

# Evaluate the QCNN accuracy on the test set once the model is trained and tested
test_predictions=calculate_predictions(x_scaled_test, res.x, num_qubits, num_shots, reps=reps)
test_accuracy=calculate_accuracy(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)



#Create a plot with subplots in a grid of 1X3
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

#Plot train accuracy in the first subplot
axs[0].plot(train_accuracy_history, label="train accuracy")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Train Accuracy")

#Plot test accuracy in the second subplot
axs[1].plot(test_accuracy_history, label="test accuracy")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Test Accuracy")

#Plot loss in the third subplot
axs[2].plot(loss_history, label="loss")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Loss")

#Show the plot
plt.show()
