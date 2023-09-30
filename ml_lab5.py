#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np

# Initialize weights and bias
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Define the activation function (step function in this case)
def activate(z):
    if z >= 0:
        return 1
    else:
        return 0

# Define the perceptron function
def perceptron(input_data, weights):
    # Calculate the weighted sum of inputs
    z = W0 + np.dot(input_data, weights)
    # Apply the activation function
    output = activate(z)
    return output

# Training data (you can modify this as needed)
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Two input features
target_output = np.array([0, 0, 0, 1])  # Adjust this for your specific problem

# Training loop
epochs = 100  # Number of training iterations
for epoch in range(epochs):
    for i in range(len(input_data)):
        # Compute the predicted output
        prediction = perceptron(input_data[i], [W1, W2])
        # Compute the error
        error = target_output[i] - prediction
        # Update the weights and bias
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * input_data[i][0]
        W2 = W2 + learning_rate * error * input_data[i][1]

# Print the final weights
print("Final Weights:")
print("W0 =", W0)
print("W1 =", W1)
print("W2 =", W2)


# In[12]:


def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

W = np.array([10, 0.2, -0.75])
learning_rate = 0.05
convergence_error = 0.002
max_epochs = 1000

input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([0, 0, 0, 1])
error_values = []
epochs = 0
while True:
    total_error = 0  
    for i in range(len(input_data)):
        weighted_sum = W[0] + W[1] * input_data[i, 0] + W[2] * input_data[i, 1]
        predicted_output = bipolar_step_activation(weighted_sum)
        error_i = target_output[i] - predicted_output
        total_error += error_i ** 2  
        W[0] += learning_rate * error_i
        W[1] += learning_rate * error_i * input_data[i, 0]
        W[2] += learning_rate * error_i * input_data[i, 1]
 
    error_values.append(total_error) 
    epochs += 1
    if total_error <= convergence_error or epochs >= max_epochs:
        break

print(" bipolar step activation : ")
print(f"Converged in {epochs} epochs.")
print("Final weights:",W)


# In[ ]:




