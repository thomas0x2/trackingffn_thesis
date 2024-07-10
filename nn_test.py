import numpy as np 
from neural_network import NeuralNetwork, Layer

nn = NeuralNetwork()

input_size = 2
hidden_size = 3 
output_size = 1

# Create weight and bias matrices/vectors
weights1 = np.random.randn(hidden_size, input_size)
biases1 = np.random.randn(hidden_size)
weights2 = np.random.randn(output_size, hidden_size)
biases2 = np.random.randn(output_size)

# Add layers to the network
layer1 = Layer(weights1, biases1)
layer2 = Layer(weights2, biases2)

nn.add_layer(layer1)
nn.add_layer(layer2)

# Create some sample data
X = np.array([1.0, 2.0])
y = np.array([3.0])

# Make a prediction
prediction = nn.predict(X)
print(f"Prediction: {prediction}")

# Calculate loss
loss = nn.mse_loss(X, y)
print(f"MSE Loss: {loss}")

# Train the network
learning_rate = 0.01
momentum = 0.9
nn.train(X, y, learning_rate, momentum)
print("Training successful!")

# Make a prediction
prediction = nn.predict(X)
print(f"Prediction: {prediction}")

# Calculate loss
loss = nn.mse_loss(X, y)
print(f"MSE Loss: {loss}")
