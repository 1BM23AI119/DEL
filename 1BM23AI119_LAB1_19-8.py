#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def step_activation(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0.0

    def predict(self, inputs):
        total_input = np.dot(inputs, self.weights) + self.bias
        return step_activation(total_input)

    def train(self, training_inputs, labels, epochs=10, learning_rate=0.1):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error

def plot_decision_boundary(perceptron, inputs, labels):
    # Plot data points with different colors for different labels
    for input_vec, label in zip(inputs, labels):
        color = 'red' if label == 0 else 'blue'
        plt.scatter(input_vec[0], input_vec[1], color=color, s=100, edgecolors='k')

    # Calculate decision boundary line: w1*x1 + w2*x2 + b = 0
    # => x2 = -(w1*x1 + b)/w2
    x1_vals = np.array([0, 1])
    if perceptron.weights[1] != 0:
        x2_vals = -(perceptron.weights[0] * x1_vals + perceptron.bias) / perceptron.weights[1]
        plt.plot(x1_vals, x2_vals, label='Decision Boundary')
    else:
        # If w2 is zero, plot vertical line at -b/w1
        x = -perceptron.bias / perceptron.weights[0]
        plt.axvline(x=x, label='Decision Boundary')

    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary (AND gate)')
    plt.legend()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.show()

# Training AND gate perceptron and plotting
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

p = Perceptron(input_size=2)
p.train(training_inputs, labels, epochs=10, learning_rate=0.1)

plot_decision_boundary(p, training_inputs, labels)


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Activation Functions
# ----------------------------
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# ----------------------------
# Softmax Function
# ----------------------------
def softmax(z):
    e_z = np.exp(z - np.max(z))  # for numerical stability
    return e_z / e_z.sum()

# ----------------------------
# Plot Activation Functions (Linear, Sigmoid, Tanh, ReLU)
# ----------------------------
x = np.linspace(-10, 10, 1000)
y_linear = linear(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

plt.figure(figsize=(12, 10))

# Linear
plt.subplot(2, 2, 1)
plt.plot(x, y_linear, label='Linear', color='blue')
plt.title("Linear Activation Function")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel("Input")
plt.ylabel("Output")

# Sigmoid
plt.subplot(2, 2, 2)
plt.plot(x, y_sigmoid, label='Sigmoid', color='green')
plt.title("Sigmoid Activation Function")
plt.grid(True)
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel("Input")
plt.ylabel("Output")

# Tanh
plt.subplot(2, 2, 3)
plt.plot(x, y_tanh, label='Tanh', color='purple')
plt.title("Tanh Activation Function")
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Input")
plt.ylabel("Output")

# ReLU
plt.subplot(2, 2, 4)
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title("ReLU Activation Function")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel("Input")
plt.ylabel("Output")

plt.tight_layout()
plt.show()

# ----------------------------
# Plot Softmax Outputs for Sample Vectors
# ----------------------------
logits_list = [
    [-1, 0, 1],
    [1, 2, 3],
    [2, 4, 6],
    [5, 5, 5],
    [3, 1, 0, 2]
]

plt.figure(figsize=(10, 6))

for i, logits in enumerate(logits_list):
    probs = softmax(np.array(logits))
    x_vals = range(len(probs))
    plt.plot(x_vals, probs, marker='o', label=f'Input {i+1}: {logits}')

plt.title('Softmax Output for Different Input Vectors (Logits)')
plt.xlabel('Class Index')
plt.ylabel('Probability')
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




