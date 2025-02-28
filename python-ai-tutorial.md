# Python for AI: From Fundamentals to Neural Networks

## Table of Contents
1. [Introduction to Python for AI Programmers](#introduction-to-python-for-ai-programmers)
2. [Linear Algebra Essentials](#linear-algebra-essentials)
3. [Calculus Essentials](#calculus-essentials)
4. [Programming Transformer Neural Networks with PyTorch](#programming-transformer-neural-networks-with-pytorch)
5. [Create Your Own Image Classifier](#create-your-own-image-classifier)

## Introduction to Python for AI Programmers

### Python Data Types
Python offers several built-in data types that are essential for AI programming:

```python
# Numeric types
x = 10          # int
y = 3.14        # float
z = complex(1, 2)  # complex number (1+2j)

# Sequence types
my_list = [1, 2, 3]        # List - mutable
my_tuple = (1, 2, 3)       # Tuple - immutable
my_range = range(5)        # Range

# Text type
text = "Hello, AI"         # String

# Mapping type
my_dict = {"model": "CNN", "accuracy": 0.95}  # Dictionary

# Set types
my_set = {1, 2, 3}         # Set - unique, unordered elements
my_frozenset = frozenset([1, 2, 3])  # Immutable set

# Boolean type
is_trained = True          # Boolean

# Binary types
my_bytes = b"binary data"  # Bytes - immutable
my_bytearray = bytearray(5)  # Bytearray - mutable
my_memoryview = memoryview(bytes(5))  # Memory view
```

### Control Flow
Control flow statements direct the execution path of your code:

```python
# If-elif-else statement
accuracy = 0.85
if accuracy > 0.9:
    print("High accuracy model")
elif accuracy > 0.7:
    print("Good accuracy model")
else:
    print("Model needs improvement")

# For loop
for epoch in range(10):
    print(f"Training epoch {epoch}")

# While loop
patience = 5
early_stop_counter = 0
while early_stop_counter < patience:
    # Training logic
    loss = calculate_loss()
    if loss < best_loss:
        best_loss = loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
```

### Python Operators
Python provides various operators for different operations:

```python
# Arithmetic operators
a = 10
b = 3
print(a + b)    # Addition: 13
print(a - b)    # Subtraction: 7
print(a * b)    # Multiplication: 30
print(a / b)    # Division: 3.333...
print(a // b)   # Floor division: 3
print(a % b)    # Modulus: 1
print(a ** b)   # Exponentiation: 1000

# Comparison operators
print(a == b)   # Equal: False
print(a != b)   # Not equal: True
print(a > b)    # Greater than: True
print(a < b)    # Less than: False
print(a >= b)   # Greater than or equal to: True
print(a <= b)   # Less than or equal to: False

# Logical operators
c = True
d = False
print(c and d)  # Logical AND: False
print(c or d)   # Logical OR: True
print(not c)    # Logical NOT: False

# Bitwise operators
print(a & b)    # AND: 2
print(a | b)    # OR: 11
print(a ^ b)    # XOR: 9
print(~a)       # NOT: -11
print(a << 2)   # Left shift: 40
print(a >> 2)   # Right shift: 2

# Assignment operators
x = 5
x += 3          # x = x + 3
x -= 2          # x = x - 2
x *= 4          # x = x * 4
```

### Python Exception Handling
Exception handling allows you to manage errors gracefully:

```python
try:
    # Attempt to load a model
    model = load_model("model.h5")
    predictions = model.predict(data)
except FileNotFoundError:
    print("Model file not found")
    model = train_new_model()
except ValueError as e:
    print(f"Value error: {e}")
    model = fallback_model()
else:
    # Executes if no exception occurs
    print("Model loaded successfully")
finally:
    # Always executes
    print("Prediction process completed")

# Custom exceptions
class ModelConvergenceError(Exception):
    """Raised when a model fails to converge"""
    pass

def train_model(epochs):
    for epoch in range(epochs):
        if loss > previous_loss and epoch > 10:
            raise ModelConvergenceError("Model is diverging")
```

### Lambda Expressions
Lambda expressions create anonymous functions:

```python
# Simple lambda function
square = lambda x: x**2
print(square(5))  # 25

# Lambda with multiple parameters
weighted_sum = lambda x, w: sum(x_i * w_i for x_i, w_i in zip(x, w))
features = [1, 2, 3]
weights = [0.5, 0.3, 0.2]
print(weighted_sum(features, weights))  # 1.4

# Lambda with conditional logic
activation = lambda x: x if x > 0 else 0  # ReLU function
print(activation(-3))  # 0
print(activation(5))   # 5

# Lambdas in sorting
data_points = [(2, 0.5), (1, 0.8), (3, 0.6)]
sorted_by_second = sorted(data_points, key=lambda point: point[1])
print(sorted_by_second)  # [(2, 0.5), (3, 0.6), (1, 0.8)]
```

### Code Debugging
Python offers several debugging techniques:

```python
# Using print statements
print(f"Shape before transform: {data.shape}")
transformed_data = transform(data)
print(f"Shape after transform: {transformed_data.shape}")

# Using assertions
assert data.shape[0] == labels.shape[0], "Data and labels must have same length"

# Using logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting model training")
logger.warning("Learning rate might be too high")
logger.error("Out of memory error")

# Using pdb (Python Debugger)
import pdb
def complex_function(data):
    transformed = preprocess(data)
    pdb.set_trace()  # Debugger stops here
    result = apply_model(transformed)
    return postprocess(result)
```

### Python Function Definition
Functions encapsulate reusable code blocks:

```python
# Basic function
def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of model predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)

# Function with default parameters
def train_model(data, epochs=100, learning_rate=0.01, batch_size=32):
    # Model training code
    return model

# Function with *args and **kwargs
def ensemble_predict(*models, **prediction_params):
    predictions = []
    for model in models:
        pred = model.predict(**prediction_params)
        predictions.append(pred)
    return aggregate_predictions(predictions)
```

### List Comprehension
List comprehensions provide a concise way to create lists:

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# Conditional list comprehension
positive_values = [x for x in data if x > 0]

# Nested list comprehension
matrix_transpose = [[row[i] for row in matrix] for i in range(len(matrix[0]))]

# Dictionary comprehension
feature_map = {feature: weight for feature, weight in zip(features, weights)}

# Set comprehension
unique_classes = {label for label in labels}

# With function calls
normalized_data = [normalize(x) for x in raw_data]
```

### Generators
Generators create iterators efficiently:

```python
# Generator function
def batch_generator(data, batch_size):
    """Generate batches of data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Using the generator
for batch in batch_generator(training_data, 32):
    train_on_batch(model, batch)

# Generator expression
batch_gen = (data[i:i + batch_size] for i in range(0, len(data), batch_size))

# Infinite generator
def learning_rate_scheduler(initial_lr, decay):
    """Generate decreasing learning rates."""
    lr = initial_lr
    while True:
        yield lr
        lr *= decay
```

### Variable Scope
Understanding variable scope is crucial:

```python
# Global and local scope
global_var = 10

def some_function():
    local_var = 5
    print(global_var)  # Accessible
    print(local_var)   # Accessible

# print(local_var)  # Error: local_var not defined

# Modifying global variables
def update_global():
    global global_var
    global_var = 20

# Nonlocal variables (for nested functions)
def outer_function():
    outer_var = "outer"
    
    def inner_function():
        nonlocal outer_var
        outer_var = "modified"
    
    inner_function()
    print(outer_var)  # "modified"
```

### Iterators
Iterators allow traversal through collections:

```python
# Creating an iterator
my_list = [1, 2, 3]
my_iter = iter(my_list)

print(next(my_iter))  # 1
print(next(my_iter))  # 2
print(next(my_iter))  # 3

# Custom iterator
class EpochTracker:
    def __init__(self, max_epochs):
        self.current_epoch = 0
        self.max_epochs = max_epochs
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_epoch < self.max_epochs:
            self.current_epoch += 1
            return self.current_epoch
        raise StopIteration

# Using the custom iterator
for epoch in EpochTracker(5):
    print(f"Training epoch {epoch}")
```

### Built-in Python Functions
Python offers many built-in functions:

```python
# Common built-in functions
print(len([1, 2, 3]))            # Length: 3
print(type(3.14))                # Type: <class 'float'>
print(max([1, 5, 3]))            # Maximum: 5
print(min([1, 5, 3]))            # Minimum: 1
print(sum([1, 2, 3]))            # Sum: 6
print(sorted([3, 1, 2]))         # Sorted: [1, 2, 3]
print(enumerate(['a', 'b', 'c'])) # Enumerate object
print(zip([1, 2], ['a', 'b']))   # Zip object
print(map(lambda x: x*2, [1, 2, 3])) # Map object
print(filter(lambda x: x > 1, [1, 2, 3])) # Filter object
print(all([True, True, False]))  # All: False
print(any([True, False, False])) # Any: True
print(abs(-5))                   # Absolute value: 5
print(round(3.7))                # Round: 4
```

### Pip
Pip is Python's package installer:

```bash
# Install a package
pip install numpy

# Install a specific version
pip install tensorflow==2.8.0

# Install from requirements file
pip install -r requirements.txt

# Upgrade a package
pip install --upgrade scikit-learn

# Uninstall a package
pip uninstall matplotlib

# List installed packages
pip list

# Show package information
pip show pandas
```

### File I/O
File operations are common in data processing:

```python
# Reading a text file
with open('data.txt', 'r') as file:
    content = file.read()
    
# Writing to a text file
with open('results.txt', 'w') as file:
    file.write("Model accuracy: 0.92")
    
# Appending to a file
with open('log.txt', 'a') as file:
    file.write("\nTraining completed at 15:30")
    
# Reading CSV files
import csv
with open('dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)
        
# Writing CSV files
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'prediction'])
    writer.writerows(predictions)
    
# Working with JSON
import json
with open('config.json', 'r') as jsonfile:
    config = json.load(jsonfile)
    
with open('model_params.json', 'w') as jsonfile:
    json.dump(model_params, jsonfile)
```

### User Input Handling
Processing user input safely:

```python
# Basic input
user_epochs = input("Enter number of epochs: ")
epochs = int(user_epochs)

# Input validation
while True:
    try:
        learning_rate = float(input("Enter learning rate (0-1): "))
        if 0 <= learning_rate <= 1:
            break
        else:
            print("Learning rate must be between 0 and 1")
    except ValueError:
        print("Please enter a valid number")
        
# Command line arguments
import argparse

parser = argparse.ArgumentParser(description='Train a neural network')
parser.add_argument('--data', type=str, required=True, help='Path to dataset')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

print(f"Training with {args.epochs} epochs and learning rate {args.lr}")
```

### Python Data Structures
Data structures organize and store data:

```python
# Lists - ordered, mutable
features = ['age', 'height', 'weight']
features.append('gender')     # Add element
features.insert(1, 'income')  # Insert at position
features.remove('height')     # Remove element
features.pop()                # Remove last element
features.sort()               # Sort in-place
features.reverse()            # Reverse in-place

# Tuples - ordered, immutable
dimensions = (224, 224, 3)
height, width, channels = dimensions  # Unpacking

# Dictionaries - key-value pairs
model_config = {
    'layers': 5,
    'units': [64, 128, 256, 128, 64],
    'activation': 'relu',
    'dropout': 0.2
}
model_config['batch_size'] = 32  # Add key-value pair
model_config.get('optimizer', 'adam')  # Get with default
model_config.keys()               # Get all keys
model_config.values()             # Get all values
model_config.items()              # Get all key-value pairs

# Sets - unordered, unique elements
unique_labels = {'cat', 'dog', 'bird'}
unique_labels.add('fish')         # Add element
unique_labels.remove('bird')      # Remove element
unique_labels.discard('tiger')    # Remove if present
set1 = {'a', 'b', 'c'}
set2 = {'b', 'c', 'd'}
print(set1.union(set2))           # Union: {'a', 'b', 'c', 'd'}
print(set1.intersection(set2))    # Intersection: {'b', 'c'}
print(set1.difference(set2))      # Difference: {'a'}
```

### Loops
Loops iterate over sequences:

```python
# For loop with range
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# For loop with enumerate
for i, value in enumerate(['a', 'b', 'c']):
    print(f"Index {i}: {value}")

# For loop with zip
for name, score in zip(['model1', 'model2'], [0.92, 0.94]):
    print(f"{name}: {score}")

# Loop control statements
for i in range(10):
    if i < 3:
        continue  # Skip to next iteration
    if i > 7:
        break     # Exit loop
    print(i)  # 3, 4, 5, 6, 7

# While loop
counter = 0
while counter < 5:
    print(counter)
    counter += 1

# Infinite loop with break
while True:
    response = get_model_response()
    if response.success:
        break
```

### Docstrings
Docstrings document your code:

```python
def train_classifier(X, y, test_size=0.2, random_state=42):
    """
    Train a classification model and evaluate performance.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,)
        The target values.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        Controls the shuffling applied to the data before splitting.
        
    Returns
    -------
    model : object
        The trained classifier model.
    accuracy : float
        The accuracy score on the test set.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model, accuracy = train_classifier(X, y)
    >>> print(f"Model accuracy: {accuracy:.2f}")
    Model accuracy: 0.97
    """
    # Implementation here
    return model, accuracy
```

### Python Scripting
Creating standalone Python scripts:

```python
#!/usr/bin/env python3
"""
A script to train and evaluate an image classification model.
"""

import argparse
import logging
import os
import sys

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifier')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--model-type', type=str, default='resnet',
                        choices=['resnet', 'vgg', 'inception'],
                        help='Type of model architecture')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    return parser.parse_args()

# Set up logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Main function
def main():
    args = parse_args()
    logger = setup_logging()
    
    logger.info(f"Starting training with {args.model_type} model")
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Model training and evaluation code here
    logger.info("Training completed successfully")

# Execute if run as a script
if __name__ == "__main__":
    main()
```

## Linear Algebra Essentials

### NumPy
NumPy is fundamental for numerical computing:

```python
import numpy as np

# Creating arrays
a = np.array([1, 2, 3])                # 1D array
b = np.array([[1, 2, 3], [4, 5, 6]])   # 2D array
c = np.zeros((3, 3))                   # 3x3 array of zeros
d = np.ones((2, 4))                    # 2x4 array of ones
e = np.eye(3)                          # 3x3 identity matrix
f = np.random.rand(2, 2)               # Random values in [0, 1)
g = np.random.randn(2, 3)              # Random values from standard normal
h = np.arange(10)                      # [0, 1, ..., 9]
i = np.linspace(0, 1, 5)               # 5 evenly spaced points in [0, 1]

# Array operations
a + b                 # Element-wise addition
a * 3                 # Scalar multiplication
a * b                 # Element-wise multiplication
np.dot(a, b)          # Dot product
b.T                   # Transpose
np.linalg.inv(f)      # Matrix inverse
w, v = np.linalg.eig(f)  # Eigenvalues and eigenvectors

# Array manipulation
np.reshape(h, (2, 5))    # Reshape to 2x5
np.concatenate([a, h])   # Concatenate arrays
b[0, 1]                  # Index (row 0, column 1)
b[:, 0]                  # First column
b[0, :]                  # First row
np.where(h > 5)          # Indices where condition is true

# Broadcasting
j = np.array([[1], [2], [3]])  # 3x1 array
k = np.array([4, 5, 6])         # 1D array with 3 elements
j + k                           # Broadcasting to 3x3 array
```

### Matplotlib
Matplotlib visualizes data and results:

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.savefig('sine_wave.png', dpi=300)
plt.show()

# Multiple plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)  # 2 rows, 2 columns, plot 1
plt.plot(x, np.sin(x))
plt.title('sin(x)')

plt.subplot(2, 2, 2)  # 2 rows, 2 columns, plot 2
plt.plot(x, np.cos(x), 'g-')
plt.title('cos(x)')

plt.subplot(2, 2, 3)  # 2 rows, 2 columns, plot 3
plt.plot(x, np.sin(2*x), 'r--')
plt.title('sin(2x)')

plt.subplot(2, 2, 4)  # 2 rows, 2 columns, plot 4
plt.plot(x, np.cos(2*x), 'y:')
plt.title('cos(2x)')

plt.tight_layout()
plt.show()

# Scatter plot
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.title('Scatter Plot')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 3D plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Plot surface
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
ax.set_title('3D Surface Plot')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
```

### Vector Visualization
Visualizing vectors helps understanding:

```python
import numpy as np
import matplotlib.pyplot as plt

# Creating vectors
v1 = np.array([2, 3])
v2 = np.array([1, 2])

# Plotting vectors
plt.figure(figsize=(10, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Plot first vector
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')

# Plot second vector
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')

# Plot vector addition
v_sum = v1 + v2
plt.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='g', label='v1 + v2')

# Alternative way to show vector addition
plt.quiver(v1[0], v1[1], v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='m', label='v2 from v1')

# Set limits and labels
plt.xlim(-1, 5)
plt.ylim(-1, 6)
plt.grid(True)
plt.legend()
plt.title('Vector Visualization')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.axis('equal')
plt.show()
```

### Systems of Linear Equations
Linear equations are fundamental to AI:

```python
import numpy as np
import matplotlib.pyplot as plt

# Solving linear system: Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# Solve using numpy
x = np.linalg.solve(A, b)
print(f"Solution: x = {x[0]}, y = {x[1]}")

# Visualize the system
x_vals = np.linspace(0, 5, 100)

# First equation: 3x + y = 9 -> y = 9 - 3x
y1 = 9 - 3 * x_vals

# Second equation: x + 2y = 8 -> y = (8 - x) / 2
y2 = (8 - x_vals) / 2

plt.figure(figsize=(10, 8))
plt.plot(x_vals, y1, label='3x + y = 9', linewidth=2)
plt.plot(x_vals, y2, label='x + 2y = 8', linewidth=2)
plt.plot(x[0], x[1], 'ro', markersize=10, label=f'Solution ({x[0]}, {x[1]})')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()
plt.title('System of Linear Equations')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
```

### Linear Algebra
Linear algebra concepts for AI:

```python
import numpy as np

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(v1, v2)
print(f"Dot product: {dot_product}")

# Vector norm (magnitude)
norm_v1 = np.linalg.norm(v1)
print(f"Norm of v1: {norm_v1}")

# Cross product (for 3D vectors)
cross_product = np.cross(v1, v2)
print(f"Cross product: {cross_product}")

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition and subtraction
print("A + B =", A + B)
print("A - B =", A - B)

# Matrix multiplication
C = np.matmul(A, B)
print("A * B =", C)

# Matrix determinant
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}")

# Matrix inverse
inv_A = np.linalg.inv(A)
print("A^(-1) =", inv_A)
print("A * A^(-1) =", np.matmul(A, inv_A))  # Should be close to identity

# Matrix trace (sum of diagonal elements)
trace_A = np.trace(A)
print(f"Trace of A: {trace_A}")

# Matrix rank
rank_A = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank_A}")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)
print("U =", U)
print("S =", S)
print("V^T =", Vt)
```

### Matrix Multiplication
Understanding matrix multiplication:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define matrices
A = np.array([[1, 2, 3],
              