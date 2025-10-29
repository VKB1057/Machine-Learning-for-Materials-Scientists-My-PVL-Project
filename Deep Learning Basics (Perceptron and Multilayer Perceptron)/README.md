Exercise 03 — Deep Learning Basics (Perceptron and Multilayer Perceptron)

Overview
This project introduces the foundational concepts of deep learning through the construction and analysis of a 
single-layer perceptron (SLP) and a multilayer perceptron (MLP). The exercise involves deriving forward and 
backpropagation equations, implementing them in Python, and verifying the results with TensorFlow.

Objectives
- Understand the mathematical basis of the perceptron and MLP.
- Derive and implement forward and backward propagation equations.
- Compare manual (NumPy) and framework-based (TensorFlow) implementations.
- Visualize activation functions and evaluate model accuracy on a simple dataset.

Implementation Workflow
1. Theoretical Derivation – Derived forward and backward propagation equations symbolically.
2. Single-Layer Perceptron – Implemented forward pass and weight updates using NumPy.
3. Activation Functions – Visualized sigmoid, tanh, and Leaky ReLU for comparison.
4. Backpropagation – Derived gradients for weights and biases using the chain rule.
5. MLP Extension – Added hidden layer and verified gradient flow correctness.
6. Validation – Compared outputs between manual and TensorFlow-based implementations.

Results Summary
- Achieved 96% training and 100% test accuracy on a two-class dataset.
- Verified theoretical and numerical consistency between manual and TensorFlow models.
- Demonstrated the reduction in loss after each backpropagation step.
- Confirmed the universal approximation capability of the MLP architecture.

Key Learnings
- Gained practical understanding of forward and backward passes in neural networks.
- Strengthened grasp of gradient computation, activation nonlinearity, and error propagation.
- Bridged theory and implementation using both mathematical derivation and code validation.

Tools & Libraries
Python 3.x, NumPy, Matplotlib, TensorFlow, Scikit-learn

Repository Structure
deeplearning_basics.ipynb       # Main Jupyter notebook
results/                        # Figures and visualizations
  ├── activation_functions.png
  ├── loss_reduction_curve.png
  └── mlp_architecture_diagram.png
README.txt

Example Plots
- Activation function curves (Sigmoid, Tanh, Leaky ReLU)
- Loss reduction after each training iteration
- Decision boundary visualization for MLP

Author
Vigneshwara Koka Balaji
M.Sc. Computational Materials Science, TU Bergakademie Freiberg
Email: vigneshwaraofficial@gmail.com
LinkedIn: https://www.linkedin.com/in/vigneshwarakb
