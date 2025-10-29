Exercise 04 — MLP Regression on Cantilever Beam Data

Overview
This project focuses on training a Multi-Layer Perceptron (MLP) to predict displacements in a cantilever beam 
based on input coordinates (x, y). The exercise includes hyperparameter tuning, comparison with classical 
interpolation methods, and a "break-even" analysis to determine the optimal training size for the MLP.

Objectives
- Train an MLP model to predict displacements (ux, uy) from coordinates (x, y) using scikit-learn's MLPRegressor.
- Perform a grid search for hyperparameter optimization.
- Compare MLP performance with traditional interpolation methods (nearest-neighbor, linear, cubic).
- Perform a "break-even" analysis to determine the minimal training set size at which the MLP outperforms interpolation.

Dataset
- FEM-computed displacement field for a cantilever beam with tip load.
- 500 collocation points: (x, y) as inputs and (ux, uy) as outputs.
- Split into training (70%), validation (20%), and testing (10%).

Implementation Workflow
1. Preprocessing – Load dataset, split into train/val/test, standardize features.
2. MLP Training – Train MLP using default settings, followed by hyperparameter tuning via grid search.
3. Interpolation Comparison – Train nearest-neighbor, linear, and cubic interpolation models.
4. Break-even Analysis – Test MLP vs. interpolation methods across various training set sizes (n ∈ {50, 100, 200, 300, 350}).
5. Evaluation – Compute MSE for all models, compare results, and plot MSE vs. training set size.

Results Summary
- MLP consistently outperformed all interpolation methods once training set size reached n ≥ 50.
- Achieved near-zero MSE with the MLP as the training set size increased.
- Hyperparameter tuning improved the MLP's performance, with ReLU activation and Adam optimizer yielding the best results.

Key Learnings
- Demonstrated how **MLP regression** can outperform **classical interpolation methods** in terms of prediction accuracy.
- Gained experience in **hyperparameter tuning** using grid search and evaluated the impact of training size on model performance.
- Strengthened skills in **data preprocessing**, model comparison, and **evaluation metrics**.

Tools & Libraries
Python 3.x, NumPy, Pandas, Matplotlib, Scikit-learn

Repository Structure
cantilever_beam_data.csv        # Dataset with FEM displacements
mlp_regression.ipynb            # Main Jupyter notebook for MLP training and evaluation
results/                        # Figures and tables
  ├── break_even_plot.png
  ├── model_comparison.png
  └── mse_vs_training_size.png
README.txt

Example Plots
- Test MSE vs Training Set Size
- MLP vs Interpolation Comparison
- Model performance across training sizes

Author
Vigneshwara Koka Balaji
M.Sc. Computational Materials Science, TU Bergakademie Freiberg
Email: vigneshwaraofficial@gmail.com
LinkedIn: https://www.linkedin.com/in/vigneshwarakb
