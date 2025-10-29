Exercise 02 — K-Means Clustering on NEU Steel Surface Defects

Overview
This project applies unsupervised learning to classify steel surface defect images using K-Means clustering and PCA. 
The goal is to group six defect types based on visual features and analyze clustering accuracy before and after dimensionality reduction.

Objectives
- Preprocess NEU dataset images (resize, normalize, flatten).
- Perform K-Means clustering (K=6) on 4096-dimensional features.
- Map clusters to defect classes using majority voting.
- Apply PCA for dimensionality reduction and repeat clustering.
- Evaluate classification accuracy vs number of PCA components.

Dataset
- NEU Steel Surface Defect Dataset: 6 classes × 300 images (1800 total).
- Used 50 training and 20 validation images per class.
- Image resolution: resized to 64×64 pixels → flattened to 4096 features.

Implementation Workflow
1. Preprocessing – Load and normalize grayscale images using OpenCV or PIL.
2. Flatten and label images by defect class.
3. Apply K-Means clustering (init='k-means++', random_state=42).
4. Compute confusion matrices and classification accuracy for train/test.
5. Apply PCA (ℓ ∈ {5,10,20,30,40,50,64}) and repeat clustering.
6. Plot test error vs number of PCA components and identify elbow point.

Results Summary
- Achieved clear separation between defect classes after PCA.
- Optimal dimensionality ≈ 20–30 components (best trade-off of accuracy and complexity).
- PCA improved both runtime and cluster compactness.
- Identified diminishing returns beyond 40 components.

Key Learnings
- PCA helps improve K-Means performance by removing noise and redundant dimensions.
- Unsupervised clustering can reveal meaningful patterns even without labeled data.
- Understood workflow of dimensionality tuning and accuracy evaluation.

Tools & Libraries
Python 3.x, NumPy, Pandas, Matplotlib, Scikit-learn, OpenCV/PIL

Repository Structure
train/                      # Training images (6 folders, 50 each)
validation/                 # Validation images (6 folders, 20 each)
kmeans_clustering.ipynb     # Main notebook
results/                    # Accuracy tables and plots
  ├── confusion_matrix_train.png
  ├── confusion_matrix_test.png
  └── error_vs_pca_components.png
README.txt

Example Plots
- Confusion matrix (train/test)
- Test error vs PCA components
- 3D PCA scatter plot (6 clusters visualized)

Author
Vigneshwara Koka Balaji
M.Sc. Computational Materials Science, TU Bergakademie Freiberg
Email: vigneshwaraofficial@gmail.com
LinkedIn: https://www.linkedin.com/in/vigneshwarakb
