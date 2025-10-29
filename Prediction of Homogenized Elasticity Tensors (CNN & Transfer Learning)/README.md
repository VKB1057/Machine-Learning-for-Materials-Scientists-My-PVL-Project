Exercise 05 — Prediction of Homogenized Elasticity Tensors (CNN & Transfer Learning)

Overview
This project explores the application of deep learning techniques to predict homogenized elasticity tensors from 
image-based microstructures. The exercise involves building a custom CNN, tuning its hyperparameters, and 
applying transfer learning using pretrained models (VGG16, ResNet50, DenseNet121) to improve prediction accuracy.

Objectives
- Build and train a custom CNN to predict the 9 components of the elasticity tensor from 64x64 grayscale images.
- Optimize the CNN using hyperparameter tuning (filters, kernel size, dropout rate, learning rate).
- Apply transfer learning (VGG16, ResNet50, DenseNet121) by freezing pretrained layers and training regression heads.
- Compare the performance of the custom CNN and transfer learning models (head-only and fine-tuned).
- Evaluate the impact of model architecture and fine-tuning on accuracy.

Dataset
- 520 grayscale images (64x64 px) representing binary-phase microstructures.
- Each image is mapped to a 9-dimensional vector of elasticity tensor components (C11, C22, C33, etc.).

Implementation Workflow
1. Data Preprocessing – Load images, normalize, convert to RGB for transfer learning compatibility.
2. Custom CNN – Build and train the CNN with multiple convolutional layers, followed by fully connected layers.
3. Hyperparameter Tuning – Optimize the CNN architecture by varying filters, kernel sizes, dropout, and learning rate.
4. Transfer Learning – Implement transfer learning with VGG16, ResNet50, and DenseNet121; train only the regression head or fine-tune all layers.
5. Comparison – Evaluate and compare the performance of all models using MSE and accuracy metrics.

Results Summary
- The custom CNN with 32 filters and 5x5 kernels achieved the lowest test MSE (~8.41e-15).
- Transfer learning using VGG16 (head-only) and ResNet50 (fine-tuned) performed similarly, achieving comparable results to the custom CNN.
- Fine-tuning in DenseNet121 resulted in performance degradation, possibly due to overfitting with the small dataset.

Key Learnings
- Transfer learning significantly improved performance on small datasets, especially when pretrained models were used with frozen layers.
- Custom CNNs can be highly effective when tuned properly, achieving competitive results with fewer resources than large pretrained models.
- The importance of **model regularization** and **fine-tuning** was clearly observed in this exercise, especially when working with small, domain-specific datasets.

Tools & Libraries
Python 3.x, NumPy, Pandas, Matplotlib, TensorFlow, Keras, Scikit-learn

Repository Structure
images/                         # Image dataset for training and testing
model_training.ipynb            # Main Jupyter notebook for training the CNN and transfer learning models
results/                        # Model performance outputs
  ├── cnn_performance.png
  ├── transfer_learning_performance.png
  └── model_comparison.png
README.txt

Example Plots
- Test MSE for Custom CNN and Transfer Learning Models
- Training and Validation Loss Curves for Transfer Learning Models
- Model architecture diagrams for custom CNN and pretrained backbones

Author
Vigneshwara Koka Balaji
M.Sc. Computational Materials Science, TU Bergakademie Freiberg
Email: vigneshwaraofficial@gmail.com
LinkedIn: https://www.linkedin.com/in/vigneshwarakb
