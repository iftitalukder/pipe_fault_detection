# Pipe Fault Detection

## Overview

This script implements and evaluates various machine learning models for classifying vibration data from a dataset. It uses both classical models (Random Forest, SVM, Logistic Regression, and kNN) and a deep learning model (CNN-LSTM) to compare performance on both clean and noisy data.


## Files and Directories

- **balanced_dataset.csv**: The input dataset containing vibration data (X, Y, Z) and class labels (class).
- **results/**: Directory where results and confusion matrices will be saved.
- **vibration_model_results_readable.csv**: Summary of model performances (accuracy, precision, recall, F1 score, etc.).
- **Model Confusion Matrices**: Individual confusion matrices for each model saved as CSV files.

## Key Features

### Model Training and Evaluation:
- Classical models: Random Forest, SVM, Logistic Regression, kNN.
- Deep learning model: CNN-LSTM with Conv1D and LSTM layers.
- Evaluation on both clean and noisy datasets.

### Data Preprocessing:
- Data is split into windows and normalized.
- Stratified train-test-validation split ensures balanced class distribution.

### Results:
- The script computes accuracy, precision, recall, F1 score, and confusion matrices.
- Model performance is evaluated for clean data and noisy data (simulating sensor noise).

## Setup

Ensure you have the required libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`

Place the dataset (`balanced_dataset.csv`) in the working directory.

Run the script, and the results will be saved in the **results/** directory.

## Usage

1. Modify the dataset path in the script if necessary (`data = pd.read_csv('balanced_dataset.csv')`).
2. The models will be trained and evaluated, and results will be saved automatically.
3. The summary CSV will contain performance metrics for each model, while individual confusion matrices will be saved separately.

## Notes

The pipeline is designed to be flexible and reproducible, allowing for easy comparison of classical and deep learning models. However, please note that results may vary depending on several factors, such as model configurations, data splits, and the noise characteristics used during evaluation. While the overall methodology remains consistent, differences in hyperparameters or environmental noise may lead to varying outcomes.

This script is structured for easy extension, so you can experiment with different configurations or datasets.
