# pipe_fault_detection
Overview

This script implements and evaluates various machine learning models for classifying vibration data from a dataset. It uses both classical models (Random Forest, SVM, Logistic Regression, and kNN) and a deep learning model (CNN-LSTM) to compare performance on both clean and noisy data. The results from this study will contribute to research presented at ICCCIT 2025.

Purpose

The objective of this research is to assess the effectiveness of traditional and deep learning models on a vibration classification task, specifically comparing their performance in clean versus noisy data environments. This analysis aims to improve sensor-based vibration classification methods, which are vital in industrial and engineering applications.

Research Context

This work is part of a larger study for the International Conference on Computer and Communication Technology (ICCCIT 2025), where we will present a detailed comparison of classical and deep learning models for vibration data classification. The findings will inform best practices for model selection in environments where sensor noise is prevalent.

Files and Directories

balanced_dataset.csv: The input dataset containing vibration data (X, Y, Z) and class labels (class).

results/: Directory where results and confusion matrices will be saved.

vibration_model_results_readable.csv: Summary of model performances (accuracy, precision, recall, F1 score, etc.).

Model Confusion Matrices: Individual confusion matrices for each model saved as CSV files.

Key Features

Model Training and Evaluation:

Classical models: Random Forest, SVM, Logistic Regression, kNN.

Deep learning model: CNN-LSTM with Conv1D and LSTM layers.

Evaluation on both clean and noisy datasets.

Data Preprocessing:

Data is split into windows and normalized.

Stratified train-test-validation split ensures balanced class distribution.

Results:

The script computes accuracy, precision, recall, F1 score, and confusion matrices.

Model performance is evaluated for clean data and noisy data (simulating sensor noise).

Setup

Ensure you have the required libraries:

numpy

pandas

scikit-learn

tensorflow

Place the dataset (balanced_dataset.csv) in the working directory.

Run the script, and the results will be saved in the results/ directory.

Usage

Modify the dataset path in the script if necessary (data = pd.read_csv('balanced_dataset.csv')).

The models will be trained and evaluated, and results will be saved automatically.

The summary CSV will contain performance metrics for each model, while individual confusion matrices will be saved separately.
