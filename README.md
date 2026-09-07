# AutoANN — Automated Neural Network Prediction Platform

AutoANN is a beginner-friendly machine learning web application that automates the creation of Artificial Neural Networks (ANN) for tabular data. Just upload a CSV, select your target column, and let AutoANN do the preprocessing, model architecture generation, training, and evaluation.

## Setup Instructions

1. **Activate the virtual environment**:
   Navigate to the project root (`C:\Users\saiki\.gemini\antigravity\scratch\autoann`) and activate the venv:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Run the Flask application**:
   ```powershell
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000` in your web browser.

## Features Included
1. **CSV Upload & Dataset Overview**: Automatic detection of data types, missing values, duplicates, and unique elements.
2. **Preprocessing Pipeline**: 
    - Numerical: Median imputation + Standardization (StandardScaler).
    - Categorical: Most frequent imputation + One-Hot Encoding.
    - Data leakage prevention by fitting only on train set.
3. **Automated Neural Network Builder**:
    - Generates customized Keras/TensorFlow architectures depending on Classification (binary/multiclass) or Regression.
4. **Training & Evaluation**:
    - Computes Accuracy, F1 for Classification.
    - Computes RMSE, R² for Regression.
5. **Architectural Explainability**:
    - Explains exactly what the inputs are, the node structure, activations, optimizers, and loss functions in the dashboard.

## Next Steps
This version successfully implements Phases 1 to 4 and 6 (Core backend engine, dynamic ANN structures, UI/UX). You can extend with custom chart visualisations and the live Prediction interface building upon the existing robust backend pipeline.
