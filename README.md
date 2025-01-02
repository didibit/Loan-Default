# Loan-Default
## Build ANN model for binary classification

This repository contains an end-to-end pipeline for training a binary classification model on a financial dataset using PyTorch. It covers data preprocessing (handling missing values and outliers), model training with SMOTE for class imbalance, and inference on unseen data.

### Requirements
Python >= 3.8

PyTorch >= 2.0.1

bayes_opt (if needed)

NumPy

Pandas

scikit-learn

Matplotlib

Seaborn

imbalanced-learn (for SMOTE)

SciPy

### Steps
Firstly, I do exploratory data analysis (EDA). Steps to Follow:
  1. Load the training dataset and perform an initial data inspection.
  2. Examine data quality appropriately.
  3. Preprocess dataset (fill missing values, make feature selection, encode categorical features, detect outliers)
  4. Summarize findings, visualizations, and insights.

Secondly, I do Neural Network Implementation. Steps to follow:
  1. Build a neural network using PyTorch to classify the dataset.
  2. Apply Bayesian Optimization for hyperparameter tunning
  3. Train the model using training data
  4. Evaluate the model's performance on test data (Accuracy, AUC, AUPRC, F1 score)


###### How to run:
Option 1. JPM_SMOTE_HPC.ipynb is run in high performance computing (HPC). torch version: 2.2.2+cu121

Option 2. 
- Preprocessing and Training. You can run the script using `python main.py`.

By default, this will:
1. Load and preprocess the training data.
2. Train the model (with SMOTE for class imbalance).
3. Validate performance on a hold-out validation set.
4. Optionally save model weights to a .pth file.

- (Optional) Directly call training. You can run the script using `python train.py`.

- Generate Predictions on Test

If your main.py already calls the test pipeline, it will generate test_predictions.csv. If you want to run test inference separately, you might do `python test.py`.


### Results
Model Performance on Val:
- Accuracy: 0.7700;
- ROC AUC: 0.6158;
- AUPRC: 0.0992;
- F1 Score: 0.1606
