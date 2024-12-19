# Loan-Default
Build ANN model for binary classification.

The task is to  perform exploratory data analysis (EDA) on a mixed-feature dataset and build a simple binary classification neural network using PyTorch. The data is highly imbalanced with missing values.

The whole exercise shouldn’t take a significant amount of time.

Firstly, I do exploratory data analysis (EDA). Steps to Follow:
  1. Load the training dataset and perform an initial data inspection.
  2. Examine data quality appropriately.
  3. Preprocess dataset (fill missing values, make feature selection, encode categorical features, detect outliers)
  4. Summarize findings, visualizations, and insights.

Secondly, I do Neural Network Implementation: Steps to follow
  1. Build a neural network using PyTorch to classify the dataset.
  2. Aplly Bayesian Optimization for hyperparameter tunning
  3. Train the model using training data
  4. Evaluate the model's performance on test data (Accuracy, AUC, AUPRC, F1 score)


How to run:
JPM_SMOTE_HPC.ipynb is run in high performance computing (HPC). torch version: 2.2.2+cu121

Model Performance on Val:
Accuracy: 0.7700
ROC AUC: 0.6158
AUPRC: 0.0992
F1 Score: 0.1606
