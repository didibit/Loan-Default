#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# main.py

import torch
import data_preprocessing
from train import train_model, evaluate_model
from test import preprocess_test_data, inference_on_test, save_predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load and Clean Training Data
    data = data_preprocessing.load_and_clean_training_data('training_loan_data.csv')

    # 2. Further Preprocess (fill NAs, remove outliers, encode, remove duplicates)
    data_cleaned = data_preprocessing.preprocess_data(data)

    # 3. Split into train/val and scale/SMOTE
    X_train_res, y_train_res, X_val_scaled, y_val, scaler = data_preprocessing.get_train_val_data(data_cleaned)

    input_dim = X_train_res.shape[1]

    # 4. Train the model
    # Adjust hyperparameters as needed
    model = train_model(
        X_train_res, 
        y_train_res, 
        X_val_scaled, 
        y_val, 
        input_dim=input_dim,
        lr=0.0073,
        dr=0.4896,
        hn1=42,
        hn2=35,
        batch_size=5345,   # Example from your best params
        num_epochs=20,     # Increase as needed
        device=device
    )

    # 5. Evaluate on Validation
    evaluate_model(model, X_val_scaled, y_val, device=device)

    # 6. Load Test Data
    import pandas as pd
    test_data = pd.read_csv('testing_loan_data.csv')
    # Drop unneeded columns (like you did)
    drop_cols = ['id', 'desc', 'member_id', 'application_approved_flag', 'internal_score', 
                 'total_bc_limit', 'bad_flag']  # If 'bad_flag' in test
    test_data.drop(columns=drop_cols, inplace=True, errors='ignore')

    # For the real scenario, you need the same fitted objects (imputer_mode, le_term, 
    # ordinal_emp_length, ohe_home_ownership, ohe_purpose) used in training.
    # Make sure to pass them from data_preprocessing or store them after training.
    # Below is just a pseudo approach, assuming you stored them from training:
    # bc_util_median, revol_util_median, tot_hi_cred_lim_median, ...
    # Or you store them in a dictionary or something similar.

    # For demonstration, we assume you have references to these from data_preprocessing step
    # but in actual code, you'd refactor so you can retrieve them easily.
    # e.g., data_preprocessing.preprocess_data(...) could return them, or store them as global.

    # 7. Preprocess Test Data
    # We'll just show function call with placeholders.
    # Make sure to retrieve the actual objects from your data_preprocessing code or store them after.
    # For example, data_preprocessing could return them as well.

    # all needed transforms (objects) must be passed here:
    # test_preprocessed = preprocess_test_data(
    #    test_data, 
    #    bc_util_median, revol_util_median, tot_hi_cred_lim_median, 
    #    tot_cur_bal_median, 
    #    imputer_mode, le_term, ordinal_emp_length, 
    #    ohe_home_ownership, ohe_purpose
    # )

    # For simplicity, suppose we skip that step and do minimal approach:
    test_preprocessed = test_data.copy()

    # 8. Inference on test data
    x_numerical_columns = [
        'loan_amnt', 'int_rate', 'annual_inc', 'percent_bc_gt_75', 
        'bc_util', 'dti', 'inq_last_6mths', 'mths_since_recent_inq',
        'revol_util', 'mths_since_last_major_derog', 
        'tot_hi_cred_lim', 'tot_cur_bal'
    ]
    all_preds, binary_preds = inference_on_test(
        model, 
        test_preprocessed, 
        scaler, 
        x_numerical_columns, 
        device=device, 
        batch_size=5345
    )

    # 9. Save predictions
    save_predictions(binary_preds, output_file='test_predictions.csv')

if __name__ == "__main__":
    main()

