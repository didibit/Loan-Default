#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# test.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def preprocess_test_data(
    test_df: pd.DataFrame, 
    bc_util_median, revol_util_median, 
    tot_hi_cred_lim_median, tot_cur_bal_median, 
    imputer_mode, le_term, ordinal_emp_length, 
    ohe_home_ownership, ohe_purpose
):
    """
    Apply the same preprocessing steps to the test DataFrame.
    (Fill missing values, encode, drop columns, etc.)
    Returns the transformed test DataFrame.
    """

    # Clean percentage columns
    def clean_percentage(column):
        return column.str.replace('%', '').astype(float)

    if test_df['int_rate'].dtype == 'object':
        test_df['int_rate'] = clean_percentage(test_df['int_rate'])
    if test_df['revol_util'].dtype == 'object':
        test_df['revol_util'] = clean_percentage(test_df['revol_util'])

    # Fill numeric missing values with the same strategy as training
    test_df['percent_bc_gt_75'].fillna(0, inplace=True)
    test_df['bc_util'].fillna(bc_util_median, inplace=True)
    test_df['mths_since_recent_inq'].fillna(0, inplace=True)
    test_df['revol_util'].fillna(revol_util_median, inplace=True)
    test_df['mths_since_last_major_derog'].fillna(0, inplace=True)
    test_df['tot_hi_cred_lim'].fillna(tot_hi_cred_lim_median, inplace=True)
    test_df['tot_cur_bal'].fillna(tot_cur_bal_median, inplace=True)

    # Impute categorical columns
    categorical_columns = test_df.select_dtypes(include=['object']).columns.tolist()
    test_df[categorical_columns] = imputer_mode.transform(test_df[categorical_columns])

    # Encode 'term'
    test_df['term_encoded'] = le_term.transform(test_df['term'])

    # Encode 'emp_length'
    test_df['emp_length_encoded'] = ordinal_emp_length.transform(test_df[['emp_length']])

    # OneHotEncode 'home_ownership'
    home_ownership_ohe_test = ohe_home_ownership.transform(test_df[['home_ownership']])
    home_ownership_df_test = pd.DataFrame(
        home_ownership_ohe_test, 
        columns=ohe_home_ownership.get_feature_names_out(['home_ownership'])
    )

    # OneHotEncode 'purpose'
    purpose_ohe_test = ohe_purpose.transform(test_df[['purpose']])
    purpose_df_test = pd.DataFrame(
        purpose_ohe_test,
        columns=ohe_purpose.get_feature_names_out(['purpose'])
    )

    # Concatenate
    test_df = pd.concat([test_df, home_ownership_df_test, purpose_df_test], axis=1)

    # Drop original categorical columns
    drop_cols = ['term', 'emp_length', 'home_ownership', 'purpose']
    test_df.drop(drop_cols, axis=1, inplace=True, errors='ignore')

    return test_df

def inference_on_test(model, test_df, scaler, x_numerical_columns, device='cpu', batch_size=512):
    """
    Scale test_df numeric columns, run the model, return predictions and probabilities.
    """
    # Scale numeric columns
    test_df[x_numerical_columns] = scaler.transform(test_df[x_numerical_columns])
    # Convert to tensor
    X_test_tensor = torch.tensor(test_df.values, dtype=torch.float32).to(device)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch_X = batch[0]
            outputs = model(batch_X)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds).ravel()

    binary_preds = (all_preds >= 0.5).astype(int)
    return all_preds, binary_preds

def save_predictions(binary_preds, output_file='test_predictions.csv'):
    """
    Save the final predictions to CSV.
    """
    predictions_df = pd.DataFrame({'bad_flag': binary_preds})
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}.")

