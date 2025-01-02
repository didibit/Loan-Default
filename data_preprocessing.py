#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data_preprocessing.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import math

def clean_percentage(column: pd.Series) -> pd.Series:
    """
    Remove '%' from a percentage column and convert it to float.
    """
    return column.str.replace('%', '').astype(float)

def load_and_clean_training_data(csv_file: str) -> pd.DataFrame:
    """
    Load training data and perform initial cleaning (remove unneeded columns, handle missing values).
    Returns a cleaned pandas DataFrame.
    """
    data = pd.read_csv(csv_file, header=1)

    # Drop columns
    data = data.drop(columns=['id', 'desc', 'member_id', 'application_approved_flag'])

    # Clean % columns
    if data['int_rate'].dtype == 'object':
        data['int_rate'] = clean_percentage(data['int_rate'])
    if data['revol_util'].dtype == 'object':
        data['revol_util'] = clean_percentage(data['revol_util'])

    # Drop rows with missing 'bad_flag'
    data = data.dropna(subset=['bad_flag'])

    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform various preprocessing tasks: 
    - Remove outliers in `annual_inc` (threshold=300k).
    - Drop highly correlated features (internal_score, total_bc_limit).
    - Fill missing values.
    - Encode categorical columns.
    - Remove duplicates.
    Returns the cleaned DataFrame.
    """

    # Remove annual_inc outliers above 300k
    data_cleaned = data[data['annual_inc'] < 3e5].copy()

    # Drop highly correlated features
    drop_cols = ['internal_score', 'total_bc_limit']
    data_cleaned.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Fill missing numeric values with median or 0
    bc_util_median = data_cleaned['bc_util'].median()
    revol_util_median = data_cleaned['revol_util'].median()
    tot_hi_cred_lim_median = data_cleaned['tot_hi_cred_lim'].median()
    tot_cur_bal_median = data_cleaned['tot_cur_bal'].median()

    data_cleaned['percent_bc_gt_75'].fillna(0, inplace=True)
    data_cleaned['bc_util'].fillna(bc_util_median, inplace=True)
    data_cleaned['mths_since_recent_inq'].fillna(0, inplace=True)
    data_cleaned['revol_util'].fillna(revol_util_median, inplace=True)
    data_cleaned['mths_since_last_major_derog'].fillna(0, inplace=True)
    data_cleaned['tot_hi_cred_lim'].fillna(tot_hi_cred_lim_median, inplace=True)
    data_cleaned['tot_cur_bal'].fillna(tot_cur_bal_median, inplace=True)

    # Impute categorical missing values using the most frequent value
    categorical_columns = data_cleaned.select_dtypes(include=['object']).columns.tolist()
    imputer_mode = SimpleImputer(strategy='most_frequent')
    data_cleaned[categorical_columns] = imputer_mode.fit_transform(data_cleaned[categorical_columns])

    # ---- ENCODING ----
    # LabelEncoder for 'term'
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
    le_term = LabelEncoder()
    data_cleaned['term_encoded'] = le_term.fit_transform(data_cleaned['term'])

    # OrdinalEncoder for 'emp_length'
    emp_length_order = [
        '< 1 year', '1 year', '2 years', '3 years', '4 years',
        '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
    ]
    ordinal_emp_length = OrdinalEncoder(categories=[emp_length_order])
    data_cleaned['emp_length_encoded'] = ordinal_emp_length.fit_transform(data_cleaned[['emp_length']])

    # OneHotEncode 'home_ownership'
    ohe_home_ownership = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    home_ownership_ohe = ohe_home_ownership.fit_transform(data_cleaned[['home_ownership']])
    home_ownership_df = pd.DataFrame(
        home_ownership_ohe, 
        columns=ohe_home_ownership.get_feature_names_out(['home_ownership'])
    )

    # OneHotEncode 'purpose'
    ohe_purpose = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    purpose_ohe = ohe_purpose.fit_transform(data_cleaned[['purpose']])
    purpose_df = pd.DataFrame(
        purpose_ohe,
        columns=ohe_purpose.get_feature_names_out(['purpose'])
    )

    # Concatenate the new OHE columns
    data_cleaned = pd.concat([data_cleaned, home_ownership_df, purpose_df], axis=1)

    # Drop original categorical columns
    drop_cols = ['term', 'emp_length', 'home_ownership', 'purpose']
    data_cleaned.drop(drop_cols, axis=1, inplace=True, errors='ignore')

    # Remove duplicates
    data_cleaned.drop_duplicates(inplace=True)

    return data_cleaned

def get_train_val_data(data: pd.DataFrame):
    """
    Split the data into train/validation sets, scale numeric columns, apply SMOTE.
    Returns (X_train, X_val, y_train, y_val, scaler, various fitted objects).
    """

    # Separate features and target
    X = data.drop('bad_flag', axis=1)
    y = data['bad_flag']

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Identify numeric columns for scaling
    numeric_cols = [
        'loan_amnt', 'int_rate', 'annual_inc', 'percent_bc_gt_75', 
        'bc_util', 'dti', 'inq_last_6mths', 'mths_since_recent_inq',
        'revol_util', 'mths_since_last_major_derog', 
        'tot_hi_cred_lim', 'tot_cur_bal'
    ]

    # StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()

    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

    # SMOTE for imbalance
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, y_train_resampled, X_val_scaled, y_val, scaler

