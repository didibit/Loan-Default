import torch
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc)
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# read data
data = pd.read_csv('training_loan_data.csv', header=1)
data.head()

data.describe()

# application_approved_flag has no variation, remove it
# id and member_id is not used for model building
#desc's info is included in purpose
data = data.drop(columns=['id', 'desc', 'member_id', 'application_approved_flag'])
data.head()

data.info()

# Function to clean percentage columns
def clean_percentage(column):
    return column.str.replace('%', '').astype(float)

# Apply the function to 'int_rate' and 'revol_rate' if they are strings
if data['int_rate'].dtype == 'object':
    data['int_rate'] = clean_percentage(data['int_rate'])

if data['revol_util'].dtype == 'object':
    data['revol_util'] = clean_percentage(data['revol_util'])


# visualize 'bad_flag' (a binary column):
count_values = data['bad_flag'].value_counts()
ratio_values = data['bad_flag'].value_counts(normalize=True)

# Print the counts and ratios
print("Counts:\n", count_values)
print("\nRatios:\n", ratio_values)

# Plot the distribution as a bar chart
plt.figure(figsize=(6,4))
count_values.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of 'bad_flag'")
plt.xlabel("Flag Value")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Check for missing values in 'bad_flag'
missing_bad_flag = data['bad_flag'].isnull().sum()
total_rows = len(data)

print(f"Total Rows: {total_rows}")
print(f"Missing 'bad_flag' Values: {missing_bad_flag}")
print(f"Percentage Missing: {missing_bad_flag / total_rows * 100:.2f}%")


# Drop rows where 'bad_flag' is missing
data = data.dropna(subset=['bad_flag'])

# Verify the removal
new_missing_bad_flag = data['bad_flag'].isnull().sum()
print(f"Missing 'bad_flag' after dropping: {new_missing_bad_flag}")
print(f"Rows after Dropping: {len(data)}")


## check missing values in all columns
missing_values_count = data.isnull().sum()

# Display the counts of missing values
print("Missing Values per Column:")
print(missing_values_count)

## check correlation
features = data.iloc[:, :-1]  # all columns except the last one
target = data.iloc[:, -1]     # just the last column

# Create a correlation heatmap
plt.figure(figsize=(12,12))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
sns.heatmap(features.corr(), annot=True, cmap='viridis', fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

features = ['loan_amnt', 'internal_score']
plt.figure(figsize=(15, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(1, 2, i)
    sns.histplot(data=data[data['bad_flag'] == 0], x=feature, bins=30, color='blue', 
                 label='Not Bad (0)', kde=False, alpha=0.6)
    sns.histplot(data=data[data['bad_flag'] == 1], x=feature, bins=30, color='red', 
                 label='Bad (1)', kde=False, alpha=0.6)
    plt.title(f'Histogram of {feature} by Bad Flag')
    plt.xlabel(feature.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.legend(title='Bad Flag')
    
plt.tight_layout()
plt.show()


features = ['total_bc_limit', 'tot_hi_cred_lim']
plt.figure(figsize=(15, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(1, 2, i)
    sns.histplot(data=data[data['bad_flag'] == 0], x=feature, bins=30, color='blue', 
                 label='Not Bad (0)', kde=False, alpha=0.6)
    sns.histplot(data=data[data['bad_flag'] == 1], x=feature, bins=30, color='red', 
                 label='Bad (1)', kde=False, alpha=0.6)
    plt.title(f'Histogram of {feature} by Bad Flag')
    plt.xlabel(feature.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.legend(title='Bad Flag')
    
plt.tight_layout()
plt.show()


# Box plot
features = ['loan_amnt', 'internal_score']

# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax, feature in zip(axes, features):
    sns.boxplot(x='bad_flag', y=feature, data=data, ax=ax, palette='Set2')
    ax.set_title(f'Box Plot of {feature.replace("_", " ").title()} by Bad Flag', fontsize=14)
    ax.set_xlabel('Bad Flag', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()


# Define the features to plot
features = ['total_bc_limit', 'tot_hi_cred_lim']

# Set up the matplotlib figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax, feature in zip(axes, features):
    sns.boxplot(x='bad_flag', y=feature, data=data, ax=ax, palette='Set2')
    ax.set_title(f'Box Plot of {feature.replace("_", " ").title()} by Bad Flag', fontsize=14)
    ax.set_xlabel('Bad Flag', fontsize=12)
    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()


data.groupby(by='bad_flag')['loan_amnt'].describe()

data.groupby(by='bad_flag')['internal_score'].describe()

data.groupby(by='bad_flag')['total_bc_limit'].describe()

data.groupby(by='bad_flag')['tot_hi_cred_lim'].describe()

## scatter plot of 'loan_amnt' and 'internal_score'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='loan_amnt', y='internal_score', data=data, color='blue', alpha=0.6)
plt.title('Scatter Plot of Loan Amount vs. Internal Score', fontsize=16)
plt.xlabel('Loan Amount', fontsize=14)
plt.ylabel('Internal Score', fontsize=14)
plt.grid(True)
plt.show()

## scatter plot of 'total_bc_limit'and 'tot_hi_cred_lim'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bc_limit', y='tot_hi_cred_lim', data=data, color='blue', alpha=0.6)
plt.title('Scatter Plot of total_bc_limit vs. tot_hi_cred_lim', fontsize=16)
plt.xlabel('total_bc_limit', fontsize=14)
plt.ylabel('tot_hi_cred_lim', fontsize=14)
plt.grid(True)
plt.show()

import math
y = 'bad_flag'

# Automatically select all numeric columns excluding the target
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
x_vars = [col for col in numeric_cols if col != y]

print("Features to plot:", x_vars)

def generate_scatter_plots(data, x_vars, y, cols=3, figsize=(20, 15)):

    num_vars = len(x_vars)
    rows = math.ceil(num_vars / cols)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of multiple rows

    for i, x in enumerate(x_vars):
        sns.scatterplot(x=x, y=y, data=data, ax=axes[i], alpha=0.6, edgecolor='w', palette='viridis')
        axes[i].set_title(f'{x} vs {y}', fontsize=14)
        axes[i].set_xlabel(x.replace('_', ' ').title(), fontsize=12)
        axes[i].set_ylabel(y.replace('_', ' ').title(), fontsize=12)
        axes[i].grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    
# Call the function with your data
generate_scatter_plots(data, x_vars, y, cols=3, figsize=(20, 15))


def generate_histogram_plots(data, x_vars, y, cols=3, figsize=(20, 15), bins=50):

    num_vars = len(x_vars)
    rows = math.ceil(num_vars / cols)
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of multiple rows
    
    for i, x in enumerate(x_vars):
        sns.histplot(data=data, x=x, hue=y, bins=bins, ax=axes[i], palette='Set2', kde=False, multiple='stack')
        axes[i].set_title(f'Histogram of {x.replace("_", " ").title()} by {y.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel(x.replace("_", " ").title(), fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Call the function with your data
generate_histogram_plots(data, x_vars, y, cols=3, figsize=(20, 15), bins=50)


from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(data['annual_inc']))
outliers_z = np.where(z_scores > 3)
print(f"Number of outliers detected by Z-score method: {len(outliers_z[0])}")

z_scores = np.abs(stats.zscore(data['tot_hi_cred_lim']))
outliers_z = np.where(z_scores > 3)
print(f"Number of outliers detected by Z-score method: {len(outliers_z[0])}")

z_scores = np.abs(stats.zscore(data['tot_cur_bal']))
outliers_z = np.where(z_scores > 3)
print(f"Number of outliers detected by Z-score method: {len(outliers_z[0])}")

# Filter data where annual_inc <= 300,000 and check distribution
filtered_data = data[data['annual_inc'] <= 3e5]
x = 'annual_inc'
y = 'bad_flag'

# Set the figure size
plt.figure(figsize=(10, 6))

# Create a histogram for each loan_status
sns.histplot(data=filtered_data, x=x, hue=y, bins=50, alpha=0.3, palette='Set2', multiple='layer')

# Customize the plot
plt.title("Bad Flag by Annual Income (<= \$300,000/Year)", fontsize=16)
plt.xlabel("Annual Income", fontsize=14)
plt.ylabel("Loan Counts", fontsize=14)
plt.legend(title=y, loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()


data.loc[data.annual_inc >= 3e5, 'bad_flag'].value_counts()

### check box plot
def generate_box_plots(data, x_vars, y, cols=2, figsize=(15, 5)):

    num_vars = len(x_vars)
    rows = math.ceil(num_vars / cols)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of multiple rows

    for i, x in enumerate(x_vars):
        sns.boxplot(x=y, y=x, data=data, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Box Plot of {x.replace("_", " ").title()} by {y.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel(y.replace("_", " ").title(), fontsize=12)
        axes[i].set_ylabel(x.replace("_", " ").title(), fontsize=12)
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    
# Call the function
generate_box_plots(data, x_vars, y, cols=2, figsize=(25, 25))


# Initialize list to store outlier indices
outlier_indices_iqr = []

for col in x_vars:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
    outlier_indices_iqr.extend(outliers)

# Remove duplicates
outlier_indices_iqr = list(set(outlier_indices_iqr))

print(f"Number of outliers detected by IQR method: {len(outlier_indices_iqr)}")


total_records = data.shape[0]
num_outliers_iqr = 31481  # Replace with your actual count
proportion_iqr = (num_outliers_iqr / total_records) * 100
print(f"Proportion of outliers detected by IQR method: {proportion_iqr:.2f}%")




# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Exclude the target variable if it's categorical
target = 'bad_flag'
categorical_cols = [col for col in categorical_cols if col != target]

print("Categorical Columns to Plot:", categorical_cols)

def generate_categorical_bar_plots(data, categorical_cols, target, cols=2, figsize=(15, 10)):

    num_vars = len(categorical_cols)
    rows = math.ceil(num_vars / cols)
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of multiple rows
    
    for i, col in enumerate(categorical_cols):
        sns.countplot(x=col, hue=target, data=data, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{col.replace("_", " ").title()} by {target.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel(col.replace("_", " ").title(), fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].legend(title=target.replace("_", " ").title())
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

    
generate_categorical_bar_plots(data, categorical_cols, target, cols=2, figsize=(15, 15))

# Define the threshold
threshold = 3e5  # 1,000,000

# Remove outliers
data_cleaned = data[data['annual_inc'] < threshold].copy()

# Reset the index if necessary
data_cleaned.reset_index(drop=True, inplace=True)

# Display the number of records before and after removal
print(f"Original dataset size: {data.shape[0]}")
print(f"Cleaned dataset size: {data_cleaned.shape[0]}")


data_cleaned = data_cleaned.drop(columns=['internal_score', 'total_bc_limit'])

## check missing value
for column in data_cleaned.columns:
    if data_cleaned[column].isna().sum() != 0:
        missing = data_cleaned[column].isna().sum()
        portion = (missing / data_cleaned.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

# Select numerical columns (integer and float types)
numerical_columns = data_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Display the list of numerical columns
print("Numerical Columns Before Encoding:")
print(numerical_columns)


# Select categorical columns (obj types)
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns.tolist()

# Display the list of numerical columns
print("Categorical Columns Before Encoding:")
print(categorical_columns)

# 1. Fill 'percent_bc_gt_75' with 0
data_cleaned['percent_bc_gt_75'].fillna(0, inplace=True)

# 2. Fill 'bc_util' with the median
bc_util_median = data_cleaned['bc_util'].median()
data_cleaned['bc_util'].fillna(bc_util_median, inplace=True)

# 3. Fill 'mths_since_recent_inq' with 0
data_cleaned['mths_since_recent_inq'].fillna(0, inplace=True)

# 4. Fill 'revol_util' with the median
revol_util_median = data_cleaned['revol_util'].median()
data_cleaned['revol_util'].fillna(revol_util_median, inplace=True)

# 5. Fill 'mths_since_last_major_derog' with 0
data_cleaned['mths_since_last_major_derog'].fillna(0, inplace=True)

# 6. Fill 'tot_hi_cred_lim' with the median
tot_hi_cred_lim_median = data_cleaned['tot_hi_cred_lim'].median()
data_cleaned['tot_hi_cred_lim'].fillna(tot_hi_cred_lim_median, inplace=True)

# 7. Fill 'tot_cur_bal' with the median
tot_cur_bal_median = data_cleaned['tot_cur_bal'].median()
data_cleaned['tot_cur_bal'].fillna(tot_cur_bal_median, inplace=True)

# Verify that there are no more missing values in these columns
print(data_cleaned[['percent_bc_gt_75', 'bc_util', 'mths_since_recent_inq', 
            'revol_util', 'mths_since_last_major_derog', 
            'tot_hi_cred_lim', 'tot_cur_bal']].isnull().sum())


## For categorical variables, use mode (most frequent) to fill
# Verify that there are no more missing values in categorical columns

from sklearn.impute import SimpleImputer
# Initialize SimpleImputer for categorical columns
imputer_mode = SimpleImputer(strategy='most_frequent')

# Apply imputer to categorical columns
data_cleaned[categorical_columns] = imputer_mode.fit_transform(data_cleaned[categorical_columns])

print(data_cleaned[categorical_columns].isnull().sum())

### Encode categorical variables (switch to numerical vars)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# 'emp_length' is ordinal, define the order
emp_length_order = [
    '< 1 year', '1 year', '2 years', '3 years', '4 years',
    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
]

# Initialize OrdinalEncoder for 'emp_length'
ordinal_emp_length = OrdinalEncoder(categories=[emp_length_order])

# Initialize OneHotEncoders for nominal features
ohe_home_ownership = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
ohe_purpose = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

# Initialize LabelEncoder for 'term' if treating it as binary
le_term = LabelEncoder()

# Encode 'term' using LabelEncoder
data_cleaned['term_encoded'] = le_term.fit_transform(data_cleaned['term'])

# Encode 'emp_length' using OrdinalEncoder
data_cleaned['emp_length_encoded'] = ordinal_emp_length.fit_transform(data_cleaned[['emp_length']])

# One-Hot Encode 'home_ownership'
home_ownership_ohe = ohe_home_ownership.fit_transform(data_cleaned[['home_ownership']])
home_ownership_df = pd.DataFrame(
    home_ownership_ohe,
    columns=ohe_home_ownership.get_feature_names_out(['home_ownership'])
)

# One-Hot Encode 'purpose'
purpose_ohe = ohe_purpose.fit_transform(data_cleaned[['purpose']])
purpose_df = pd.DataFrame(
    purpose_ohe,
    columns=ohe_purpose.get_feature_names_out(['purpose'])
)

# Concatenate the One-Hot Encoded columns with the original DataFrame
data_cleaned = pd.concat([data_cleaned, home_ownership_df, purpose_df], axis=1)

# Drop the original categorical columns
data_cleaned.drop(['term', 'emp_length', 'home_ownership', 'purpose'], axis=1, inplace=True)
data_cleaned.head()


duplicate_rows = data_cleaned.duplicated()

# Count the number of duplicate rows
num_duplicates = duplicate_rows.sum()
#print(f"Number of duplicate rows: {num_duplicates}")

# Retrieve duplicate rows
duplicate_data = data_cleaned[data_cleaned.duplicated()]

# Display the first few duplicate rows
print(duplicate_data.head(20))

# Remove exact duplicates, keeping the first occurrence
data_cleaned_no_duplicates = data_cleaned.drop_duplicates()

## To scale x variables
x_numerical_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'percent_bc_gt_75', 
                       'bc_util', 'dti', 'inq_last_6mths', 'mths_since_recent_inq',
                     'revol_util', 'mths_since_last_major_derog', 'tot_hi_cred_lim', 'tot_cur_bal']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

X = data_cleaned_no_duplicates.drop('bad_flag', axis=1)
y = data_cleaned_no_duplicates['bad_flag']

# Split the data into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.25,          # 25% for Validation
    random_state=42,         # Ensures reproducibility
    stratify=y               # Maintains the distribution of 'bad_flag'
)

print(f"Training Set Size (before SMOTE): {X_train.shape[0]} samples")
print(f"Validation Set Size: {X_val.shape[0]} samples")

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the Training data
scaler.fit(X_train[x_numerical_columns])

# Transform the Training and Validation data
X_train_scaled = X_train.copy()
X_train_scaled[x_numerical_columns] = scaler.transform(X_train[x_numerical_columns])

X_val_scaled = X_val.copy()
X_val_scaled[x_numerical_columns] = scaler.transform(X_val[x_numerical_columns])

# Convert to NumPy arrays if they are not already
X_train_scaled = X_train_scaled.values.astype('float32')
X_val_scaled = X_val_scaled.values.astype('float32')
y_train = y_train.values
y_val = y_val.values

# Apply SMOTE on the training set to handle class imbalance
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

print(f"Training Set Size (after SMOTE): {X_train_resampled.shape[0]} samples")

# Convert the resampled data to tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

X_train_tensor = X_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)

# Define batch size
batch_size = 5345

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 42),
            nn.ReLU(),
            nn.Linear(42, 35),
            nn.ReLU(),
            nn.Dropout(0.4896),
            nn.Linear(35, 1)  # Output layer for binary classification
        )
        
    def forward(self, x):
        return self.network(x)

input_dim = X_train_resampled.shape[1]
model = BinaryClassificationModel(input_dim).to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0073409)

num_epochs = 100
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_val_loss += loss.item() * batch_X.size(0)
            
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())
    
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    binary_preds = (all_preds >= 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, binary_preds)
    val_accuracies.append(accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {epoch_train_loss:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f} | '
          f'Val Accuracy: {accuracy:.4f}')

# Final Evaluation on Validation Set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X)
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.cpu())
        all_labels.append(batch_y.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

binary_preds = (all_preds >= 0.5).astype(int)

accuracy = accuracy_score(all_labels, binary_preds)
roc_auc = roc_auc_score(all_labels, all_preds)
auprc = average_precision_score(all_labels, all_preds)
f1 = f1_score(all_labels, binary_preds)

print("\nValidation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot Training and Validation Loss
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.legend()
plt.show()

# Save the model (optional)
torch.save(model.state_dict(), 'binary_classification_model_smote.pth')
print("\nModel saved as 'binary_classification_model_smote.pth'.")



from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from bayes_opt import BayesianOptimization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# Define a function to create and train the model given hyperparameters
def train_and_evaluate(hn1, hn2, dr, lr, batch_size):
    hn1 = int(hn1)
    hn2 = int(hn2)
    dr = float(dr)
    lr = float(lr)
    batch_size = int(batch_size)

    class TunedBinaryClassificationModel(nn.Module):
        def __init__(self, input_dim, hn1, hn2, dr):
            super(TunedBinaryClassificationModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, hn1)
            self.fc2 = nn.Linear(hn1, hn2)
            self.fc3 = nn.Linear(hn2, 1)
            self.dropout = nn.Dropout(dr)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    model = TunedBinaryClassificationModel(input_dim, hn1, hn2, dr).to(device)

    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create DataLoaders with the given batch_size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 100  
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds)
    return roc_auc


def objective(hn1, hn2, dr, lr, batch_size):
    auc = train_and_evaluate(hn1, hn2, dr, lr, batch_size)
    return auc  # We want to maximize AUC, and bayes_opt maximizes by default

# Define search space
pbounds = {
    'hn1': (32, 128),
    'hn2': (16, 64),
    'dr': (0.0, 0.5),
    'lr': (1e-5, 1e-2),
    'batch_size': (32, 6800)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Initialize and run optimization
# init_points=10, then n_iter=40 for a total of 50 evaluations
optimizer.maximize(init_points=10, n_iter=40)

print("Best Result:", optimizer.max)
best_params = optimizer.max['params']
print("Best Hyperparameters found by Bayesian Optimization:")
print(best_params)


## Check Val's performance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

# best_params from optimizer.max['params']
best_params ={'batch_size': 5345.879758680623, 'dr': 0.48961483633324776, 
              'hn1': 42.06460657796822, 'hn2': 35.822043448436816, 'lr': 0.0073409113435169565}
best_params = optimizer.max['params']
print("Best Hyperparameters:", best_params)

hn1 = int(best_params['hn1'])
hn2 = int(best_params['hn2'])
dr = float(best_params['dr'])
lr = float(best_params['lr'])
batch_size = int(best_params['batch_size'])

# Re-create DataLoaders with best batch_size
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class FinalBinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, hn1, hn2, dr):
        super(FinalBinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hn1)
        self.fc2 = nn.Linear(hn1, hn2)
        self.fc3 = nn.Linear(hn2, 1)
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = FinalBinaryClassificationModel(input_dim, hn1, hn2, dr).to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the final model with more epochs for better performance
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_X.size(0)
    epoch_train_loss /= len(train_loader.dataset)

    # Optional: Monitor validation performance
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds)
    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val ROC AUC: {roc_auc:.4f}')

# Final evaluation on validation set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch_y.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Convert probabilities to binary predictions
binary_preds = (all_preds >= 0.5).astype(int)

# Calculate final metrics
accuracy = accuracy_score(all_labels, binary_preds)
roc_auc = roc_auc_score(all_labels, all_preds)
auprc = average_precision_score(all_labels, all_preds)
f1 = f1_score(all_labels, binary_preds)

print("\nFinal Validation Metrics with Optimized Hyperparameters:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1 Score: {f1:.4f}")
